using CoherentNoise: sample, perlin_2d
using Random: seed!

# minus one means that this is a matrix?
# TODO: Check this, maybe this is not necessary at all... Also we have another primitives_with_arity in the ExprGenerators.jl file
# primitives_with_arity = Dict(
#     :sin => 1,
#     :cos => 1,
#     :tan => 1,
#     :perlin_color => 2,
#     :safe_divide => 2,
#     :x => 0,
#     :y => 0,
#     :A => -1,
#     :B => -1,
#     :C => -1,
#     :D => -1,
#     :t => 0
# )

custom_operations = Dict(
    :ifs => ternary,
    :rand_scalar => rand_scalar,
    :rand_color => rand_color
    # add more custom operations as needed
)

# TODO: Format this function properly
function convert_expr(expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    if isa(expr, Expr) && expr.head == :call
        func = expr.args[1]
        args = map(a -> convert_expr(a, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers), expr.args[2:end])

        if func == :perlin_2d || func == :perlin_color
            seed = expr.args[2]

            if !haskey(samplers, seed)
                samplers[seed] = perlin_2d(seed=hash(seed))  # Initialize the Perlin noise generator
            end
            sampler = samplers[seed]
            if func == :perlin_2d
                return Expr(:call, :sample_perlin_2d, sampler, args[2:end]...) # Return an expression that will perform sampling at runtime
            elseif func == :perlin_color
                return Expr(:call, :sample_perlin_color, sampler, args[2:end]...)
            end
        elseif haskey(gradient_functions, func)
            return handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
        elseif haskey(custom_operations, func)
            return Expr(:call, custom_operations[func], args...)
        else
            return Expr(:call, func, args...)
        end
    elseif isa(expr, Symbol)
        if get(primitives_with_arity, expr, 1) == 0
            return :(vars[$(QuoteNode(expr))])
        elseif get(primitives_with_arity, expr, 1) == -1
            return :(vars[$(QuoteNode(expr))][(vars[:x] + 0.5) * (width-1) + 1 |> round |> Int, (vars[:y] + 0.5) * (height-1) + 1 |> round |> Int])
        else
            return expr
        end
    else
        return expr
    end
end

function sample_perlin_2d(sampler, args...)
    return sample.(sampler, args...)
end

function sample_perlin_color(sampler, args...)
    offset = args[3]
    return sample.(sampler, args[1] .+ offset, args[2] .+ offset)
end

function handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    args = expr.args[2:end]  # extract arguments from the expression
    kwargs = Dict()
    positional_args = []

    for arg in args
        if isa(arg, Expr) && arg.head == :parameters
            # Handle keyword arguments nested within :parameters
            for kw in arg.args
                if isa(kw, Expr) && kw.head == :kw
                    key = kw.args[1]
                    value = kw.args[2]
                    kwargs[Symbol(key)] = value  # Store kwargs to pass later
                end
            end
        elseif isa(arg, Expr) && arg.head == :kw
            # Handle keyword arguments directly
            key = arg.args[1]
            value = arg.args[2]
            kwargs[Symbol(key)] = value
        else
            # It's a positional argument, add to positional_args
            push!(positional_args, arg)
        end
    end

    caller_func = positional_args[1]

    # if caller_func isa Symbol, then convert it to a function + 0
    # TODO: Fix this embarrassing hack...
    if caller_func isa Symbol
        caller_func = Expr(:call, +, caller_func, 0)
    end

    # Convert the primary expression argument into a function if not already
    if !isempty(positional_args)
        expr_func = compile_expr(caller_func, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    else
        throw(ErrorException("No positional arguments provided for the gradient function"))
    end

    # Construct a call to the proper func, incorporating kwargs correctly
    grad_expr = Expr(:call, Symbol(func), expr_func, :(vars), :(width), :(height))
    for (k, v) in kwargs
        push!(grad_expr.args, Expr(:kw, k, v))
    end

    return grad_expr
end

# TODO: Do we need this function?
function compile_expr(expr::Symbol, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height, samplers)
    if get(primitives_with_arity, expr, 1) == 0
        return :(vars[$(QuoteNode(expr))])
    elseif get(primitives_with_arity, expr, 1) == -1
        return :(vars[$(QuoteNode(expr))][(vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int, (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int])
    else
        return expr
    end
end

compile_expr(expr::Union{Number, Color}, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height, samplers) =
    return expr

function compile_expr(expr::Expr, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height, samplers)
    # First transform the expression to properly reference `vars`
    expr = convert_expr(expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)

    # Now compile the transformed expression into a Julia function
    # This function explicitly requires `vars` to be passed as an argument
    return eval(:( (vars) -> $expr ))
end

compile_expr(expr::GeneticExpr, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height) =
    compile_expr(expr.expr, custom_operations, primitives_with_arity, gradient_functions, width, height, Dict())