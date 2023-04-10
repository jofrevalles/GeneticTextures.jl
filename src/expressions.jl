using CoherentNoise: sample, perlin_2d
using ForwardDiff: gradient
using Base: show

const primitives_with_arity = Dict(
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :sin => 1,
    :cos => 1,
    :sinh => 1,
    :cosh => 1,
    :abs => 1,
    :sqrt => 1,
    :mod => 2,
    :perlin_2d => 2,
    :perlin_color => 4,
    :grad_dir => 3,  # grad_dir takes 3 arguments: the expression and the x, y coordinates
    :atan => 2,
    :log => 1,
    :exp => 1,
    :or => 2,
    :and => 2,
    :xor => 2,
    :x => 0,
    :y => 0,
    :rand_scalar => 0,
    :rand_vector => 0
    )

function random_expr(primitives_with_arity, max_depth; kwargs...)
    return CustomExpr(random_function(primitives_with_arity, max_depth; kwargs...))
end

function random_function(primitives_with_arity, max_depth; boolean_functions_depth_threshold = 1)
    """
    Function that creates random texture description functions using the primitive
    functions and the `Expr` type. This function should take the maximum depth of the
    expression tree as an input and return an `Expr`` object.
    """
    if max_depth == 0
        # Select a random primitive function with arity 0 (constant or variable)
        f = rand([k for (k, v) in primitives_with_arity if v == 0])

        if f == :rand_scalar
            return rand()
        elseif f == :rand_vector
            return Color(rand(3))
        else
            return f
        end
    else
        if max_depth > boolean_functions_depth_threshold # only allow boolean functions deep in the function graph
            available_funcs = [k for k in keys(primitives_with_arity) if k ∉ [:or, :xor, :and]]
        else
            available_funcs = keys(primitives_with_arity)
        end

        # Select a random primitive function
        f = rand(available_funcs)
        n_args = primitives_with_arity[f]

        #TODO: refactor the code so looks cleaner, I don't like how we are calling `rand()/5` etc
        #TODO:  make that the random numbers in the print only have 2 digits after the decimal point

        # TODO: check if this is the best way to handle perlin_2d
        if f == :perlin_2d
            args = Expr(:call, f, :x, :y)
            # f = :perlin_2d
            # n_args = 2
            # args = [:x, :y]
        elseif f == :perlin_color # TODO: maybe we would like to disable this part
            args = Expr(:call, f, :x, :y, rand()/4, rand()/4, rand()/4)

            # TODO: change this so it just has arguments that are not of type `Color`anyomore
            # Maybe we can remove that from the `primitives_with_arity` dict
            # Also maybe we can test if we want low depth of other functions inside this arguments

            # available_funcs = [k for k in keys(primitives_with_arity) if k ∉ [:perlin_color, :rand_color]]
            # f = :perlin_color
            # n_args = 4
            # args = [:x, :y, rand()/5, rand()/5]
        elseif f == :rand_scalar
            return rand()
        elseif f == :rand_vector
            return Color(rand(3))
        elseif f == :grad_dir  # ??remove the Color maker functions from primitives_with_arity
            op = rand(keys(primitives_with_arity)) #maybe disable boolean functions here?
            args = [op, [random_function(primitives_with_arity, max_depth - 1) for _ in 2:n_args]...]

            return Expr(:call, f, args...)
        else
            # Generate random arguments recursively
            args = [random_function(primitives_with_arity, max_depth - 1) for _ in 1:n_args]

            # Return the expression
            if n_args > 0
                return Expr(:call, f, args...)
            else
                return f
            end
        end
    end
end


function arity(f::Function)
    m = first(methods(f))
    return length(m.sig.parameters) - 2 # Subtract 1 for `typeof(f)` and 1 for `#self#`
end

# TODO: Modify `grad_dir` so it can take functions that have different
#       number of arguments (not just 2)
function grad_dir(f, x, y)
    """
    Compute the gradient of f and return the direction of the gradient (in radians).
    """
    g = gradient(z -> f(z[1], z[2]), [x, y])
    return atan(g[2], g[1])
end

struct CustomExpr
    expr::Expr
end

function CustomExpr(x::Union{Number, Symbol})
    return x
end

function Base.show(io::IO, c_expr::CustomExpr)
    function short_expr(expr)
        if expr.head == :call && length(expr.args) > 0
            new_args = Any[]
            for arg in expr.args
                if arg isa Expr
                    push!(new_args, short_expr(arg))
                elseif arg isa Number || arg isa Color
                    push!(new_args, round.(arg, digits=2))
                else
                    push!(new_args, arg)
                end
            end
            return Expr(expr.head, new_args...)
        end
        return expr
    end

    show(io, short_expr(c_expr.expr))
end

function custom_eval(ce::CustomExpr, vars; sampler = nothing, primitives_with_arity = primitives_with_arity)
    return custom_eval(ce.expr, vars; sampler, primitives_with_arity)
end

function custom_eval(expr, vars; sampler = nothing, primitives_with_arity = primitives_with_arity)
    if expr isa Symbol
        if primitives_with_arity[expr] == 0
            return vars[expr]
        else
            # TODO: this now currently won't work for functions defined in custom_eval, for example
            return getfield(Main, expr)  # Return the function associated with the symbol
        end
    elseif expr isa Number || expr isa Color
        return expr
    else
        # Assume expr is an Expr with head :call
        func = expr.args[1]
        args = expr.args[2:end]
        evaluated_args = custom_eval.(args, Ref(vars); sampler)

        # Check for infinite values in the arguments
        for i in eachindex(evaluated_args)
            arg = evaluated_args[i]

            if arg isa Number || arg isa Color
                mask_inf = isinf.(arg) .| isnan.(arg)
                mask_large = arg .> 1e6
                mask_small = arg .< -1e6

                new_arg = map((m, a) -> ifelse(m, 0.0, a), mask_inf, arg)
                new_arg = map((m, a) -> ifelse(m, 1.0, a), mask_large, new_arg)
                new_arg = map((m, a) -> ifelse(m, -1.0, a), mask_small, new_arg)

                if new_arg isa Number
                    evaluated_args[i] = new_arg
                else
                    evaluated_args[i] = Color(new_arg...)
                end
            end
        end

        if func == :+
            return evaluated_args[1] .+ evaluated_args[2]
        elseif func == :-
            return evaluated_args[1] .- evaluated_args[2]
        elseif func == :*
            return evaluated_args[1] .* evaluated_args[2]
        elseif func == :/
            return evaluated_args[1] ./ evaluated_args[2]
        elseif func == :sin
            return sin.(evaluated_args[1])
        elseif func == :cos
            return cos.(evaluated_args[1])
        elseif func == :sinh
            return sinh.(evaluated_args[1])
        elseif func == :cosh
            return cosh.(evaluated_args[1])
        elseif func == :abs
            return abs.(evaluated_args[1])
        elseif func == :sqrt
            return sqrt.(abs.(evaluated_args[1]))
        elseif func == :mod
            return mod.(evaluated_args[1], evaluated_args[2])
        elseif func == :log
            return log.(abs.(evaluated_args[1]))
        elseif func == :exp
            return exp.(evaluated_args[1])
        elseif func == :or
            return apply_elementwise((x, y) -> x | y, threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :and
            return apply_elementwise((x, y) -> x & y, threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :xor
            return apply_elementwise((x, y) -> xor(x, y), threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :perlin_2d
            # sampler = perlin_2d()
            return sample(sampler, evaluated_args[1], evaluated_args[2])
        elseif func == :perlin_color
            # sampler = perlin_3d()
            offset_1 = evaluated_args[3]
            offset_2 = evaluated_args[4]
            offset_3 = evaluated_args[5]
            r = sample(sampler, evaluated_args[1]+offset_1, evaluated_args[2]+offset_1)
            g = sample(sampler, evaluated_args[1]+offset_2, evaluated_args[2]+offset_1)
            b = sample(sampler, evaluated_args[1]+offset_3, evaluated_args[2]+offset_3)
            return Color(r, g, b)
        elseif func == :grad_dir
            # TODO: Modify condition so it can take functions that have different
            #       number of arguments (not just 2)

            # Define a wrapper function for custom_eval to pass to grad_dir
            wrapped_f(x, y) = custom_eval(args[1], vars; sampler)
            return grad_dir.(getfield(Main, args[1]), evaluated_args[2], evaluated_args[3])
        elseif func == :atan
            return atan.(evaluated_args[1], evaluated_args[2])
        elseif func == :Color
            return Color(evaluated_args...)
        else
            error("Unknown function: $func")
        end
    end
end

function threshold(x, t=0.5)
    return x >= t ? true : false
end

function apply_elementwise(op, args...)
    is_color = any(x -> x isa Color, args)
    result = op.(args...)
    return is_color ? Color(result) : result
end
