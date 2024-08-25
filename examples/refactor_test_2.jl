struct CustomExpr
    func::Function
end

# Define custom operations
function safe_divide(a, b)
    isapprox(b, 0) ? 0 : a / b
end

function perlin_color(x, y, vars)
    # Assume perlin_2d is correctly set up in `vars`
    vars[:perlin](x, y)
end

# More custom functions here...
custom_operations = Dict(
    :safe_divide => safe_divide,
    # add more custom operations as needed
)

# Helper to handle conversion from Expr to Julia functions
# Updated function to handle Symbols and other literals correctly
function convert_expr(expr, custom_operations, primitives_with_arity, gradient_functions, width, height)
    if isa(expr, Expr) && expr.head == :call
        func = expr.args[1]
        # Convert each argument, now passing width and height where needed
        args = map(a -> convert_expr(a, custom_operations, primitives_with_arity, gradient_functions, width, height), expr.args[2:end])

        if haskey(gradient_functions, func)
            # Handle special gradient functions
            println("Handling gradient function: $func")
            return handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height)
        elseif haskey(custom_operations, func)
            return Expr(:call, custom_operations[func], args...)
        else
            return Expr(:call, func, args...)
        end
    elseif isa(expr, Symbol)
        if get(primitives_with_arity, expr, 1) == 0
            return :(vars[$(QuoteNode(expr))])
        else
            return expr
        end
    else
        return expr
    end
end


# function handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height)
#     args = expr.args[2:end]  # extract arguments from the expression

#     # Convert the expression argument into a function if not already
#     expr_func = compile_expr(args[1], custom_operations, primitives_with_arity, gradient_functions, width, height)

#     # Create a new expression for x_grad with the compiled function
#     grad_expr = quote
#         x_grad($expr_func, vars, width, height)
#     end

#     return grad_expr
# end

function handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height)
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
    println("positional_args: $positional_args, kwargs: $kwargs")

    # Convert the primary expression argument into a function if not already
    if !isempty(positional_args)
        expr_func = compile_expr(positional_args[1], custom_operations, primitives_with_arity, gradient_functions, width, height)
    else
        throw(ErrorException("No positional arguments provided for the gradient function"))
    end

    # Construct a call to x_grad, incorporating kwargs correctly
    grad_expr = Expr(:call, Symbol(func), expr_func, :(vars), :(width), :(height))
    for (k, v) in kwargs
        push!(grad_expr.args, Expr(:kw, k, v))
    end

    return grad_expr
end


function compile_expr(expr::Expr, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height)
    # First transform the expression to properly reference `vars`
    expr = convert_expr(expr, custom_operations, primitives_with_arity, gradient_functions, width, height)
    # Now compile the transformed expression into a Julia function
    # This function explicitly requires `vars` to be passed as an argument
    return eval(:( (vars) -> $expr ))
end

function generate_image_refactored(custom_expr::CustomExpr, width::Int, height::Int; clean = true)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    for y in 1:height
        for x in 1:width
            vars = Dict(:x => (x - 1) / (width - 1) - 0.5, :y => (y - 1) / (height - 1) - 0.5)
            # Add more variables if needed, e.g., :t => time
            rgb = custom_expr.func(vars)

            if rgb isa Color
                img[y, x] = RGB(rgb.r, rgb.g, rgb.b)
            elseif isa(rgb, Number)
                img[y, x] = RGB(rgb, rgb, rgb)
            else
                error("Invalid type output from custom_eval: $(typeof(rgb))")
            end
        end
    end

    clean && GeneticTextures.clean!(img)
    return img
end


primitives_with_arity = Dict(
    :sin => 1,
    :cos => 1,
    :tan => 1,
    :perlin_color => 2,
    :safe_divide => 2,
    :x => 0,
    :y => 0,
    :A => 0
)

# This dictionary indicates which functions are gradient-related and need special handling
gradient_functions = Dict(
    # :grad_magnitude => grad_magnitude,  # grad_magnitude function reference
    # :grad_direction => grad_direction,  # grad_direction function reference
    :x_grad => x_grad                   # x_grad function reference
)


function x_grad(func, vars, width, height; Δx = 1)
    x_val = vars[:x]
    Δx_scaled = Δx / (width - 1)  # scale Δx to be proportional to the image width

    # Evaluate function at x
    center_val = func(merge(vars, Dict(:x => x_val)))

    # Evaluate function at x + Δx
    x_plus_val = func(merge(vars, Dict(:x => x_val + Δx_scaled)))

    # Compute the finite difference
    grad_x = (x_plus_val - center_val) / Δx_scaled
    return grad_x
end

w = h = 512


# Example of using this system
# Assuming custom_operations and primitives_with_arity are already defined as shown
# expr = :(safe_divide(sin(100*x), sin(y)))  # Example expression
expr = :(x_grad(x^2))
# expr = :(2*x)
compiled = compile_expr(expr, custom_operations, primitives_with_arity, gradient_functions, w, h)
custom_expr = CustomExpr(compiled)

# Generate the image
# @time image = generate_image_refactored(custom_expr, w, h)

@time image = generate_image_refactored(custom_expr, w, h; clean=false)

# @time image = generate_image(GeneticTextures.CustomExpr(expr), w, h)
