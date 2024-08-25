struct CustomExpr
    func::Function
end

# Define custom operations
function safe_divide(a, b)
    isapprox(b, 0) ? 0 : a / b
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

function y_grad(func, vars, width, height; Δy = 1)
    y_val = vars[:y]
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    # Evaluate function at y
    center_val = func(merge(vars, Dict(:y => y_val)))

    # Evaluate function at y + Δy
    y_plus_val = func(merge(vars, Dict(:y => y_val + Δy_scaled)))

    # Compute the finite difference
    grad_y = (y_plus_val - center_val) / Δy_scaled
    return grad_y
end

function grad_magnitude(expr, vars, width, height; Δx = 1, Δy = 1)
    ∂f_∂x = x_grad(expr, vars, width, height; Δx = Δx)
    ∂f_∂y = y_grad(expr, vars, width, height; Δy = Δy)
    return sqrt.(∂f_∂x .^ 2 + ∂f_∂y .^ 2)
end

function grad_direction(expr, vars, width, height; Δx = 1, Δy = 1)
    ∂f_∂x = x_grad(expr, vars, width, height; Δx = Δx)
    ∂f_∂y = y_grad(expr, vars, width, height; Δy = Δy)
    return atan.(∂f_∂y, ∂f_∂x)
end

function laplacian(func, vars, width, height; Δx = 1, Δy = 1)
    x_val = vars[:x]
    y_val = vars[:y]

    Δx_scaled = Δx / (width - 1)  # scale Δx to be proportional to the image width
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    center_val = func(merge(vars, Dict(:x => x_val, :y => y_val)))

    x_plus_val = func(merge(vars, Dict(:x => x_val + Δx_scaled, :y => y_val)))
    x_minus_val = func(merge(vars, Dict(:x => x_val - Δx_scaled, :y => y_val)))
    ∇x = (x_plus_val + x_minus_val - 2 * center_val) / Δx_scaled^2

    y_plus_val = func(merge(vars, Dict(:x => x_val, :y => y_val + Δy_scaled)))
    y_minus_val = func(merge(vars, Dict(:x => x_val, :y => y_val - Δy_scaled)))
    ∇y = (y_plus_val + y_minus_val - 2 * center_val) / Δy_scaled^2

    return ∇x + ∇y
end

# Return the smalles value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_min(expr, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]
    min_val = expr(vars)  # Directly use vars, no need to merge if x, y are already set

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Evaluate neighborhood
    for dx in -Δx:Δx, dy in -Δy:Δy
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        val = expr(temp_vars)
        if val < min_val
            min_val = val
        end
    end

    return min_val
end

# Return the largest value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_max(expr, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]
    max_val = expr(vars)  # Directly use vars, no need to merge if x, y are already set

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Evaluate neighborhood
    for dx in -Δx:Δx, dy in -Δy:Δy
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        val = expr(temp_vars)
        if val > max_val
            max_val = val
        end
    end

    return max_val
end

# Return the average value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_ave(expr, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]
    sum_val = expr(vars)  # Directly use vars, no need to merge if x, y are already set
    count = 1

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Evaluate neighborhood
    for dx in -Δx:Δx, dy in -Δy:Δy
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        sum_val += expr(temp_vars)
        count += 1
    end

    return sum_val / count
end

# This dictionary indicates which functions are gradient-related and need special handling
gradient_functions = Dict(
    :grad_magnitude => grad_magnitude,
    :grad_direction => grad_direction,
    :x_grad => x_grad,
    :y_grad => y_grad,
    :laplacian => laplacian,
    :neighbor_min => neighbor_min,
    :neighbor_max => neighbor_max,
    :neighbor_ave => neighbor_ave
)

perlin_functions = Dict(
    :perlin_2d => 3,
    :perlin_color => 4,
)

w = h = 512


# Example of using this system
# Assuming custom_operations and primitives_with_arity are already defined as shown
# expr = :(safe_divide(sin(100*x), sin(y)))  # Example expression
# expr = :(laplacian(y^3))
expr = :(neighbor_max(sin(100*x*y); Δx=4, Δy=1))
# expr = :(sin(100*x))
compiled = compile_expr(expr, custom_operations, primitives_with_arity, gradient_functions, w, h)
custom_expr = CustomExpr(compiled)

# Generate the image
# @time image = generate_image_refactored(custom_expr, w, h)

@time image = generate_image_refactored(custom_expr, w, h; clean=false)

# @time image = generate_image(GeneticTextures.CustomExpr(expr), w, h)
