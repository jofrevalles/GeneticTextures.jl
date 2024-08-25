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
function convert_expr(expr, custom_operations, primitives_with_arity)
    if isa(expr, Expr) && expr.head == :call
        func = expr.args[1]
        args = map(a -> convert_expr(a, custom_operations, primitives_with_arity), expr.args[2:end])
        if haskey(custom_operations, func)
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

function compile_expr(expr::Expr, custom_operations::Dict, primitives_with_arity::Dict)
    # First transform the expression to properly reference `vars`
    expr = convert_expr(expr, custom_operations, primitives_with_arity)
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
    :y => 0
)

# This dictionary indicates which functions are gradient-related and need special handling
gradient_functions = Dict(
    :grad_magnitude => true,
    :grad_direction => true
)


# Example of using this system
# Assuming custom_operations and primitives_with_arity are already defined as shown
expr = :(safe_divide(sin(100*x), sin(y)))  # Example expression
compiled = compile_expr(expr, custom_operations, primitives_with_arity)
custom_expr = CustomExpr(compiled)

w = h = 512
# Generate the image
# @time image = generate_image_refactored(custom_expr, w, h)

@time image = generate_image_refactored(custom_expr, w, h; clean=false)

# @time image = generate_image(GeneticTextures.CustomExpr(expr), w, h)
