using Images, Colors

function render(expr, width, height)
    img = Array{RGB{Float64}}(undef, height, width)

    for y in 1:height
        for x in 1:width
            # Normalize coordinates to the range [0, 1]
            nx, ny = (x - 1) / (width - 1), (y - 1) / (height - 1)

            # Create an expression with the pixel coordinates as input
            pixel_expr = substitute(expr, [:x => nx, :y => ny])

            # Evaluate the expression for the current pixel
            value = eval(pixel_expr)

            # Map the result to a color (e.g., grayscale)
            color = RGB(value, value, value)

            # Set the pixel color in the image
            img[y, x] = color
        end
    end

    return img
end

# Replace all occurrences of a symbol in an expression with a given value
function substitute(expr, replacements)
    if isa(expr, Expr)
        return Expr(expr.head, [substitute(arg, replacements) for arg in expr.args]...)
    elseif isa(expr, Symbol) && haskey(replacements, expr)
        return replacements[expr]
    else
        return expr
    end
end