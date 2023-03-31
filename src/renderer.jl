using Images, Colors
using Base.Threads

function render(expr, width, height)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    @threads for y in 1:height
        for x in 1:width
            # # Normalize coordinates to the range [0, 1]
            # nx, ny = (x - 1) / (width - 1), (y - 1) / (height - 1)

            # Normalize coordinates to the range [-0.5, 0.5]
            nx, ny = (x - 1) / (width - 1) - 0.5, (y - 1) / (height - 1) - 0.5

            # # Create an expression with the pixel coordinates as input
            # pixel_expr = substitute(expr, [:x => nx, :y => ny])

            # # Evaluate the expression for the current pixel
            # value = eval(pixel_expr)

            # # Map the result to a color (e.g., grayscale)
            # color = RGB(value, value, value)

            # # Set the pixel color in the image
            # img[y, x] = color

            # # Create an expression with the pixel coordinates as input
            # red_expr = substitute(expr, [:x => nx, :y => ny, :c => 0.0])
            # green_expr = substitute(expr, [:x => nx, :y => ny, :c => 0.5])
            # blue_expr = substitute(expr, [:x => nx, :y => ny, :c => 1.0])

            # # Evaluate the expression for the current pixel
            # red_value = clamp(eval(red_expr), 0.0, 1.0)
            # green_value = clamp(eval(green_expr), 0.0, 1.0)
            # blue_value = clamp(eval(blue_expr), 0.0, 1.0)

            # # Map the result to a color (e.g., grayscale)
            # img[y, x] = RGB(red_value, green_value, blue_value)

            # img = clean(RGB(eval(red_expr), eval(green_expr), eval(blue_expr))

            # red_expr = substitute(expr, [:x => nx, :y => ny, :c => 0.0])
            # green_expr = substitute(expr, [:x => nx, :y => ny, :c => 0.5])
            # blue_expr = substitute(expr, [:x => nx, :y => ny, :c => 1.0])


            # # Evaluate the expression for the current pixel using custom_eval
            # red_value = clamp(custom_eval(red_expr, Dict(:x => nx, :y => ny, :c => 0.0)), 0.0, 1.0)
            # green_value = clamp(custom_eval(green_expr, Dict(:x => nx, :y => ny, :c => 0.5)), 0.0, 1.0)
            # blue_value = clamp(custom_eval(blue_expr, Dict(:x => nx, :y => ny, :c => 1.0)), 0.0, 1.0)

            # Evaluate the expression for the current pixel using custom_eval
            red_value = custom_eval(expr, Dict(:x => nx, :y => ny, :c => -0.5))
            green_value = custom_eval(expr, Dict(:x => nx, :y => ny, :c => 0.5))
            blue_value = custom_eval(expr, Dict(:x => nx, :y => ny, :c => 0.5))

            if red_value === NaN
                red_value = 0.0
            end
            if green_value === NaN
                green_value = 0.0
            end
            if blue_value === NaN
                blue_value = 0.0
            end

            # Map the result to a color (e.g., grayscale)
            img[y, x] = RGB(red_value, green_value, blue_value)
        end
    end
    clean!(img)

    return img
end

# Replace all occurrences of a symbol in an expression with a given value
function substitute(expr::Expr, replacements::Array{Pair{Symbol, Float64}})
    if expr.head == :call
        # Apply substitutions recursively to the arguments of the function call
        new_args = [substitute(arg, replacements) for arg in expr.args[2:end]]
        return Expr(:call, expr.args[1], new_args...)
    elseif expr.head == :f
        # Return the symbol if it is not being replaced
        return expr
    elseif expr.head == :sym
        # Replace the symbol with its corresponding value, if present in the replacements array
        for (sym, value) in replacements
            if expr.value == sym
                return value
            end
        end
        return expr.value
    else
        throw(ArgumentError("Invalid expression type: $(expr.head)"))
    end
end

function substitute(sym::Symbol, replacements::Array{Pair{Symbol, Float64}})
    for (key, value) in replacements
        if sym == key
            return value
        end
    end
    return sym
end

function save_population(population, generation, width, height)
    for (i, expr) in enumerate(population)
        img = render(expr, width, height)
        filename = "gen_$(generation)_img_$(i).png"
        save(filename, img)
    end
end

function clean!(img)
    # normalize img to be in [0, 1]
    min_r, max_r = extrema((p -> (p.r)).(img))
    min_g, max_g = extrema((p -> (p.g)).(img))
    min_b, max_b = extrema((p -> (p.b)).(img))

    min_val, max_val = min(min_r, min_g, min_b), max(max_r, max_g, max_b)

    for y in 1:size(img, 1)
        for x in 1:size(img, 2)
            img[y, x] = RGB((img[y, x].r - min_val) / (max_val - min_val),
                            (img[y, x].g - min_val) / (max_val - min_val),
                            (img[y, x].b - min_val) / (max_val - min_val))
        end
    end

    return img
end