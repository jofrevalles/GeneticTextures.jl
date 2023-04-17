using Images, Colors
using CoherentNoise: perlin_2d

function generate_image(expr, width, height)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    for y in 1:height
        for x in 1:width
            vars = Dict(:x => (x - 1) / (width - 1) - 0.5, :y => (y - 1) / (height - 1) - 0.5)
            rgb = custom_eval(expr, vars)

            if rgb isa Color
                img[y, x] = RGB(rgb.r, rgb.g, rgb.b)
            elseif isa(rgb, Number)
                img[y, x] = RGB(rgb, rgb, rgb)
            else
                error("Invalid type output from custom_eval: $(typeof(rgb))")
            end
        end
    end

    clean!(img)

    return img
end

# TODO: Maybe normalize each channel separately?
function clean!(img)
    # normalize img to be in [0, 1]
    (min_val, _), (_, max_val) = extrema((p -> (min(p.r, p.g, p.b), max(p.r, p.g, p.b))).(img))

    for y in 1:size(img, 1)
        for x in 1:size(img, 2)
            r, g, b = img[y, x].r, img[y, x].g, img[y, x].b

            # Normalize the values
            r = (r - min_val) / (max_val - min_val)
            g = (g - min_val) / (max_val - min_val)
            b = (b - min_val) / (max_val - min_val)

            # Replace NaN values with 0.0
            img[y, x] = RGB(isnan(r) ? 0.0 : r, isnan(g) ? 0.0 : g, isnan(b) ? 0.0 : b)
        end
    end

    return img
end

function save_image_and_expr(img::Matrix{T}, custom_expr::CustomExpr; folder = "saves", prefix = "images") where {T}
    # Create the folder if it doesn't exist
    if !isdir(folder)
        mkdir(folder)
    end

    # Generate a unique filename
    filename = generate_unique_filename(folder, prefix)

    # Save the image to a file
    image_file = folder * "/" * filename * ".png"
    save(image_file, img)

    # Save the expression to a file
    expr_file = folder * "/" * filename * ".txt"
    open(expr_file, "w") do f
        write(f, string(custom_expr))
    end

    println("Image saved to: $image_file")
    println("Expression saved to: $expr_file")
end
