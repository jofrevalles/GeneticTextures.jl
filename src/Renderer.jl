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
