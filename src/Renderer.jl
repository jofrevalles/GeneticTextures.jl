using Images, Colors
using CoherentNoise: perlin_2d

clean!(img::Matrix{Color}) = clean!(RGB.(img)) # convert img to RGB and call clean! again

# TODO: Maybe normalize each channel separately?
function clean!(img)
    # @show typeof(img), img
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

function save_image_and_expr(img::Matrix{T}, genetic_expr::GeneticExpr; folder = "saves", prefix = "images") where {T}
    # Create the folder if it doesn't exist
    !isdir(folder) && mkdir(folder)

    # Generate a unique filename
    filename = generate_unique_filename(folder, prefix)

    # Save the image to a file
    image_file = folder * "/" * filename * ".png"
    save(image_file, img)

    # Save the expression to a file
    expr_file = folder * "/" * filename * ".txt"
    open(expr_file, "w") do f
        write(f, string(genetic_expr))
    end

    println("Image saved to: $image_file")
    println("Expression saved to: $expr_file")
end

function generate_image_basic(func, width::Int, height::Int; clean = true)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    for y in 1:height
        for x in 1:width
            vars = Dict(:x => (x - 1) / (width - 1) - 0.5, :y => (y - 1) / (height - 1) - 0.5) # Add more variables if needed, e.g., :t => time
            rgb = invokelatest(func, vars)

            if rgb isa GeneticTextures.Color
                img[y, x] = RGB(rgb.r, rgb.g, rgb.b)
            elseif isa(rgb, Number)
                img[y, x] = RGB(rgb, rgb, rgb)
            else
                error("Invalid type output from evaluation: $(typeof(rgb))")
            end
        end
    end

    clean && clean!(img)
    return img
end

function generate_image_threaded(func, width::Int, height::Int; clean = true)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    Threads.@threads for y in 1:height
        for x in 1:width
            vars = Dict(:x => (x - 1) / (width - 1) - 0.5, :y => (y - 1) / (height - 1) - 0.5) # Add more variables if needed, e.g., :t => time
            rgb = invokelatest(func, vars)

            if rgb isa Color
                img[y, x] = RGB(rgb.r, rgb.g, rgb.b)
            elseif isa(rgb, Number)
                img[y, x] = RGB(rgb, rgb, rgb)
            else
                error("Invalid type output from evaluation: $(typeof(rgb))")
            end
        end
    end

    clean && clean!(img)
    return img
end

function generate_image_vectorized(func, width::Int, height::Int; clean = true)
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    img = broadcast((x, y) -> invokelatest(func, Dict(:x => x, :y => y)), X, Y)

    is_color = [r isa Color for r in img]
    img[is_color] = RGB.(img[is_color])
    img[!is_color] = RGB.(img[!is_color], img[!is_color], img[!is_color])

    clean && clean!(img)
    return img
end

function generate_image_vectorized_threaded(func, width::Int, height::Int; clean = true, n_blocks = Threads.nthreads() * 4)
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    img = Array{RGB{Float64}, 2}(undef, height, width)

    block_size = div(height, n_blocks)
    Threads.@threads for block in 1:n_blocks
        start_y = (block - 1) * block_size + 1
        end_y = min(block * block_size, height)
        X_block = X[start_y:end_y, :]
        Y_block = Y[start_y:end_y, :]

        # Vectorize within the block
        img_block = broadcast((x, y) -> invokelatest(func, Dict(:x => x, :y => y)), X_block, Y_block)

        is_color = [r isa Color for r in img_block]
        img_block[is_color] = RGB.(img_block[is_color])
        img_block[!is_color] = RGB.(img_block[!is_color], img_block[!is_color], img_block[!is_color])

        img[start_y:end_y, :] = img_block # Assign the block to the full image
    end

    clean && clean!(img)
    return img
end

# Declare global variables
global width = 128  # Default width
global height = 128 # Default height

# if geneticexpr is a number or symbol, convert it to a GeneticExpr
generate_image(geneticexpr::Union{Number, Symbol}, width::Int, height::Int; kwargs...) = generate_image(GeneticExpr(geneticexpr), width, height; kwargs...)

# TODO: Allow for complex results, add a complex_func argument
function generate_image(geneticexpr::GeneticExpr, w::Int, h::Int; clean = true, renderer = :threaded, kwargs...)
    # TODO: Find a better way to pass these arguments to the function
    global width = w
    global height = h

    func = compile_expr(geneticexpr, custom_operations, primitives_with_arity, gradient_functions, width, height) # Compile the expression

    if renderer == :basic
        return generate_image_basic(func, width, height; clean = clean)
    elseif renderer == :vectorized
        return generate_image_vectorized(func, width, height; clean = clean)
    elseif renderer == :threaded
        return generate_image_threaded(func, width, height; clean = clean)
    elseif renderer == :vectorized_threaded
        return generate_image_vectorized_threaded(func, width, height; clean = clean, kwargs...)
    else
        error("Invalid renderer: $renderer")
    end
end