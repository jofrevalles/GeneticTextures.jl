using Images, Colors
using CoherentNoise: perlin_2d
using Statistics

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
# Function to adjust the brightness of an image to a specific target value using luminosity weights
function adjust_brightness!(img::Matrix{RGB{Float64}}; target_brightness::Float64=0.71)
    # Calculate current average brightness using weighted luminosity
    current_brightness = mean([0.299*p.r + 0.587*p.g + 0.114*p.b for p in img])

    # Avoid division by zero and unnecessary adjustments
    if current_brightness > 0 && target_brightness > 0
        scale_factor = target_brightness / current_brightness
    else
        return img  # Return image as is if brightness is zero or target is zero
    end

    # Adjust each pixel's RGB values based on the scale factor
    for y in 1:size(img, 1)
        for x in 1:size(img, 2)
            r = img[y, x].r * scale_factor
            g = img[y, x].g * scale_factor
            b = img[y, x].b * scale_factor

            # Clamp values to the range [0, 1] and handle any NaNs that might appear
            img[y, x] = RGB(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
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

# TODO: Fix this if F0 isa Matrix{RGB{N0f8}}
function determine_type(ge::GeneticExpr)
    # Initialize flags for detected types
    is_color = false
    is_complex = false

    expr = ge.expr

    if expr isa Symbol
        return (false, false)

    elseif expr isa Expr
        # Handle potential type-defining function calls or constructors
        if occursin("Color(", string(expr)) || occursin("rand_color(", string(expr)) || occursin("RGB(", string(expr))
            is_color = true
        end
        if occursin("Complex(", string(expr)) || occursin("imag(", string(expr))
            is_complex = true
        end
    end

    return (is_color, is_complex)
end

function generate_image_basic(func, possible_types, width::Int, height::Int; clean = true)
    img = Array{RGB{Float64}, 2}(undef, height, width)
    vars = Dict{Symbol, Union{Float64, possible_types..., Matrix{Union{Float64, Color, ComplexF64}}}}()

    for y in 1:height
        for x in 1:width
            # vars = Dict{Symbol, Union{Float64, Matrix{Float64}}}(:x => (x - 1) / (width - 1) - 0.5, :y => (y - 1) / (height - 1) - 0.5) # Add more variables if needed, e.g., :t => time
            vars[:x] = (x - 1) / (width - 1) - 0.5
            vars[:y] = (y - 1) / (height - 1) - 0.5

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
    display(img)
    return img
end

function generate_image_threaded(func, possible_types, width::Int, height::Int; clean = true)
    img = Array{RGB{Float64}, 2}(undef, height, width)
    vars = Dict{Symbol, Union{Float64, possible_types..., Matrix{Union{Float64, Color, ComplexF64}}}}()

    Threads.@threads for y in 1:height
        for x in 1:width
            # check if there are concurrent problems here, I am sure there are
            vars[:x] = (x - 1) / (width - 1) - 0.5
            vars[:y] = (y - 1) / (height - 1) - 0.5
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
    display(img)
    return img
end

function generate_image_vectorized(func, possible_types, width::Int, height::Int; clean = true)
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    img = broadcast((x, y) -> invokelatest(func, Dict{Symbol, Union{Float64, possible_types..., Matrix{Union{Float64, Color, ComplexF64}}}}(:x => x, :y => y)), X, Y)

    output = Array{RGB{Float64}, 2}(undef, height, width)

    is_color = [r isa Color for r in img]
    output[is_color] = RGB.(red.(img[is_color]), green.(img[is_color]), blue.(img[is_color]))
    output[.!is_color] = RGB.(img[.!is_color], img[.!is_color], img[.!is_color])

    clean && clean!(output)
    display(output)
    return output
end

function generate_image_vectorized_threaded(func, possible_types, width::Int, height::Int; clean = true, n_blocks = Threads.nthreads() * 4)
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
        img_block = broadcast((x, y) -> invokelatest(func, Dict{Symbol, Union{Float64, possible_types..., Matrix{Union{Float64, Color, ComplexF64}}}}(:x => x, :y => y)), X_block, Y_block)

        is_color = [r isa Color for r in img_block]
        img_block[is_color] = RGB.(img_block[is_color])
        img_block[!is_color] = RGB.(img_block[!is_color], img_block[!is_color], img_block[!is_color])

        img[start_y:end_y, :] = img_block # Assign the block to the full image
    end

    clean && clean!(img)
    display(img)
    return img
end

# Declare global variables
global width = 128  # Default width
global height = 128 # Default height

# if geneticexpr is a number or symbol, convert it to a GeneticExpr
generate_image(geneticexpr::Union{Number, Symbol}, width::Int, height::Int; kwargs...) = generate_image(GeneticExpr(geneticexpr), width, height; kwargs...)

# TODO: Allow for complex results, add a complex_func argument
function generate_image(geneticexpr::GeneticExpr, w::Int, h::Int; clean = true, renderer = :basic, kwargs...)
    # TODO: Find a better way to pass these arguments to the function
    global width = w
    global height = h

    # Initialize vals as a vector of matrices with the appropriate type
    vals = []
    possible_types = []
    is_color, is_complex = determine_type(geneticexpr)
    if is_color
        push!(vals, Matrix{Color}(undef, height, width))

        if Matrix{Color} ∉ possible_types
            push!(possible_types, Matrix{Color})
        end
    elseif is_complex
        push!(vals, Matrix{Complex{Float64}}(undef, height, width))

        if Matrix{Complex{Float64}} ∉ possible_types
            push!(possible_types, Matrix{Complex{Float64}})
        end
    else
        push!(vals, Matrix{Float64}(undef, height, width))

        if Matrix{Float64} ∉ possible_types
            push!(possible_types, Matrix{Float64})
        end
    end

    func = compile_expr(geneticexpr, custom_operations, primitives_with_arity, gradient_functions, width, height) # Compile the expression

    if renderer == :basic
        return generate_image_basic(func, possible_types, width, height; clean = clean)
    elseif renderer == :vectorized
        return generate_image_vectorized(func, possible_types, width, height; clean = clean)
    elseif renderer == :threaded
        return generate_image_threaded(func, possible_types, width, height; clean = clean)
    elseif renderer == :vectorized_threaded
        return generate_image_vectorized_threaded(func, possible_types, width, height; clean = clean, kwargs...)
    else
        error("Invalid renderer: $renderer")
    end
end