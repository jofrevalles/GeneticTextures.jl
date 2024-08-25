using FileIO
using Images
using Colors
using ProgressMeter

struct VariableDynamics
    name::Symbol
    F_0::Union{GeneticExpr, Symbol, Number, Matrix{RGB{N0f8}}, Color}
    δF::Union{GeneticExpr, Symbol, Number, Color}
    image::Union{Matrix{RGB{N0f8}}, Nothing}

    function VariableDynamics(name, F_0, δF)
        if F_0 isa Matrix{RGB{N0f8}}
            return new(name, GeneticExpr(:(0.0+0.0)), GeneticExpr(δF), F_0)
        else
            return new(name, GeneticExpr(F_0), GeneticExpr(δF), nothing)
        end
    end
end

name(var::VariableDynamics) = var.name
F_0(var::VariableDynamics) = var.F_0
δF(var::VariableDynamics) = var.δF

struct DynamicalSystem
    dynamics::Vector{VariableDynamics}
end

Base.length(ds::DynamicalSystem) = length(ds.dynamics)
Base.iterate(ds::DynamicalSystem, state = 1) = state > length(ds.dynamics) ? nothing : (ds.dynamics[state], state + 1)

function evolve_system!(vals, dynamics::DynamicalSystem, genetic_funcs, possible_types, w, h, t, dt, complex_func::Function; renderer = :threaded, kwargs...)
    # TODO: Find a better way to pass these arguments to the function
    global width = w
    global height = h

    if renderer == :basic
        return evolve_system_basic!(vals, dynamics, genetic_funcs, possible_types, width, height, t, dt, complex_func)
    elseif renderer == :threaded
        return evolve_system_threaded!(vals, dynamics, genetic_funcs, width, height, t, dt, complex_func)
    elseif renderer == :vectorized
        return evolve_system_vectorized!(vals, dynamics, genetic_funcs, width, height, t, dt, complex_func)
    elseif renderer == :threadandvectorized
        return evolve_system_vectorized_threaded!(vals, dynamics, genetic_funcs, width, height, t, dt, complex_func; kwargs...)
    else
        error("Invalid renderer: $renderer")
    end
end

contains_color_code(ge::GeneticExpr) = contains_color_code(ge.expr)

# Check method's source code for color-specific logic (this is just a placeholder)
contains_color_code(e::Expr) = occursin("Color(", string(e))

contains_complex_code(ge::GeneticExpr) = contains_complex_code(ge.expr)

# Check if the expression involves Complex number logic
# TODO: This will have problems for functions that f(Real) -> Complex ... RETHINK this problem
#   maybe in this case we can force the functions to behave like f(Real) -> complex_func(Complex)
contains_complex_code(e::Expr) = occursin("Complex(", string(e))


using ExprTools

# TODO: Fix this if F0 isa Matrix{RGB{N0f8}}
function determine_type(variable::VariableDynamics, dynamics::DynamicalSystem, checked=Set{Symbol}())
    # Initialize flags for detected types
    is_color = false
    is_complex = false

    # Recursive helper function to analyze expressions
    function recurse_expr(expr)
        if expr isa Symbol
            # Prevent infinite recursion for cycles in dynamics
            if expr in checked
                return (false, false)
            end
            push!(checked, expr)

            # Find and analyze the dynamic expressions associated with the symbol
            for dyn in dynamics
                if dyn.name == expr
                    color_f0, complex_f0 = recurse_expr(F_0(dyn).expr)
                    color_δf, complex_δf = recurse_expr(δF(dyn).expr)
                    return (color_f0 || color_δf, complex_f0 || complex_δf)
                end
            end

            return (false, false)  # Default if the symbol doesn't match any dynamic
        elseif expr isa Expr
            # Handle potential type-defining function calls or constructors
            if occursin("Color(", string(expr)) || occursin("rand_color(", string(expr)) || occursin("RGB(", string(expr))
                is_color = true
            end
            if occursin("Complex(", string(expr)) || occursin("imag(", string(expr))
                is_complex = true
            end

            # Analyze each component of the expression recursively
            for arg in expr.args
                color, complex = recurse_expr(arg)
                is_color |= color
                is_complex |= complex
            end
        end

        return (is_color, is_complex)
    end

    # Check both F_0 and δF expressions of the given variable
    if variable.image !== nothing
        is_color_F0, is_complex_F0 = (true, false)
    else
        is_color_F0, is_complex_F0 = recurse_expr(F_0(variable).expr)
    end
    is_color_δF, is_complex_δF = recurse_expr(δF(variable).expr)

    # Combine results from both initial and dynamic function checks
    return (is_color_F0 || is_color_δF, is_complex_F0 || is_complex_δF)
end

function animate_system(dynamics::DynamicalSystem, width, height, T, dt; normalize_img = false, adjust_brighness = true, plot = true, renderer = :threaded, color_expr::Expr = :((vals...) -> RGB(sum(red.(vals))/length(vals), sum(green.(vals))/length(vals), sum(blue.(vals))/length(vals))), complex_expr::Expr = :((c) -> real(c)))
    color_func = eval(color_expr)
    complex_func = eval(complex_expr)

    init_funcs = [compile_expr(F_0(ds), custom_operations, primitives_with_arity, gradient_functions, width, height) for ds in dynamics]

    # IMPORTANT TODO: Initialize `vals` as Matrix{Float64} instead, and only convert to Color at the end
    # we could also inspect the expr of F_0 and δF to determine if the result is a color or not
    #   vals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)] # Initialize each vars' grid using their F_0 expression

    # vals = []
    # for i in 1:length(dynamics)
    #     if contains_color_code(F_0(dynamics.dynamics[i]))  # Assuming dynamics[i].F_0 is the function or its expression
    #         push!(vals, Matrix{Color}(undef, height, width))
    #     elseif contains_complex_code(F_0(dynamics.dynamics[i]))
    #         push!(vals, Matrix{Complex{Float64}}(undef, height, width))
    #     else
    #         push!(vals, Matrix{Float64}(undef, height, width))
    #     end
    # end

    # Initialize vals as a vector of matrices with the appropriate type
    return_types = [determine_type(ds, dynamics) for ds in dynamics]
    vals = []
    possible_types = []
    for (is_color, is_complex) in return_types
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
    end
    t = 0.

    # TODO: The type of vars Dict should also be determined by the expressions
    # vars = Dict{Symbol, Union{Float64, Matrix{Float64}}}()
    # The Matrix{Union{Float64, Color, ComplexF64}} is needed for the cache_computed_values function... This is sad, RETHINK. But maybe this is not a big overhead
    vars = Dict{Symbol, Union{Float64, possible_types..., Matrix{Union{Float64, Color, ComplexF64}}}}()
    images = [ds.image === nothing ? ds.image : imresize(ds.image, (height, width)) for ds in dynamics]

    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            for i in 1:length(dynamics)
                if (images[i] !== nothing)
                    vals[i][y_pixel, x_pixel] = Color(images[i][y_pixel, x_pixel])
                else
                    vars[:x] = x
                    vars[:y] = y
                    vars[:t] = t
                    val = invokelatest(init_funcs[i], vars)

                    vals[i][y_pixel, x_pixel] = val
                end

                # if val isa Color
                #     vals[i][y_pixel, x_pixel] = val
                # else
                #     vals[i][y_pixel, x_pixel] = isreal(val) ? Color(val, val, val) : Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                # end
            end
        end
    end

    # Generate a unique filename
    base_dir = "saves"
    if !isdir(base_dir)
        mkdir(base_dir)
    end

    animation_id = length(readdir(base_dir)) + 1
    animation_dir = base_dir * "/animation_$animation_id"

    # If the directory already exists, increment the id until we find one that doesn't
    while isdir(animation_dir)
        animation_id += 1
        animation_dir = base_dir * "/animation_$animation_id"
    end

    mkdir(animation_dir)

    # Save the system's expressions to a file
    expr_file = animation_dir * "/expressions.txt"
    open(expr_file, "w") do f
        write(f, "Animated using 'animate_system_2' function\n")
        for ds in dynamics
            write(f, "$(name(ds))_0 = CustomExpr($(string(F_0(ds))))\n")
            write(f, "δ$(name(ds))/δt = CustomExpr($(string(δF(ds))))\n")
        end
        write(f, "color_func= $(capture_function(color_expr))\n")
        write(f, "complex_func= $(capture_function(complex_expr))\n")
        write(f, "T= $T\n")
        write(f, "dt= $dt\n")
        write(f, "width= $width\n")
        write(f, "height= $height\n")
    end

    image_files = []  # Store the names of the image files to use for creating the gif

    # We only need to compile once each expression
    genetic_funcs = [compile_expr(δF(ds), custom_operations, primitives_with_arity, gradient_functions, width, height) for ds in dynamics]

    total_frames = ceil(Int, T / dt)
    progress = Progress(total_frames, desc="Initializing everything...", barlen=80)

    # Save the initial state
    img = Array{RGB{Float64}, 2}(undef, height, width)
    for x_pixel in 1:width
        for y_pixel in 1:height
            values = [var[y_pixel, x_pixel] for var in vals]

            # convert values to color
            values = Color.(values)

            img[y_pixel, x_pixel] =
                invokelatest(color_func, [isreal(val) ? val : invokelatest(complex_func, val) for val in values]...)
        end
    end
    normalize_img && clean!(img) # Clean the image if requested
    adjust_brighness && adjust_brightness!(img) # Adjust the brightness if requested

    plot && display(img)

    frame_file = animation_dir * "/frame_$(lpad(0, 5, '0')).png"
    save(frame_file, map(clamp01nan, img))
    push!(image_files, frame_file) # Append the image file to the list

    # Evolve the system over time
    start_time = time()
    for (i, t) in enumerate(range(0, T, step=dt))
        vals = evolve_system!(vals, dynamics, genetic_funcs, possible_types, width, height, t, dt, complex_func; renderer = renderer) # Evolve the system

        # Create an image from current state
        img = Array{RGB{Float64}, 2}(undef, height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                values = [var[y_pixel, x_pixel] for var in vals]

                # convert values to color
                values = Color.(values)

                img[y_pixel, x_pixel] =
                 invokelatest(color_func, [isreal(val) ? val : invokelatest(complex_func, val) for val in values]...)
            end
        end

        normalize_img && clean!(img) # Clean the image if requested
        adjust_brighness && adjust_brightness!(img) # Adjust the brightness if requested

        plot && display(img)

        frame_file = animation_dir * "/frame_$(lpad(i, 5, '0')).png"
        save(frame_file, map(clamp01nan, img))
        push!(image_files, frame_file) # Append the image file to the list

        elapsed_time = time() - start_time
        avg_time_per_frame = elapsed_time / i
        remaining_time = avg_time_per_frame * (total_frames - i)

        ProgressMeter.update!(progress, i, desc="Processing Frame $i: Avg time per frame $(round(avg_time_per_frame, digits=2))s, Remaining $(round(remaining_time, digits=2))s")
    end

    # Create the gif
    println("Creating GIF...")
    gif_file = animation_dir * "/animation_$animation_id.gif"
    run(`convert -delay 4.16 -loop 0 $(image_files) $gif_file`)  # Use ImageMagick to create a GIF animation at 24 fps

    println("Animation saved to: $gif_file")
    println("Frames saved to: $animation_dir")
    println("Expressions saved to: $expr_file")
end

function evolve_system_basic!(vals, dynamics::DynamicalSystem, genetic_funcs, possible_types, width, height, t, dt, complex_func::Function)
    # δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]
    # PROBLEM HERE! This will error for coupled variables that one have Color or Complex and the others
    #   don't. We should rethink this or add a smarter way to check if there will be Color or Complex
    # δvals = []
    # for i in 1:length(dynamics)
    #     if contains_color_code(δF(dynamics.dynamics[i]))  # Assuming dynamics[i].F_0 is the function or its expression
    #         push!(δvals, Matrix{Color}(undef, height, width))
    #     elseif contains_complex_code(δF(dynamics.dynamics[i]))
    #         push!(δvals, Matrix{Complex{Float64}}(undef, height, width))
    #     else
    #         push!(δvals, Matrix{Float64}(undef, height, width))
    #     end
    # end
    # δvals = [Matrix{Float64}(undef, height, width) for _ in 1:length(dynamics)]

    vars = Dict{Symbol, Union{Float64, possible_types..., Matrix{Union{Float64, Color, ComplexF64}}}}(name(ds) => vals[i] for (i, ds) in enumerate(dynamics))
    vars[:t] = t
    # vars[:possible_types] = possible_types

    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            for i in 1:length(dynamics)
                vars[:x] = x
                vars[:y] = y
                val =  dt .* invokelatest(genetic_funcs[i], vars)

                vals[i][y_pixel, x_pixel] += val

                # δvals[i][y_pixel, x_pixel] = val

                # if val isa Color
                #     δvals[i][y_pixel, x_pixel] = val
                # elseif isreal(val)
                #     δvals[i][y_pixel, x_pixel] = Color(val, val, val)
                # else
                #     δvals[i][y_pixel, x_pixel] = Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                # end
            end
        end
    end

    # # Update vals
    # for i in 1:length(dynamics)
    #     vals[i] += δvals[i]
    # end

    return vals
end

# TODO: Check if using threads may lead to unexpected results when there are random number generators involved
function evolve_system_threaded!(vals, dynamics, genetic_funcs, width, height, t, dt, complex_func)
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]
    vars = Dict{Symbol, Union{Float64, Matrix{Float64}}}(name(ds) => vals[i] for (i, ds) in enumerate(dynamics))
    vars[:t] = t

    Threads.@threads for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            for i in 1:length(dynamics)
                val = dt .* invokelatest(genetic_funcs[i], merge(vars, Dict(:x => x, :y => y)))

                if val isa Color
                    δvals[i][y_pixel, x_pixel] = val
                elseif isreal(val)
                    δvals[i][y_pixel, x_pixel] = Color(val, val, val)
                else
                    δvals[i][y_pixel, x_pixel] = Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                end
            end
        end
    end

    # Update vals
    for i in 1:length(dynamics)
        vals[i] += δvals[i]
    end

    return vals
end

function vectorize_color_decision!(results, δvals, complex_func, i)
    is_color = [r isa Color for r in results] # Determine the type of each element in results
    is_real_and_not_color = [isreal(r) && !(r isa Color) for r in results]

    # Assign directly where result elements are Color
    δvals[i][is_color] = results[is_color]

    # Where the results are real numbers, create Color objects
    real_vals = results[is_real_and_not_color]
    δvals[i][is_real_and_not_color] = Color.(real_vals, real_vals, real_vals)

    # For remaining cases, apply the complex function and create Color objects
    needs_complex = .!(is_color .| is_real_and_not_color)
    complex_results = results[needs_complex]
    processed_complex = complex_func.(complex_results)
    δvals[i][needs_complex] = Color.(processed_complex, processed_complex, processed_complex)
end


function evolve_system_vectorized!(vals, dynamics::DynamicalSystem, genetic_funcs, width, height, t, dt, complex_func::Function)
    # Precompute coordinate grids
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    # Prepare δvals to accumulate changes
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    vars = Dict{Symbol, Union{Float64, Matrix{Float64}}}(name(ds) => vals[i] for (i, ds) in enumerate(dynamics))
    vars[:t] = t

    # Loop through each dynamical system
    for i in 1:length(dynamics)
        # Evaluate the function for all pixels in a vectorized manner
        result = broadcast((x, y) -> dt .* invokelatest(genetic_funcs[i], merge(vars, Dict(:x => x, :y => y))), X, Y)

        # After obtaining results for each dynamic system
        vectorize_color_decision!(result, δvals, complex_func, i)
    end

    # Update vals
    for i in 1:length(dynamics)
        vals[i] += δvals[i]
    end

    return vals
end

function evolve_system_vectorized_threaded!(vals, dynamics::DynamicalSystem, genetic_funcs, width, height, t, dt, complex_func::Function; n_blocks = Threads.nthreads() * 4)
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    # Compute coordinate grids once
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    vars = Dict{Symbol, Union{Float64, Matrix{Float64}}}(name(ds) => vals[i] for (i, ds) in enumerate(dynamics))
    vars[:t] = t

    # Multithreading across either columns or blocks of columns
    Threads.@threads for block in 1:n_blocks
        x_start = 1 + (block - 1) * Int(width / n_blocks)
        x_end = block * Int(width / n_blocks)
        X_block = X[:, x_start:x_end]
        Y_block = Y[:, x_start:x_end]
        δvals_block = [Matrix{Color}(undef, height, x_end - x_start + 1) for _ in 1:length(dynamics)]

        # Vectorized computation within each block
        for i in 1:length(dynamics)
            result_block = broadcast((x, y) -> dt .* invokelatest(genetic_funcs[i], merge(vars, Dict(:x => x, :y => y))), X_block, Y_block)

            # Use a vectorized color decision
            vectorize_color_decision!(result_block, δvals_block, complex_func, i)
        end

        # Update the global δvals with the block's results
        for i in 1:length(dynamics)
            δvals[i][:, x_start:x_end] .= δvals_block[i]
        end
    end

    # Update vals
    for i in 1:length(dynamics)
        vals[i] += δvals[i]
    end

    return vals
end