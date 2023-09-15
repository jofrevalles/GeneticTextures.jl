using FileIO
using Images
using Colors
using Base: invokelatest

struct VariableDynamics
    name::Symbol
    F_0::Union{CustomExpr, Symbol, Number, Color}
    δF::Union{CustomExpr, Symbol, Number, Color}

    function VariableDynamics(name, F_0, δF)
        return new(name, CustomExpr(F_0), CustomExpr(δF))
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

function evolve_system(ds::DynamicalSystem, width, height, T, dt)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    A = zeros(height, width)
    B = zeros(height, width)

    # Initialize A and B using F_A and F_B
    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            A[y_pixel, x_pixel] = custom_eval(ds.F_A0, Dict(:x => x, :y => y), width, height)
            B[y_pixel, x_pixel] = custom_eval(ds.F_B0, Dict(:x => x, :y => y), width, height)
        end
    end

    # Time evolution
    for t in range(0, T, step=dt)
        dA = zeros(height, width)
        dB = zeros(height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                x = (x_pixel - 1) / (width - 1) - 0.5
                y = (y_pixel - 1) / (height - 1) - 0.5

                dA[y_pixel, x_pixel] = dt * custom_eval(ds.F_dA, Dict(:x => x, :y => y, :A => A, :B => B), width, height)
                dB[y_pixel, x_pixel] = dt * custom_eval(ds.F_dB, Dict(:x => x, :y => y, :A => A, :B => B), width, height)
            end
        end

        # Update A and B
        A += dA
        B += dB
    end

    # Create final image
    for x_pixel in 1:width
        for y_pixel in 1:height
            r = clamp(A[y_pixel, x_pixel], 0.0, 1.0)
            g = clamp(B[y_pixel, x_pixel], 0.0, 1.0)
            img[y_pixel, x_pixel] = RGB(r, g, 0.0)  # we set blue = 0 for simplicity
        end
    end

    img
end


function evolve_system_step!(vars, dynamics::DynamicalSystem, width, height, t, dt)
    δvars = [zeros(height, width) for _ in 1:length(dynamics)]

    variable_dict = merge(Dict(:t => t), Dict(name(ds) => vars[i] for (i, ds) in enumerate(dynamics)))

    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            variable_dict[:x] = x
            variable_dict[:y] = y

            for (i, ds) in enumerate(dynamics)
                δvars[i][y_pixel, x_pixel] = dt * custom_eval(δF(ds), variable_dict, width, height)
            end
        end
    end

    # Update vars
    for (i, ds) in enumerate(dynamics)
        vars[i] += δvars[i]
    end

    return vars
end

function evolve_system_step_2!(vars, dynamics::DynamicalSystem, width, height, t, dt, complex_func::Function)
    δvars = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    variable_dict = merge(Dict(:t => t), Dict(name(ds) => vars[i] for (i, ds) in enumerate(dynamics)))

    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            variable_dict[:x] = x
            variable_dict[:y] = y

            for (i, ds) in enumerate(dynamics)
                val =  dt .* custom_eval(δF(ds), variable_dict, width, height)

                if val isa Color
                    δvars[i][y_pixel, x_pixel] = val
                elseif isreal(val)
                    δvars[i][y_pixel, x_pixel] = Color(val, val, val)
                else
                    δvars[i][y_pixel, x_pixel] = Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                end
            end
        end
    end

    # Update vars
    for (i, ds) in enumerate(dynamics)
        vars[i] += δvars[i]
    end

    return vars
end

"""
    animate_system(ds::DynamicalSystem, width, height, T, dt, color_func::Function)

Animate a dynamical system by evolving it over time and saving the frames to a folder.

# Arguments
- `dynamics::DynamicalSystem`: The dynamical system to animate.
- `width::Int`: The width of the image in pixels.
- `height::Int`: The height of the image in pixels.
- `T::Number`: The total time to evolve the system.
- `dt::Number`: The time step to use when evolving the system.

# Optional Arguments
- `color_expr::Expr`: An expr that contains a function that tells how to combine the values of A and B to create a color. e.g., `:((A, B) -> RGB(A, B, 0))`
- `complex_expr::Expr`: An expr that contains a function that tells how convert a complex number to a real number. e.g., `:((c) -> real(c))`
"""
function animate_system(dynamics::DynamicalSystem, width, height, T, dt; color_expr::Expr = :((vals...) -> RGB(sum(vals)/length(vals), sum(vals)/length(vals), sum(vals)/length(vals))), complex_expr::Expr = :((c) -> real(c)))
    color_func = eval(color_expr)
    complex_func = eval(complex_expr)

    # Initialize each vars' grid using their F_0 expression
    vars = [zeros(height, width) for _ in 1:length(dynamics)]
    t = 0
    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            for (i, ds) in enumerate(dynamics)
                val = custom_eval(F_0(ds), Dict(:x => x, :y => y, :t => t), width, height)
                vars[i][y_pixel, x_pixel] = isreal(val) ? val : invokelatest(complex_func, val)
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
        write(f, "Animated using 'animate_system' function\n")
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

    for (i, t) in enumerate(range(0, T, step=dt))
        # Evolve the system
        vars = evolve_system_step!(vars, dynamics, width, height, t, dt)

        # Create an image from current state
        img = Array{RGB{Float64}, 2}(undef, height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                values = [isreal(var[y_pixel, x_pixel]) ? var[y_pixel, x_pixel] : invokelatest(complex_func, var[y_pixel, x_pixel]) for var in vars]
                img[y_pixel, x_pixel] = invokelatest(color_func, values...)
            end
        end

        frame_file = animation_dir * "/frame_$(lpad(i, 5, '0')).png"
        save(frame_file, map(clamp01nan, img))

        # Append the image file to the list
        push!(image_files, frame_file)
    end
    # Create the gif
    gif_file = animation_dir * "/animation.gif"
    run(`convert -delay 4.16 -loop 0 $(image_files) $gif_file`)  # Use ImageMagick to create a GIF animation at 24 fps
    # run(`ffmpeg -framerate 24 -pattern_type glob -i '$(animation_dir)/*.png' -r 15 -vf scale=512:-1 $gif_file`)

    println("Animation saved to: $gif_file")
    println("Frames saved to: $animation_dir")
    println("Expressions saved to: $expr_file")
end

function animate_system_2(dynamics::DynamicalSystem, width, height, T, dt; normalize_img = false, color_expr::Expr = :((vals...) -> RGB(sum(red.(vals))/length(vals), sum(green.(vals))/length(vals), sum(blue.(vals))/length(vals))), complex_expr::Expr = :((c) -> real(c)))
    color_func = eval(color_expr)
    complex_func = eval(complex_expr)

    # Initialize each vars' grid using their F_0 expression
    vars = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]
    t = 0
    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            for (i, ds) in enumerate(dynamics)
                val = custom_eval(F_0(ds), Dict(:x => x, :y => y, :t => t), width, height)

                if val isa Color
                    vars[i][y_pixel, x_pixel] = val
                else
                    vars[i][y_pixel, x_pixel] = isreal(val) ? Color(val, val, val) : Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                end
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

    for (i, t) in enumerate(range(0, T, step=dt))
        vars = evolve_system_step_2!(vars, dynamics, width, height, t, dt, complex_func) # Evolve the system

        # Create an image from current state
        img = Array{RGB{Float64}, 2}(undef, height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                values = [var[y_pixel, x_pixel] for var in vars]

                img[y_pixel, x_pixel] =
                 invokelatest(color_func, [isreal(val) ? val : invokelatest(complex_func, val) for val in values]...)
            end
        end

        if normalize_img
            img = clean!(img)
        end

        frame_file = animation_dir * "/frame_$(lpad(i, 5, '0')).png"
        save(frame_file, map(clamp01nan, img))

        # Append the image file to the list
        push!(image_files, frame_file)
    end
    # Create the gif
    gif_file = animation_dir * "/animation_$animation_id.gif"
    run(`convert -delay 4.16 -loop 0 $(image_files) $gif_file`)  # Use ImageMagick to create a GIF animation at 24 fps
    # run(`ffmpeg -framerate 24 -pattern_type glob -i '$(animation_dir)/*.png' -r 15 -vf scale=512:-1 $gif_file`)
    # run convert -delay 4.16 -loop 0 $(ls frame_*.png | sort -V) animation_356.gif to do it manually in the terminal

    println("Animation saved to: $gif_file")
    println("Frames saved to: $animation_dir")
    println("Expressions saved to: $expr_file")
end