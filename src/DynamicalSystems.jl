using Plots
using FileIO
using Images
using Colors
using GeneticTextures: capture_function

struct DynamicalSystem
    F_A0::Union{GeneticTextures.CustomExpr, Symbol, Number, GeneticTextures.Color}
    F_B0::Union{GeneticTextures.CustomExpr, Symbol, Number, GeneticTextures.Color}
    F_dA::Union{GeneticTextures.CustomExpr, Symbol, Number, GeneticTextures.Color}
    F_dB::Union{GeneticTextures.CustomExpr, Symbol, Number, GeneticTextures.Color}
end

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


function evolve_system_step!(A, B, ds::DynamicalSystem, width, height, dt)
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

    return A, B
end

function plots_animate_system(ds::DynamicalSystem, width, height, T, dt, save_figure = false)
    # Initialize A and B using F_A and F_B
    A = zeros(height, width)
    B = zeros(height, width)
    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            A[y_pixel, x_pixel] = custom_eval(ds.F_A0, Dict(:x => x, :y => y), width, height)
            B[y_pixel, x_pixel] = custom_eval(ds.F_B0, Dict(:x => x, :y => y), width, height)
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
        write(f, "F_A0: $(string(ds.F_A0))\n")
        write(f, "F_B0: $(string(ds.F_B0))\n")
        write(f, "F_dA: $(string(ds.F_dA))\n")
        write(f, "F_dB: $(string(ds.F_dB))\n")
        write(f, "T: $T\n")
        write(f, "dt: $dt\n")
        write(f, "width: $width\n")
        write(f, "height: $height\n")
    end

    anim = @animate for (i, t) in enumerate(range(0, T, step=dt))
        # Evolve the system
        A, B = evolve_system_step!(A, B, ds, width, height, dt)

        # Create an image from current state
        img = Array{RGB{Float64}, 2}(undef, height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                r = clamp(A[y_pixel, x_pixel], 0.0, 1.0)
                g = clamp(B[y_pixel, x_pixel], 0.0, 1.0)
                img[y_pixel, x_pixel] = RGB(r, g, 0)  # we set blue = 0 for simplicity
            end
        end
        plot = heatmap(img)
        if save_figure
            savefig(plot, "$animation_dir/frame_$(lpad(i, 3, '0')).png")  # pad with zeros for consistent filename lengths
        end
        plot
    end

    gif(anim, "$animation_dir/dynamical_system.gif", fps = 15)
end

"""
    animate_system(ds::DynamicalSystem, width, height, T, dt, color_func::Function)

Animate a dynamical system by evolving it over time and saving the frames to a folder.

# Arguments
- `ds::DynamicalSystem`: The dynamical system to animate.
- `width::Int`: The width of the image in pixels.
- `height::Int`: The height of the image in pixels.
- `T::Number`: The total time to evolve the system.
- `dt::Number`: The time step to use when evolving the system.
- `color_expr::Expr`: An expr that contains a function that tells how to combine the values of A and B to create a color. e.g., `:((A, B) -> RGB(A, B, 0))`
"""
function animate_system(ds::DynamicalSystem, width, height, T, dt, color_expr::Expr)
    color_func = eval(color_expr)

    # Initialize A and B using F_A and F_B
    A = zeros(height, width)
    B = zeros(height, width)
    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            A[y_pixel, x_pixel] = custom_eval(ds.F_A0, Dict(:x => x, :y => y), width, height)
            B[y_pixel, x_pixel] = custom_eval(ds.F_B0, Dict(:x => x, :y => y), width, height)
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
        write(f, "F_A0= GeneticTextures.CustomExpr($(string(ds.F_A0)))\n")
        write(f, "F_B0= GeneticTextures.CustomExpr($(string(ds.F_B0)))\n")
        write(f, "F_dA= GeneticTextures.CustomExpr($(string(ds.F_dA)))\n")
        write(f, "F_dB= GeneticTextures.CustomExpr($(string(ds.F_dB)))\n")
        write(f, "color_func= $(capture_function(color_expr))\n")
        write(f, "T= $T\n")
        write(f, "dt= $dt\n")
        write(f, "width= $width\n")
        write(f, "height= $height\n")
    end

    image_files = []  # Store the names of the image files to use for creating the gif

    for (i, t) in enumerate(range(0, T, step=dt))
        # Evolve the system
        A, B = evolve_system_step!(A, B, ds, width, height, dt)

        # Create an image from current state
        img = Array{RGB{Float64}, 2}(undef, height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                img[y_pixel, x_pixel] = Base.invokelatest(color_func, A[y_pixel, x_pixel], B[y_pixel, x_pixel])
            end
        end

        # Save the frame
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

function animate_system_test(ds::DynamicalSystem, width, height, T, dt, color_expr::Expr)
    color_func = eval(color_expr)

    # Initialize A and B using F_A and F_B
    # A = zeros(height, width)
    # B = zeros(height, width)
    A = Array{RGB{Float64}, 2}(undef, height, width)
    B = Array{RGB{Float64}, 2}(undef, height, width)
    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            val_a = custom_eval(ds.F_A0, Dict(:x => x, :y => y), width, height)
            val_b = custom_eval(ds.F_B0, Dict(:x => x, :y => y), width, height)
            if val_a isa Color
                A[y, x] = RGB(val_a.r, val_a.g, val_a.b)
            elseif isa(rgb, Number)
                A[y, x] = RGB(val_a, val_a, val_a)
            else
                error("Invalid type output from custom_eval: $(typeof(rgb))")
            end

            if val_b isa Color
                B[y, x] = RGB(val_b.r, val_b.g, val_b.b)
            elseif isa(rgb, Number)
                B[y, x] = RGB(val_b, val_b, val_b)
            else
                error("Invalid type output from custom_eval: $(typeof(rgb))")
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
        write(f, "F_A0= GeneticTextures.CustomExpr($(string(ds.F_A0)))\n")
        write(f, "F_B0= GeneticTextures.CustomExpr($(string(ds.F_B0)))\n")
        write(f, "F_dA= GeneticTextures.CustomExpr($(string(ds.F_dA)))\n")
        write(f, "F_dB= GeneticTextures.CustomExpr($(string(ds.F_dB)))\n")
        write(f, "color_func= $(capture_function(color_expr))\n")
        write(f, "T= $T\n")
        write(f, "dt= $dt\n")
        write(f, "width= $width\n")
        write(f, "height= $height\n")
    end

    image_files = []  # Store the names of the image files to use for creating the gif

    for (i, t) in enumerate(range(0, T, step=dt))
        # Evolve the system
        A, B = evolve_system_step!(A, B, ds, width, height, dt)

        # Create an image from current state
        img = Array{RGB{Float64}, 2}(undef, height, width)
        for x_pixel in 1:width
            for y_pixel in 1:height
                img[y_pixel, x_pixel] = Base.invokelatest(color_func, A[y_pixel, x_pixel], B[y_pixel, x_pixel])
            end
        end

        # Save the frame
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

# Example usage:
# Example expressions, replace with instances of CustomExpr.
# F_A0 = GeneticTextures.CustomExpr(:(-1*ifs(rand_scalar(18.2345*(1+y*x)) < 0.93, 0., 0.6)-1.12))
# F_B0 = GeneticTextures.CustomExpr(:(-0.0345))
# F_dA = GeneticTextures.CustomExpr(:(neighbor_min(A)))
# F_dB = GeneticTextures.CustomExpr(:(laplacian(A)*1.99))
F_A0 = GeneticTextures.CustomExpr(:(or(or(xor(sin(x), y + 0.4), xor(cos(x), y + 0.5)), or(xor(sin(y), x + 0.4), xor(cos(y), x + 0.5)))))
F_B0 = GeneticTextures.CustomExpr(:(ifs(rand_scalar(18.2345 * (1.0 + y * x)) > 0.99, 0.0, 1)))
F_dA = GeneticTextures.CustomExpr(:(-1.0 * laplacian(A)*B+0.4*neighbor_min(A; Δx=2, Δy=2)))
F_dB = GeneticTextures.CustomExpr(:(-1.0 * laplacian(B)*A-0.4*neighbor_min(B)))
# F_B0 = GeneticTextures.CustomExpr(:(ifs(x^2 + y^2 < 0.1, 1.0, 0.0)))
# draw a happy face
# circle = x^2 + y^2 < 0.1 && x^2 + y^2 > 0.05
# mouth = x^2 + y^2 < 0.05 && x^2 + y^2 > 0.03 && y < 0 && x > -0.01 && x < 0.01
# eye1 = (x + 0.05)^2 + (y + 0.05)^2 < 0.01
# eye2 = (x - 0.05)^2 + (y + 0.05)^2 < 0.01
# F_A0 = GeneticTextures.CustomExpr(:(ifs(or(or(or(and((x^2 + y^2) < 0.1, (x^2 + y^2) > 0.09), and(and((x^2 + y^2) < 0.05, (x^2 + y^2) > 0.04), and(and(y > 0, x > -0.1), x < 0.1))),and((x + 0.05)^2 + (y + 0.05)^2 < 0.002 ,(x + 0.05)^2 + (y + 0.05)^2 > 0.001)),and((x - 0.05)^2 + (y + 0.05)^2 < 0.002, (x - 0.05)^2 + (y + 0.05)^2 > 0.001)), 0.0, 1.0)))
F_B0 = GeneticTextures.CustomExpr(:(ifs(and((x^2 + y^2) < 0.1, (x^2 + y^2) > 0.09), 0.0, 1.0)))
F_A0 = GeneticTextures.CustomExpr(:(ifs(rand_scalar(18.2345 * (1.0 + y * x)) > 0.99, 0.0, 1)))
color_expr = :((a, b) -> RGB(a, b, 0.0))

ds = DynamicalSystem(F_A0, F_B0, F_dA, F_dB)
# plots_animate_system(ds, 64, 64, 3.6, 0.01, false)
img = animate_system(ds, 32, 32, 2.0, 0.01, color_expr)
# display(heatmap(img))