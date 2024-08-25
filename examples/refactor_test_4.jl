struct CustomExpr
    func::Function
end

# Define custom operations
function safe_divide(a, b)
    isapprox(b, 0) ? 0 : a / b
end

ternary(cond, x, y) = cond ? x : y
ternary(cond::Float64, x, y) = Bool(cond) ? x : y # If cond is a float, convert the float to a boolean

using Random: seed!

function rand_scalar(args...)
    if length(args) == 0
        return rand(1) |> first
    else
        # TODO: Fix the seed fix, right now it will create a homogeneous image
        seed!(trunc(Int, args[1] * 1000))
        return rand(1) |> first
    end
end

function rand_color(args...)
    if length(args) == 0
        return Color(rand(3)...)
    else
        # TODO: Fix the seed fix, right now it will create a homogeneous image
        seed!(trunc(Int, args[1] * 1000))
        return Color(rand(3)...)
    end
end

# More custom functions here...
# Is this Dict necessary? I don't think so
custom_operations = Dict(
    :safe_divide => safe_divide,
    :ifs => ternary,
    :rand_scalar => rand_scalar,
    :rand_color => rand_color
    # add more custom operations as needed
)

# Helper to handle conversion from Expr to Julia functions
# Updated function to handle Symbols and other literals correctly
function convert_expr(expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    if isa(expr, Expr) && expr.head == :call
        func = expr.args[1]
        args = map(a -> convert_expr(a, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers), expr.args[2:end])

        if func == :perlin_2d || func == :perlin_color
            seed = expr.args[2]
            if !haskey(samplers, seed)
                samplers[seed] = perlin_2d(seed=hash(seed))  # Initialize the Perlin noise generator
            end
            sampler = samplers[seed]

            if func == :perlin_2d
                return Expr(:call, :sample_perlin_2d, sampler, args[2:end]...) # Return an expression that will perform sampling at runtime
            elseif func == :perlin_color
                return Expr(:call, :sample_perlin_color, sampler, args[2:end]...)
            end
        elseif haskey(gradient_functions, func)
            return handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
        elseif haskey(custom_operations, func)
            return Expr(:call, custom_operations[func], args...)
        else
            # TODO: Should we broadcast everything? Or only certain functions? (for now, broadcast everything)
            # return Expr(:call, broadcast(func, args...))

            # convert func to a broadcasted function

            return Expr(:call, func, args...)
            # broadcast_expr = Expr(:call, func, args...)
            # return Expr(:., broadcast_expr)
        end
    elseif isa(expr, Symbol)
        println("expr: $expr")
        if get(primitives_with_arity, expr, 1) == 0
            return :(vars[$(QuoteNode(expr))])
        elseif get(primitives_with_arity, expr, 1) == -1
            return :(vars[$(QuoteNode(expr))][(vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int, (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int])
        else
            return expr
        end
    else
        return expr
    end
end


function sample_perlin_2d(sampler, args...)
    return sample.(sampler, args...)
end

function sample_perlin_color(sampler, args...)
    offset = args[3]
    return sample.(sampler, args[1] .+ offset, args[2] .+ offset)
end


function handle_gradient_function(func, expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    args = expr.args[2:end]  # extract arguments from the expression
    kwargs = Dict()
    positional_args = []

    for arg in args
        if isa(arg, Expr) && arg.head == :parameters
            # Handle keyword arguments nested within :parameters
            for kw in arg.args
                if isa(kw, Expr) && kw.head == :kw
                    key = kw.args[1]
                    value = kw.args[2]
                    kwargs[Symbol(key)] = value  # Store kwargs to pass later
                end
            end
        elseif isa(arg, Expr) && arg.head == :kw
            # Handle keyword arguments directly
            key = arg.args[1]
            value = arg.args[2]
            kwargs[Symbol(key)] = value
        else
            # It's a positional argument, add to positional_args
            push!(positional_args, arg)
        end
    end
    println("positional_args: $positional_args, kwargs: $kwargs")

    # Convert the primary expression argument into a function if not already
    if !isempty(positional_args)
        expr_func = compile_expr(positional_args[1], custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    else
        throw(ErrorException("No positional arguments provided for the gradient function"))
    end

    # Construct a call to the proper func, incorporating kwargs correctly
    grad_expr = Expr(:call, Symbol(func), expr_func, :(vars), :(width), :(height))
    for (k, v) in kwargs
        push!(grad_expr.args, Expr(:kw, k, v))
    end

    return grad_expr
end

function compile_expr(expr::Symbol, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height, samplers)
    if get(primitives_with_arity, expr, 1) == 0
        return :(vars[$(QuoteNode(expr))])
    elseif get(primitives_with_arity, expr, 1) == -1
        return :(vars[$(QuoteNode(expr))][(vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int, (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int])
    else
        return expr
    end
end


function compile_expr(expr::Expr, custom_operations::Dict, primitives_with_arity::Dict, gradient_functions::Dict, width, height, samplers)
    # First transform the expression to properly reference `vars`
    expr = convert_expr(expr, custom_operations, primitives_with_arity, gradient_functions, width, height, samplers)
    # Now compile the transformed expression into a Julia function
    # This function explicitly requires `vars` to be passed as an argument
    return eval(:( (vars) -> $expr ))
end



# minus one means that this is a matrix?
primitives_with_arity = Dict(
    :sin => 1,
    :cos => 1,
    :tan => 1,
    :perlin_color => 2,
    :safe_divide => 2,
    :x => 0,
    :y => 0,
    :A => -1,
    :B => -1,
    :C => -1,
    :D => -1,
    :t => 0
)

function x_grad(func, vars, width, height; Δx = 1)
    x_val = vars[:x]
    Δx_scaled = Δx / (width - 1)  # scale Δx to be proportional to the image width

    # Evaluate function at x
    center_val = func(merge(vars, Dict(:x => x_val)))

    # Evaluate function at x + Δx
    x_plus_val = func(merge(vars, Dict(:x => x_val + Δx_scaled)))

    # Compute the finite difference
    grad_x = (x_plus_val - center_val) / Δx_scaled
    return grad_x
end

function y_grad(func, vars, width, height; Δy = 1)
    y_val = vars[:y]
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    # Evaluate function at y
    center_val = func(merge(vars, Dict(:y => y_val)))

    # Evaluate function at y + Δy
    y_plus_val = func(merge(vars, Dict(:y => y_val + Δy_scaled)))

    # Compute the finite difference
    grad_y = (y_plus_val - center_val) / Δy_scaled
    return grad_y
end

function grad_magnitude(expr, vars, width, height; Δx = 1, Δy = 1)
    ∂f_∂x = x_grad(expr, vars, width, height; Δx = Δx)
    ∂f_∂y = y_grad(expr, vars, width, height; Δy = Δy)
    return sqrt.(∂f_∂x .^ 2 + ∂f_∂y .^ 2)
end

function grad_direction(expr, vars, width, height; Δx = 1, Δy = 1)
    ∂f_∂x = x_grad(expr, vars, width, height; Δx = Δx)
    ∂f_∂y = y_grad(expr, vars, width, height; Δy = Δy)
    return atan.(∂f_∂y, ∂f_∂x)
end

function laplacian(func, vars, width, height; Δx = 1, Δy = 1)
    x_val = vars[:x]
    y_val = vars[:y]

    Δx_scaled = Δx / (width - 1)  # scale Δx to be proportional to the image width
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    center_val = func(merge(vars, Dict(:x => x_val, :y => y_val)))

    x_plus_val = func(merge(vars, Dict(:x => x_val + Δx_scaled, :y => y_val)))
    x_minus_val = func(merge(vars, Dict(:x => x_val - Δx_scaled, :y => y_val)))
    ∇x = (x_plus_val + x_minus_val - 2 * center_val) / Δx_scaled^2

    y_plus_val = func(merge(vars, Dict(:x => x_val, :y => y_val + Δy_scaled)))
    y_minus_val = func(merge(vars, Dict(:x => x_val, :y => y_val - Δy_scaled)))
    ∇y = (y_plus_val + y_minus_val - 2 * center_val) / Δy_scaled^2

    return ∇x + ∇y
end

# Return the smalles value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_min(expr, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]
    min_val = expr(vars)  # Directly use vars, no need to merge if x, y are already set

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Evaluate neighborhood
    for dx in -Δx:Δx, dy in -Δy:Δy
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        val = expr(temp_vars)
        if val < min_val
            min_val = val
        end
    end

    return min_val
end

# Return the largest value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_max(expr, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]

    idx_x = (vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int
    idx_y = (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int

    println("merge: $(merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)))")
    max_val = expr(vars)

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Evaluate neighborhood
    for dx in -Δx:Δx, dy in -Δy:Δy
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        val = expr(temp_vars)
        if val > max_val
            max_val = val
        end
    end

    return max_val
end

# Return the average value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_ave(expr, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]
    sum_val = expr(vars)  # Directly use vars, no need to merge if x, y are already set
    count = 1

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Evaluate neighborhood
    for dx in -Δx:Δx, dy in -Δy:Δy
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        sum_val += expr(temp_vars)
        count += 1
    end

    return sum_val / count
end

# This dictionary indicates which functions are gradient-related and need special handling
gradient_functions = Dict(
    :grad_magnitude => grad_magnitude,
    :grad_direction => grad_direction,
    :x_grad => x_grad,
    :y_grad => y_grad,
    :laplacian => laplacian,
    :neighbor_min => neighbor_min,
    :neighbor_max => neighbor_max,
    :neighbor_ave => neighbor_ave
)

perlin_functions = Dict(
    :perlin_2d => 3,
    :perlin_color => 4,
)

function generate_image_refactored(custom_expr::CustomExpr, width::Int, height::Int; clean = true)
    img = Array{RGB{Float64}, 2}(undef, height, width)

    cval = rand(w, h)
    for y in 1:height
        for x in 1:width
            vars = Dict(:x => (x - 1) / (width - 1) - 0.5, :y => (y - 1) / (height - 1) - 0.5, :D => cval)
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

w = h = 512



# Example of using this system
# Assuming custom_operations and primitives_with_arity are already defined as shown
# expr = :(safe_divide(sin(100*x), sin(y)))  # Example expression
# expr = :(laplacian(y^3))
# expr = :(neighbor_max(sin(100*x*y); Δx=4, Δy=1))
# expr = :(perlin_color(321, sin(y * Color(0.2, 0.4, 0.8))*10, x*8, x))
expr = :(sin(100*x) + neighbor_ave(D))
# expr = :(rand_color())
samplers = Dict()
compiled = compile_expr(expr, custom_operations, primitives_with_arity, gradient_functions, w, h, samplers)
custom_expr = CustomExpr(compiled)

# Generate the image
# @time image = generate_image_refactored(custom_expr, w, h)

@time image = generate_image_refactored(custom_expr, w, h; clean=true)

# @time image = generate_image(GeneticTextures.CustomExpr(expr), w, h)


# struct VariableDynamics3
#     name::Symbol
#     F_0::Union{Expr, Symbol, Number, Color}
#     δF::Union{Expr, Symbol, Number, Color}

#     function VariableDynamics3(name, F_0, δF)
#         return new(name, F_0, δF)
#     end
# end

# name(var::VariableDynamics3) = var.name
# F_0(var::VariableDynamics3) = var.F_0
# δF(var::VariableDynamics3) = var.δF

# struct DynamicalSystem3
#     dynamics::Vector{VariableDynamics3}
# end

# function evolve_system_step_2!(vals, dynamics::DynamicalSystem3, width, height, t, dt, complex_func::Function)
#     δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

#     custom_expr = [CustomExpr(compile_expr(δF(ds), custom_operations, primitives_with_arity, gradient_functions, width, height, Dict())) for ds in dynamics]

#     vars = merge(Dict(:t => t), Dict(name(ds) => vals[i] for (i, ds) in enumerate(dynamics)))

#     for x_pixel in 1:width
#         for y_pixel in 1:height
#             x = (x_pixel - 1) / (width - 1) - 0.5
#             y = (y_pixel - 1) / (height - 1) - 0.5

#             vars[:x] = x
#             vars[:y] = y

#             for (i, ds) in enumerate(dynamics)

#                 val =  dt .* invokelatest(custom_expr[i].func, vars)

#                 if val isa Color
#                     δvals[i][y_pixel, x_pixel] = val
#                 elseif isreal(val)
#                     δvals[i][y_pixel, x_pixel] = Color(val, val, val)
#                 else
#                     δvals[i][y_pixel, x_pixel] = Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
#                 end
#             end
#         end
#     end

#     # Update vals
#     for (i, ds) in enumerate(dynamics)
#         vals[i] += δvals[i]
#     end

#     return vals
# end

# function animate_system_2threaded(dynamics::DynamicalSystem3, width, height, T, dt; normalize_img = false, color_expr::Expr = :((vals...) -> RGB(sum(red.(vals))/length(vals), sum(green.(vals))/length(vals), sum(blue.(vals))/length(vals))), complex_expr::Expr = :((c) -> real(c)))
#     color_func = eval(color_expr)
#     complex_func = eval(complex_expr)


#     custom_exprs = [CustomExpr(compile_expr(F_0(ds), custom_operations, primitives_with_arity, gradient_functions, width, height, Dict()))        for ds in dynamics]

#     # Initialize each vars' grid using their F_0 expression
#     vals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]
#     t = 0
#     for x_pixel in 1:width
#         for y_pixel in 1:height
#             x = (x_pixel - 1) / (width - 1) - 0.5
#             y = (y_pixel - 1) / (height - 1) - 0.5

#             for (i, ds) in enumerate(dynamics)
#                 vars = Dict(:x => x, :y => y, :t => t)
#                 val = invokelatest(custom_exprs[i].func, vars)

#                 if val isa Color
#                     vals[i][y_pixel, x_pixel] = val
#                 else
#                     vals[i][y_pixel, x_pixel] = isreal(val) ? Color(val, val, val) : Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
#                 end
#             end
#         end
#     end

#     # Generate a unique filename
#     base_dir = "saves"
#     if !isdir(base_dir)
#         mkdir(base_dir)
#     end

#     animation_id = length(readdir(base_dir)) + 1
#     animation_dir = base_dir * "/animation_$animation_id"

#     # If the directory already exists, increment the id until we find one that doesn't
#     while isdir(animation_dir)
#         animation_id += 1
#         animation_dir = base_dir * "/animation_$animation_id"
#     end

#     mkdir(animation_dir)

#     # Save the system's expressions to a file
#     expr_file = animation_dir * "/expressions.txt"
#     open(expr_file, "w") do f
#         write(f, "Animated using 'animate_system_2' function\n")
#         for ds in dynamics
#             write(f, "$(name(ds))_0 = CustomExpr($(string(F_0(ds))))\n")
#             write(f, "δ$(name(ds))/δt = CustomExpr($(string(δF(ds))))\n")
#         end
#         write(f, "color_func= $(capture_function(color_expr))\n")
#         write(f, "complex_func= $(capture_function(complex_expr))\n")
#         write(f, "T= $T\n")
#         write(f, "dt= $dt\n")
#         write(f, "width= $width\n")
#         write(f, "height= $height\n")
#     end

#     image_files = []  # Store the names of the image files to use for creating the gif

#     for (i, t) in enumerate(range(0, T, step=dt))
#         vals = evolve_system_step_2!(vals, dynamics, width, height, t, dt, complex_func) # Evolve the system

#         # Create an image from current state
#         img = Array{RGB{Float64}, 2}(undef, height, width)
#         for x_pixel in 1:width
#             for y_pixel in 1:height
#                 values = [var[y_pixel, x_pixel] for var in vals]

#                 img[y_pixel, x_pixel] =
#                  invokelatest(color_func, [isreal(val) ? val : invokelatest(complex_func, val) for val in values]...)
#             end
#         end

#         if normalize_img
#             img = clean!(img)
#         end

#         frame_file = animation_dir * "/frame_$(lpad(i, 5, '0')).png"
#         save(frame_file, map(clamp01nan, img))

#         # Append the image file to the list
#         push!(image_files, frame_file)
#     end
#     # Create the gif
#     gif_file = animation_dir * "/animation_$animation_id.gif"
#     run(`convert -delay 4.16 -loop 0 $(image_files) $gif_file`)  # Use ImageMagick to create a GIF animation at 24 fps
#     # run(`ffmpeg -framerate 24 -pattern_type glob -i '$(animation_dir)/*.png' -r 15 -vf scale=512:-1 $gif_file`)
#     # run convert -delay 4.16 -loop 0 $(ls frame_*.png | sort -V) animation_356.gif to do it manually in the terminal

#     println("Animation saved to: $gif_file")
#     println("Frames saved to: $animation_dir")
#     println("Expressions saved to: $expr_file")
# end

# Base.length(ds::DynamicalSystem3) = length(ds.dynamics)
# Base.iterate(ds::DynamicalSystem3, state = 1) = state > length(ds.dynamics) ? nothing : (ds.dynamics[state], state + 1)


# F_A0 = :(y + 0)
# F_dA = :(neighbor_max(neighbor_max(C; Δx=4, Δy=4); Δx=4, Δy=4))

# F_B0 = :(1.0+0*x)
# F_dB = :(x_grad(C))

# F_C = :((1 - rand_scalar()*1.68+0.12) + y)
# F_dC = :(neighbor_ave(grad_direction(B * 0.25; Δx=4, Δy=4)))


# color_expr = :((a, b, c) -> RGB(abs(c.r), abs(c.g), abs(c.b)))
# complex_expr = :((c) -> real(c))

# A = VariableDynamics3(:A, F_A0, F_dA)
# B = VariableDynamics3(:B, F_B0, F_dB)
# C = VariableDynamics3(:C, F_C, F_dC)

# ds = DynamicalSystem3([A, B, C])
# w = h = 128
# animate_system_2threaded(ds, w, h, 10.0, 0.1; color_expr, complex_expr)