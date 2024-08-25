using Random

ternary(cond, x, y) = cond ? x : y
ternary(cond::Float64, x, y) = Bool(cond) ? x : y # If cond is a float, convert the float to a boolean

threshold(x, t = 0.5) = x >= t ? 1 : 0

and(x::Number, y::Number) = convert(Float64, threshold(x) & threshold(y))
or(x::Number, y::Number) = convert(Float64, threshold(x) | threshold(y))
xor(x::Number, y::Number) = convert(Float64, threshold(x) ⊻ threshold(y))
ifs(cond::Number, x::Number, y::Number) =  ternary(cond, x, y)
and(x::Color, y::Color) = and.(x, y)
or(x::Color, y::Color) = or.(x, y)
xor(x::Color, y::Color) = xor.(x, y)
ifs(cond::Color, x::Color, y::Color) =  ternary.(cond, x, y)

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

function apply_elementwise(op, args...)
    is_color = any(x -> x isa Color, args)
    result = op.(args...)
    return is_color ? Color(result) : result
end

function dissolve(f1, f2, weight)
    return weight * f1 + (1 - weight) * f2
end

function blur(f, x, y, matrix=gaussian_kernel(5, 1.0), increment=0.01)
    n_rows, n_cols = size(matrix)
    result = 0.0
    total_weight = sum(matrix)

    # Calculate the offsets for the center of the blurring matrix
    center_row_offset = (n_rows - 1) / 2
    center_col_offset = (n_cols - 1) / 2

    for row in 1:n_rows
        for col in 1:n_cols
            # Get the weight for the current position in the blurring matrix
            weight = matrix[row, col]

            # Calculate the coordinates to sample from with a smaller increment
            sample_x = x - center_col_offset * increment + (col - 1) * increment
            sample_y = y - center_row_offset * increment + (row - 1) * increment

            # Add the weighted value of the input function
            result += weight * f(sample_x, sample_y)
        end
    end
    return result / total_weight
end

function gaussian_kernel(size, sigma)
    kernel = zeros(size, size)
    center = (size + 1) / 2

    for i in 1:size
        for j in 1:size
            x_offset = i - center
            y_offset = j - center
            kernel[i, j] = exp(-(x_offset^2 + y_offset^2) / (2 * sigma^2))
        end
    end

    # Normalize the kernel
    kernel = kernel ./ sum(kernel)
    return kernel
end

# TODO: Remove kwargs? Right now only works for Δx = 1
function x_grad(func, vars, width, height; Δx = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)
    Δx_scaled = Δx / (width - 1)

    center_val = computed_values[idx_y, idx_x]

    if idx_x == width
        x_minus_val = computed_values[idx_y, idx_x - Δx]
        return (center_val - x_minus_val) / Δx_scaled
    else
        x_plus_val = computed_values[idx_y, idx_x + Δx]
        return (x_plus_val - center_val) / Δx_scaled
    end
end

# TODO: Remove kwargs? Right now only works for Δy = 1
function y_grad(func, vars, width, height; Δy = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)
    Δy_scaled = Δy / (height - 1)

    center_val = computed_values[idx_y, idx_x]

    if idx_y == height
        y_minus_val = computed_values[idx_y - Δy, idx_x]
        return (center_val - y_minus_val) / Δy_scaled
    else
        y_plus_val = computed_values[idx_y + Δy, idx_x]
        return (y_plus_val - center_val) / Δy_scaled
    end
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

# TODO: Remove kwargs? Right now only works for Δx = Δy = 1
function laplacian(func, vars, width, height; Δx = 1, Δy = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    Δx_scaled = Δx / (width - 1)  # Scale Δx to be proportional to the image width
    Δy_scaled = Δy / (height - 1)  # Scale Δy to be proportional to the image height

    # Handle boundary conditions for Laplacian calculation
    # Ensuring not to go out of bounds with @inbounds when accessing the array
    center_val = computed_values[idx_y, idx_x]
    ∇x, ∇y = 0.0, 0.0

    if Δx > 0 && idx_x > 1 && idx_x < width
        x_plus_val = computed_values[idx_y, idx_x + Δx]
        x_minus_val = computed_values[idx_y, idx_x - Δx]
        ∇x = (x_plus_val + x_minus_val - 2 * center_val) / Δx_scaled^2
    end

    if Δy > 0 && idx_y > 1 && idx_y < height
        y_plus_val = computed_values[idx_y + Δy, idx_x]
        y_minus_val = computed_values[idx_y - Δy, idx_x]
        ∇y = (y_plus_val + y_minus_val - 2 * center_val) / Δy_scaled^2
    end

    return ∇x + ∇y
end

# Return the smallest value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_min(func, vars, width, height; Δx = 1, Δy = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize min_val using the value at the central point
    min_val = computed_values[idx_y, idx_x]

    # Define the ranges, ensuring they stay within bounds
    min_x = max(1, idx_x - Δx)
    max_x = min(width, idx_x + Δx)
    min_y = max(1, idx_y - Δy)
    max_y = min(height, idx_y + Δy)

    # Loop through the neighborhood
    @inbounds for dy in min_y:max_y, dx in min_x:max_x
        # Avoid considering the center again
        if dx == idx_x && dy == idx_y
            continue
        end
        val = computed_values[dy, dx]
        if val < min_val
            min_val = val
        end
    end

    return min_val
end

# Return the minimum value from a neighborhood of radius Δr around the point (x, y)
function neighbor_min_radius(func, vars, width, height; Δr = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize min_val using the center point value from the cached data
    min_val = computed_values[idx_y, idx_x]

    # Evaluate within a circular neighborhood using cached data
    for dx in -Δr:Δr, dy in -Δr:Δr
        if dx^2 + dy^2 <= Δr^2  # Check if the point (dx, dy) is within the circular radius
            new_x = idx_x + dx
            new_y = idx_y + dy

            if 1 <= new_x <= width && 1 <= new_y <= height  # Check if the indices are within image boundaries
                val = computed_values[new_y, new_x]
                if val < min_val
                    min_val = val
                end
            end
        end
    end

    return min_val
end

# Return the largest value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_max(func, vars, width, height; Δx = 1, Δy = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize max_val using the value at the central point
    max_val = computed_values[idx_y, idx_x]

    # Define the ranges, ensuring they stay within bounds
    min_x = max(1, idx_x - Δx)
    max_x = min(width, idx_x + Δx)
    min_y = max(1, idx_y - Δy)
    max_y = min(height, idx_y + Δy)

    # Loop through the neighborhood
    @inbounds for dy in min_y:max_y, dx in min_x:max_x
        # Avoid considering the center again
        if dx == idx_x && dy == idx_y
            continue
        end
        val = computed_values[dy, dx]
        if val > max_val
            max_val = val
        end
    end

    return max_val
end

# Return the maximum value from a neighborhood of radius Δr around the point (x, y)
function neighbor_max_radius(func, vars, width, height; Δr = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize max_val using the center point value from the cached data
    max_val = computed_values[idx_y, idx_x]

    # Evaluate within a circular neighborhood using cached data
    for dx in -Δr:Δr, dy in -Δr:Δr
        if dx^2 + dy^2 <= Δr^2  # Check if the point (dx, dy) is within the circular radius
            new_x = idx_x + dx
            new_y = idx_y + dy

            if 1 <= new_x <= width && 1 <= new_y <= height  # Check if the indices are within image boundaries
                val = computed_values[new_y, new_x]
                if val > max_val
                    max_val = val
                end
            end
        end
    end

    return max_val
end

function cache_computed_values!(func, width::Int, height::Int, vars)
    func_key = Symbol(string(func))  # Create a unique key based on function name
    if !haskey(vars, func_key)
        # computed_values = Matrix{Float64}(undef, height, width)
        computed_values = Matrix{Union{Float64, Color, ComplexF64}}(undef, height, width)
        for y in 1:height
            for x in 1:width
                vars[:x] = (x - 1) / (width - 1) - 0.5
                vars[:y] = (y - 1) / (height - 1) - 0.5
                val = func(vars)
                computed_values[y, x] = val
                # if val isa Color
                #     computed_values[y, x] = val
                # else
                #     computed_values[y, x] = Color(val, val, val)
                # end
            end
        end
        vars[func_key] = computed_values  # Store the computed values in vars
    end
end

function neighbor_ave(func, vars, width, height; Δx = 1, Δy = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    min_x, max_x = max(1, idx_x - Δx), min(width, idx_x + Δx)
    min_y, max_y = max(1, idx_y - Δy), min(height, idx_y + Δy)
    sum_val = 0.0
    count = 0

    @inbounds for dy in min_y:max_y, dx in min_x:max_x
        sum_val += computed_values[dy, dx]
        count += 1
    end

    return sum_val / count
end

# Return the average value from a neighborhood of radius Δr around the point (x, y)
function neighbor_ave_radius(func, vars, width, height; Δr = 1)
    func_key = Symbol(string(func))
    if !haskey(vars, func_key)
        cache_computed_values!(func, width, height, vars)  # Ensure values are cached
    end

    computed_values = vars[func_key]
    x_val, y_val = vars[:x], vars[:y]
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    sum_val = 0.0
    count = 0

    for dx in -Δr:Δr, dy in -Δr:Δr
        if dx^2 + dy^2 <= Δr^2  # Check if the point (dx, dy) is within the circular radius
            new_x = idx_x + dx
            new_y = idx_y + dy

            if 1 <= new_x <= width && 1 <= new_y <= height  # Check if the indices are within image boundaries
                sum_val += computed_values[new_y, new_x]
                count += 1
            end
        end
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
    :neighbor_min_radius => neighbor_min_radius,
    :neighbor_max => neighbor_max,
    :neighbor_max_radius => neighbor_max_radius,
    :neighbor_ave => neighbor_ave,
    :neighbor_ave_radius => neighbor_ave_radius
)