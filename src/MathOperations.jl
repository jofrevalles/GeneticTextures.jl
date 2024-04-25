using Random

ternary(cond, x, y) = cond ? x : y
ternary(cond::Float64, x, y) = Bool(cond) ? x : y # If cond is a float, convert the float to a boolean

threshold(x, t = 0.5) = x >= t ? 1 : 0

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

function x_grad(func, vars, width, height; Δx = 1)
    x_val = vars[:x]
    Δx_scaled = Δx / (width - 1)  # scale Δx to be proportional to the image width

    idx_x = (x_val + 0.5) * (width - 1) + 1 |> trunc |> Int

    # Evaluate function at x
    center_val = func(merge(vars, Dict(:x => x_val)))

    if idx_x == width
        x_minus_val = func(merge(vars, Dict(:x => x_val - Δx_scaled))) # Evaluate function at x - Δx
        return (center_val - x_minus_val) / Δx_scaled
    else
        x_plus_val = func(merge(vars, Dict(:x => x_val + Δx_scaled))) # Evaluate function at x + Δx
        return (x_plus_val - center_val) / Δx_scaled
    end
end

function y_grad(func, vars, width, height; Δy = 1)
    y_val = vars[:y]
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    idx_y = (y_val + 0.5) * (height - 1) + 1 |> trunc |> Int

    # Evaluate function at y
    center_val = func(merge(vars, Dict(:y => y_val)))

    # Compute the finite difference
    if idx_y == height
        y_minus = func(merge(vars, Dict(:y => y_val - Δy_scaled))) # Evaluate function at y - Δy
        return (center_val - y_minus) / Δy_scaled
    else
        y_plus_val = func(merge(vars, Dict(:y => y_val + Δy_scaled))) # Evaluate function at y + Δy
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

function laplacian(func, vars, width, height; Δx = 1, Δy = 1)
    x_val = vars[:x]
    y_val = vars[:y]

    Δx_scaled = Δx / (width - 1)  # scale Δx to be proportional to the image width
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    idx_x = (x_val + 0.5) * (width - 1) + 1 |> trunc |> Int
    idx_y = (y_val + 0.5) * (height - 1) + 1 |> trunc |> Int

    center_val = func(merge(vars, Dict(:x => x_val, :y => y_val)))

    if Δx == 0
        ∇x = 0
    else
        if idx_x > 1 && idx_x < width
            x_plus_val = func(merge(vars, Dict(:x => x_val + Δx_scaled, :y => y_val)))
            x_minus_val = func(merge(vars, Dict(:x => x_val - Δx_scaled, :y => y_val)))
            ∇x = (x_plus_val + x_minus_val - 2 * center_val) / Δx_scaled^2
        elseif idx_x == 1
            x_plus = func(merge(vars, Dict(:x => x_val + Δx_scaled, :y => y_val)))
            ∇x = (x_plus - center_val) / Δx_scaled^2
        else # idx_x == width
            x_minus = func(merge(vars, Dict(:x => x_val - Δx_scaled, :y => y_val)))
            ∇x = (center_val - x_minus) / Δx_scaled^2
        end
    end

    if Δy == 0
        ∇y = 0
    else
        if idx_y > 1 && idx_y < height
            y_plus_val = func(merge(vars, Dict(:x => x_val, :y => y_val + Δy_scaled)))
            y_minus_val = func(merge(vars, Dict(:x => x_val, :y => y_val - Δy_scaled)))
            ∇y = (y_plus_val + y_minus_val - 2 * center_val) / Δy_scaled^2
        elseif idx_y == 1
            y_plus = func(merge(vars, Dict(:x => x_val, :y => y_val + Δy_scaled)))
            ∇y = (y_plus - center_val) / Δy_scaled^2
        else # idx_y == height
            y_minus = func(merge(vars, Dict(:x => x_val, :y => y_val - Δy_scaled)))
            ∇y = (center_val - y_minus) / Δy_scaled^2
        end
    end

    return ∇x + ∇y
end

# Return the smalles value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_min(func, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]

    min_val = func(vars)  # Directly use vars, no need to merge if x, y are already set

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # if there are any Matrix in vars values, then filter the iterations
    range_x = -Δx:Δx
    range_y = -Δy:Δy


    if any([isa(v, Matrix) for v in values(vars)])
        idx_x = (x_val + 0.5) * (width - 1) + 1 |> trunc |> Int
        idx_y = (y_val + 0.5) * (height - 1) + 1 |> trunc |> Int

        # Filter the iterations that are not in the matrix
        range_x = filter(x -> 1 <= idx_x + x <= width, range_x)
        range_y = filter(y -> 1 <= idx_y + y <= height, range_y)
    end

    # Evaluate neighborhood
    for dx in range_x, dy in range_y
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        val = func(temp_vars)
        if val < min_val
            min_val = val
        end
    end

    return min_val
end

# Return the largest value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_max(func, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]

    max_val = func(vars)

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # if there are any Matrix in vars values, then filter the iterations
    range_x = -Δx:Δx
    range_y = -Δy:Δy

    if any([isa(v, Matrix) for v in values(vars)])
        idx_x = (x_val + 0.5) * (width - 1) + 1 |> trunc |> Int
        idx_y = (y_val + 0.5) * (height - 1) + 1 |> trunc |> Int

        # Filter the iterations that are not in the matrix
        range_x = filter(x -> 1 <= idx_x + x <= width, range_x)
        range_y = filter(y -> 1 <= idx_y + y <= height, range_y)
    end

    # Evaluate neighborhood
    for dx in range_x, dy in range_y
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        val = func(temp_vars)
        if val > max_val
            max_val = val
        end
    end

    return max_val
end

# Return the average value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_ave(func, vars, width, height; Δx = 1, Δy = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]

    sum_val = func(vars)
    count = 1

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # if there are any Matrix in vars values, then filter the iterations
    range_x = -Δx:Δx
    range_y = -Δy:Δy

    if any([isa(v, Matrix) for v in values(vars)])
        idx_x = (x_val + 0.5) * (width - 1) + 1 |> trunc |> Int
        idx_y = (y_val + 0.5) * (height - 1) + 1 |> trunc |> Int

        # Filter the iterations that are not in the matrix
        range_x = filter(x -> 1 <= idx_x + x <= width, range_x)
        range_y = filter(y -> 1 <= idx_y + y <= height, range_y)
    end

    # Evaluate neighborhood
    for dx in range_x, dy in range_y
        if dx == 0 && dy == 0
            continue
        end

        temp_vars[:x] = x_val + dx / (width - 1)
        temp_vars[:y] = y_val + dy / (height - 1)

        sum_val += func(temp_vars)
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