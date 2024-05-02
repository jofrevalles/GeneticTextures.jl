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

    idx_x = (x_val + 0.5) * (width - 1) + 1 |> round |> Int

    # Evaluate function at x
    vars[:x] = x_val
    center_val = func(vars)

    if idx_x == width
        vars[:x] = x_val - Δx_scaled
        x_minus_val = func(vars) # Evaluate function at x - Δx
        return (center_val - x_minus_val) / Δx_scaled
    else
        vars[:x] = x_val + Δx_scaled
        x_plus_val = func(vars) # Evaluate function at x + Δx
        return (x_plus_val - center_val) / Δx_scaled
    end
end

function y_grad(func, vars, width, height; Δy = 1)
    y_val = vars[:y]
    Δy_scaled = Δy / (height - 1)  # scale Δy to be proportional to the image height

    idx_y = (y_val + 0.5) * (height - 1) + 1 |> round |> Int

    # Evaluate function at y
    vars[:y] = y_val
    center_val = func(vars)

    # Compute the finite difference
    if idx_y == height
        vars[:y] = y_val - Δy_scaled
        y_minus = func(vars) # Evaluate function at y - Δy
        return (center_val - y_minus) / Δy_scaled
    else
        vars[:y] = y_val + Δy_scaled
        y_plus_val = func(vars) # Evaluate function at y + Δy
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

    idx_x = (x_val + 0.5) * (width - 1) + 1 |> round |> Int
    idx_y = (y_val + 0.5) * (height - 1) + 1 |> round |> Int

    center_val = func(vars)

    if Δx == 0
        ∇x = 0
    else
        vars[:y] = y_val
        if idx_x > 1 && idx_x < width
            vars[:x] = x_val + Δx_scaled
            x_plus_val = func(vars)
            vars[:x] = x_val - Δx_scaled
            x_minus_val = func(vars)
            ∇x = (x_plus_val + x_minus_val - 2 * center_val) / Δx_scaled^2
        elseif idx_x == 1
            vars[:x] = x_val + Δx_scaled
            x_plus = func(vars)
            ∇x = (x_plus - center_val) / Δx_scaled^2
        else # idx_x == width
            vars[:x] = x_val - Δx_scaled
            x_minus = func(vars)
            ∇x = (center_val - x_minus) / Δx_scaled^2
        end
    end

    if Δy == 0
        ∇y = 0
    else
        vars[:x] = x_val
        if idx_y > 1 && idx_y < height
            vars[:y] = y_val + Δy_scaled
            y_plus_val = func(vars)
            vars[:y] = y_val - Δy_scaled
            y_minus_val = func(vars)
            ∇y = (y_plus_val + y_minus_val - 2 * center_val) / Δy_scaled^2
        elseif idx_y == 1
            vars[:y] = y_val + Δy_scaled
            y_plus = func(vars)
            ∇y = (y_plus - center_val) / Δy_scaled^2
        else # idx_y == height
            vars[:y] = y_val - Δy_scaled
            y_minus = func(vars)
            ∇y = (center_val - y_minus) / Δy_scaled^2
        end
    end

    return ∇x + ∇y
end

# Return the smalles value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_min(func, vars, width, height; Δx = 1, Δy = 1)
    # Extract x and y values directly
    x_val = vars[:x]
    y_val = vars[:y]

    # Pre-calculate positions for x and y in the array/matrix
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize min_val
    min_val = func(vars)

    # Define the ranges, ensuring they stay within bounds
    min_x = max(1, idx_x - Δx)
    max_x = min(width, idx_x + Δx)
    min_y = max(1, idx_y - Δy)
    max_y = min(height, idx_y + Δy)

    # Calculate adjusted ranges to avoid division by zero in the loop
    range_x = (min_x:max_x) .- idx_x
    range_y = (min_y:max_y) .- idx_y

    # Loop through the neighborhood
    @inbounds for dx in range_x, dy in range_y
        if dx == 0 && dy == 0
            continue
        end

        # Adjust the temp_vars for each iteration
        vars[:x] = x_val + dx / (width - 1)
        vars[:y] = y_val + dy / (height - 1)

        # Evaluate the function and update min_val
        val = func(vars)
        if val < min_val
            min_val = val
        end
    end

    return min_val
end

# Return the minimum value from a neighborhood of radius Δr around the point (x, y)
function neighbor_min_radius(func, vars, width, height; Δr = 1)
    # Initialize the center values
    x_val = vars[:x]
    y_val = vars[:y]

    min_val = func(vars)  # Directly use vars, no need to merge if x, y are already set

    # Temporary variables to avoid repeated dictionary updates
    temp_vars = copy(vars)

    # Calculate pixel indices for the center point
    idx_x = (x_val + 0.5) * (width - 1) + 1 |> round |> Int
    idx_y = (y_val + 0.5) * (height - 1) + 1 |> round |> Int

    # Evaluate within a circular neighborhood
    for dx in -Δr:Δr, dy in -Δr:Δr
        if dx^2 + dy^2 <= Δr^2  # Check if the point (dx, dy) is within the circular radius
            new_x = idx_x + dx
            new_y = idx_y + dy

            if 1 <= new_x <= width && 1 <= new_y <= height  # Check if the indices are within image boundaries
                temp_vars[:x] = (new_x - 1) / (width - 1) - 0.5
                temp_vars[:y] = (new_y - 1) / (height - 1) - 0.5

                val = func(temp_vars)
                if val < min_val
                    min_val = val
                end
            end
        end
    end

    return min_val
end

function neighbor_max(func, vars, width, height; Δx = 1, Δy = 1)
    # Extract x and y values directly
    x_val = vars[:x]
    y_val = vars[:y]

    # Pre-calculate positions for x and y in the array/matrix
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize max_val
    max_val = func(vars)

    # Define the ranges, ensuring they stay within bounds
    min_x = max(1, idx_x - Δx)
    max_x = min(width, idx_x + Δx)
    min_y = max(1, idx_y - Δy)
    max_y = min(height, idx_y + Δy)

    # Calculate adjusted ranges to avoid division by zero in the loop
    range_x = (min_x:max_x) .- idx_x
    range_y = (min_y:max_y) .- idx_y

    # Loop through the neighborhood
    @inbounds for dx in range_x, dy in range_y
        if dx == 0 && dy == 0
            continue
        end

        # Adjust the temp_vars for each iteration
        vars[:x] = x_val + dx / (width - 1)
        vars[:y] = y_val + dy / (height - 1)

        # Evaluate the function and update max_val
        val = func(vars)
        if val > max_val
            max_val = val
        end
    end

    return max_val
end

# Return the average value from a neighborhood of size (2Δx + 1) x (2Δy + 1) around the point (x, y)
function neighbor_ave(func, vars, width, height; Δx = 1, Δy = 1)
    # Extract x and y values directly
    x_val = vars[:x]
    y_val = vars[:y]

    # Pre-calculate positions for x and y in the array/matrix
    idx_x = round(Int, (x_val + 0.5) * (width - 1) + 1)
    idx_y = round(Int, (y_val + 0.5) * (height - 1) + 1)

    # Initialize sum and count
    sum_val = func(vars)
    count = 1

    # Define the ranges, ensuring they stay within bounds
    min_x = max(1, idx_x - Δx)
    max_x = min(width, idx_x + Δx)
    min_y = max(1, idx_y - Δy)
    max_y = min(height, idx_y + Δy)

    # Calculate adjusted ranges to avoid division by zero in the loop
    range_x = (min_x:max_x) .- idx_x
    range_y = (min_y:max_y) .- idx_y

    # Loop through the neighborhood
    @inbounds for dx in range_x, dy in range_y
        if dx == 0 && dy == 0
            continue
        end

        # Adjust the temp_vars for each iteration
        vars[:x] = x_val + dx / (width - 1)
        vars[:y] = y_val + dy / (height - 1)

        # Add the function evaluation to sum_val
        sum_val += func(vars)
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
    :neighbor_min_radius => neighbor_min_radius,
    :neighbor_max => neighbor_max,
    :neighbor_ave => neighbor_ave
)