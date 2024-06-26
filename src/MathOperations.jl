using ForwardDiff: gradient, derivative

function threshold(x, t = 0.5)
    return x >= t ? 1 : 0
end

function apply_elementwise(op, args...)
    is_color = any(x -> x isa Color, args)
    result = op.(args...)
    return is_color ? Color(result) : result
end

function grad_dir(f, x, y)
    """
    Compute the gradient of f and return the direction of the gradient (in radians).
    """

    g = gradient(z -> f(z[1], z[2]), [x, y])
    return atan(g[2], g[1])
end

function grad_mag(f, coords::Vararg{Number})
    """
    Compute the gradient of f and return the magnitude of the gradient.
    """

    if f == log
        f = x -> log(abs(x))
    elseif f == sqrt
        f = x -> sqrt(abs(x))
    end

    g = gradient(c -> f(c...), collect(coords))
    return sqrt(sum(x^2 for x in g))
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

function laplacian(expr, vars, width, height; Δx = 1, Δy = 1)
    # Compute the Laplacian of an expression at a point (x, y)
    # by comparing the value of expr at (x, y) with its values at (x±Δ, y) and (x, y±Δ).

    idx_x = (vars[:x]+0.5) * (width-1) + 1 |> trunc |> Int
    idx_y = (vars[:y]+0.5) * (height-1) + 1 |> trunc |> Int

    center = custom_eval(expr, merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)), width, height)

    if Δx == 0
        ∇x = 0
    else
        if idx_x > 1 && idx_x < width
            vars_plus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x + Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x + Δx))
            vars_minus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x - Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x - Δx))
            x_plus = custom_eval(expr, vars_plus_Δx, width, height)
            x_minus = custom_eval(expr, vars_minus_Δx, width, height)
            ∇x = (x_plus + x_minus - 2 * center) / Δx^2
        elseif idx_x == 1
            vars_plus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x + Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x + Δx))
            x_plus = custom_eval(expr, vars_plus_Δx, width, height)
            ∇x = (x_plus.- center) / Δx^2
        else # idx_x == width
            vars_minus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x - Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x - Δx))
            x_minus = custom_eval(expr, vars_minus_Δx, width, height)
            ∇x = (x_minus - center) / Δx^2
        end
    end

    if Δy == 0
        ∇y = 0
    else
        if idx_y > 1 && idx_y < height
            vars_plus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y + Δy] : v) for (k, v) in vars), Dict(:y => idx_y + Δy))
            vars_minus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y - Δy] : v) for (k, v) in vars), Dict(:y => idx_y - Δy))
            y_plus = custom_eval(expr, vars_plus_Δy, width, height)
            y_minus = custom_eval(expr, vars_minus_Δy, width, height)
            ∇y = (y_plus + y_minus - 2 * center) / Δy^2
        elseif idx_y == 1
            vars_plus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y + Δy] : v) for (k, v) in vars), Dict(:y => idx_y + Δy))
            y_plus = custom_eval(expr, vars_plus_Δy, width, height)
            ∇y = (y_plus - center) / Δy^2
        else # idx_y == height
            vars_minus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y - Δy] : v) for (k, v) in vars), Dict(:y => idx_y - Δy))
            y_minus = custom_eval(expr, vars_minus_Δy, width, height)
            ∇y = (y_minus - center) / Δy^2
        end
    end

    return ∇x + ∇y
end

function x_grad(expr, vars, width, height; Δx = 1)
    idx_x = (vars[:x] + 0.5) * (width - 1) + 1 |> trunc |> Int
    idx_y = (vars[:y] + 0.5) * (height - 1) + 1 |> trunc |> Int

    center = custom_eval(expr, merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)), width, height)

    if Δx == 0
        return 0
    else
        if idx_x > 1 && idx_x <= width - Δx
            vars_plus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x + Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x + Δx))
            x_plus = custom_eval(expr, vars_plus_Δx, width, height)
            return (x_plus - center) / Δx
        elseif idx_x == 1
            vars_plus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x + Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x + Δx))
            x_plus = custom_eval(expr, vars_plus_Δx, width, height)
            return (x_plus - center) / Δx
        else # idx_x == width
            vars_minus_Δx = merge(Dict(k => (isa(v, Matrix) ? v[idx_x - Δx, idx_y] : v) for (k, v) in vars), Dict(:x => idx_x - Δx))
            x_minus = custom_eval(expr, vars_minus_Δx, width, height)
            return (center - x_minus) / Δx
        end
    end
end

function y_grad(expr, vars, width, height; Δy = 1)
    idx_x = (vars[:x] + 0.5) * (width - 1) + 1 |> trunc |> Int
    idx_y = (vars[:y] + 0.5) * (height - 1) + 1 |> trunc |> Int

    center = custom_eval(expr, merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)), width, height)

    if Δy == 0
        return 0
    else
        if idx_y > 1 && idx_y <= height - Δy
            vars_plus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y + Δy] : v) for (k, v) in vars), Dict(:y => idx_y + Δy))
            y_plus = custom_eval(expr, vars_plus_Δy, width, height)
            return (y_plus - center) / Δy
        elseif idx_y == 1
            vars_plus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y + Δy] : v) for (k, v) in vars), Dict(:y => idx_y + Δy))
            y_plus = custom_eval(expr, vars_plus_Δy, width, height)
            return (y_plus - center) / Δy
        else # idx_y == height
            vars_minus_Δy = merge(Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y - Δy] : v) for (k, v) in vars), Dict(:y => idx_y - Δy))
            y_minus = custom_eval(expr, vars_minus_Δy, width, height)
            return (center - y_minus) / Δy
        end
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

function neighbor_min(expr, vars, width, height; Δx = 1, Δy = 1)
    # Return the smalles value from a neighborhood of size (2Δx + 1) x (2Δy + 1)
    # around the point (x, y)

    idx_x = (vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int
    idx_y = (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int

    center = custom_eval(expr, merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)), width, height)
    min_val = center

    for i in filter(x -> x > 0 && x != idx_x && x <= width, idx_x-Δx:idx_x+Δx)
        for j in filter(y -> y > 0 && y != idx_y && y <= height, idx_y-Δy:idx_y+Δy)
            new_vars = merge(Dict(k => (isa(v, Matrix) ? v[i, j] : v) for (k, v) in vars), Dict(:x => i, :y => j))
            val = custom_eval(expr, new_vars, width, height)
            if val < min_val
                min_val = val
            end
        end
    end

    return min_val
end

function neighbor_max(expr, vars, width, height; Δx = 1, Δy = 1)
    # Return the largest value from a neighborhood of size (2Δx + 1) x (2Δy + 1)
    # around the point (x, y)

    idx_x = (vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int
    idx_y = (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int

    center = custom_eval(expr, merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)), width, height)
    max_val = center

    for i in filter(x -> x > 0 && x != idx_x && x <= width, idx_x-Δx:idx_x+Δx)
        for j in filter(y -> y > 0 && y != idx_y && y <= height, idx_y-Δy:idx_y+Δy)
            new_vars = merge(Dict(k => (isa(v, Matrix) ? v[i, j] : v) for (k, v) in vars), Dict(:x => i, :y => j))
            val = custom_eval(expr, new_vars, width, height)

            if isreal(val) && isreal(max_val)
                if val > max_val
                    max_val = val
                end
            else
                if abs(val) > abs(max_val)
                    max_val = val
                end
            end
        end
    end

    return max_val
end

function neighbor_ave(expr, vars, width, height; Δx = 1, Δy = 1)
    # Return the average from a neighborhood of size (2Δx + 1) x (2Δy + 1)
    # around the point (x, y)

    idx_x = (vars[:x] + 0.5) * (width-1) + 1 |> trunc |> Int
    idx_y = (vars[:y] + 0.5) * (height-1) + 1 |> trunc |> Int

    center = custom_eval(expr, merge(vars, Dict(k => (isa(v, Matrix) ? v[idx_x, idx_y] : v) for (k, v) in vars)), width, height)
    sum_val = center

    iterations = 1
    for i in filter(x -> x > 0 && x != idx_x && x <= width, idx_x-Δx:idx_x+Δx)
        for j in filter(y -> y > 0 && y != idx_y && y <= height, idx_y-Δy:idx_y+Δy)
            new_vars = merge(Dict(k => (isa(v, Matrix) ? v[i, j] : v) for (k, v) in vars), Dict(:x => i, :y => j))
            val = custom_eval(expr, new_vars, width, height)
            sum_val += val
            iterations += 1
        end
    end

    return sum_val / iterations
end
