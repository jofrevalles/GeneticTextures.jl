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

function grad_dir(f, x)
    """
    Compute the derivative of f and return the derivative
    """
    if f == log
        f = x -> log(abs(x))
    elseif f == sqrt
        f = x -> sqrt(abs(x))
    end

    g = derivative(f, x)
    return g[1]
end

function dissolve(f1, f2, weight)
    return weight * f1 + (1 - weight) * f2
end
