using ForwardDiff: gradient

function threshold(x, t = 0.5)
    return x >= t ? true : false
end

function apply_elementwise(op, args...)
    is_color = any(x -> x isa Color, args)
    result = op.(args...)
    return is_color ? Color(result) : result
end

# TODO: Modify `grad_dir` so it can take functions that have different
#       number of arguments (not just 2)
function grad_dir(f, x, y)
    """
    Compute the gradient of f and return the direction of the gradient (in radians).
    """
    g = gradient(z -> f(z[1], z[2]), [x, y])
    return atan(g[2], g[1])
end