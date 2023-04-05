using CoherentNoise: sample, perlin_2d
using ForwardDiff: gradient

const primitives_with_arity = Dict(
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :sin => 1,
    :cos => 1,
    :sinh => 1,
    :cosh => 1,
    :abs => 1,
    :sqrt => 1,
    :mod => 2,
    :perlin_2d => 2,
    :grad_dir => 3,  # grad_dir takes 3 arguments: the expression and the x, y coordinates
    :atan => 2,
    :log => 1,
    :exp => 1,
    :or => 2,
    :and => 2,
    :xor => 2,
    :x => 0,
    :y => 0,
    :c => 0)

function random_function(primitives_with_arity, max_depth)
    """
    Function that creates random texture description functions using the primitive
    functions and the `Expr` type. This function should take the maximum depth of the
    expression tree as an input and return an `Expr`` object.
    """
    if max_depth == 0
        # Select a random primitive function with arity 0 (constant or variable)
        f = rand([k for (k, v) in primitives_with_arity if v == 0])
        return f
    else
        # Select a random primitive function
        f = rand(keys(primitives_with_arity))
        n_args = primitives_with_arity[f]
        # TODO: maybe disable perlin_2d or remove it from grad_dir function args
        # TODO: check if grad_dir works properly (and if it works with perlin_2d)

        # TODO: check if this is the best way to handle perlin_2d
        if f == :perlin_2d
            args = Expr(:call, f, :x, :y)
        else
            # Generate random arguments recursively
            args = [random_function(primitives_with_arity, max_depth - 1) for _ in 1:n_args]

            # Return the expression
            if n_args > 0
                return Expr(:call, f, args...)
            else
                return f
            end
        end
    end
end

function random_function_v2(primitives_with_arity, max_depth; boolean_functions_depth_threshold = 1)
    """
    Function that creates random texture description functions using the primitive
    functions and the `Expr` type. This function should take the maximum depth of the
    expression tree as an input and return an `Expr`` object.
    """
    if max_depth == 0
        # Select a random primitive function with arity 0 (constant or variable)
        f = rand([k for (k, v) in primitives_with_arity if v == 0])
        return f
    else
        if max_depth > boolean_functions_depth_threshold # only allow boolean functions deep in the function graph
            available_funcs = [k for k in keys(primitives_with_arity) if k ∉ [:or, :xor, :and]]
        else
            available_funcs = keys(primitives_with_arity)
        end

        # Select a random primitive function
        f = rand(available_funcs)
        n_args = primitives_with_arity[f]

        # TODO: check if this is the best way to handle perlin_2d
        if f == :perlin_2d
            args = Expr(:call, f, :x, :y)
        else
            # Generate random arguments recursively
            args = [random_function_v2(primitives_with_arity, max_depth - 1) for _ in 1:n_args]

            # Return the expression
            if n_args > 0
                return Expr(:call, f, args...)
            else
                return f
            end
        end
    end
end


function arity(f::Function)
    m = first(methods(f))
    return length(m.sig.parameters) - 2 # Subtract 1 for `typeof(f)` and 1 for `#self#`
end

function grad_dir(f, x, y)
    """
    Compute the gradient of f and return the direction of the gradient (in radians).
    """
    g = gradient(z -> f(z[1], z[2]), [x, y])
    return atan(g[2], g[1])
end

function custom_eval(expr, vars; sampler = nothing)
    if expr isa Symbol
        return vars[expr]
    elseif expr isa Number
        return expr
    else
        # Assume expr is an Expr with head :call
        func = expr.args[1]
        args = expr.args[2:end]
        evaluated_args = custom_eval.(args, Ref(vars); sampler)

        # Check for infinite values in the arguments
        for arg in evaluated_args
            if isinf(arg) || isnan(arg)
                return 0.0
            elseif arg > 1e6
                return 1.
            elseif arg < -1e6
                return -1.
            end
        end

        if func == :+
            return sum(evaluated_args)
        elseif func == :-
            return evaluated_args[1] - evaluated_args[2]
        elseif func == :*
            return prod(evaluated_args)
        elseif func == :/
            return evaluated_args[1] / evaluated_args[2]
        elseif func == :sin
            return sin(evaluated_args[1])
        elseif func == :cos
            return cos(evaluated_args[1])
        elseif func == :sinh
            return sinh(evaluated_args[1])
        elseif func == :cosh
            return cosh(evaluated_args[1])
        elseif func == :abs
            return abs(evaluated_args[1])
        elseif func == :sqrt
            return sqrt(abs(evaluated_args[1]))
        elseif func == :mod
            return mod(evaluated_args[1], evaluated_args[2])
        elseif func == :log
            return log(abs(evaluated_args[1]))
        elseif func == :exp
            return exp(evaluated_args[1])
        elseif func == :or
            return Int(threshold(evaluated_args[1]) || threshold(evaluated_args[2]))
        elseif func == :and
            return Int(threshold(evaluated_args[1]) && threshold(evaluated_args[2]))
        elseif func == :xor
            return Int(xor(threshold(evaluated_args[1]), threshold(evaluated_args[2])))
        elseif func == :perlin_2d
            # sampler = perlin_2d()
            return sample(sampler, evaluated_args[1], evaluated_args[2])
        elseif func == :grad_dir
            # Define a wrapper function for custom_eval to pass to grad_dir
            wrapped_f(x, y) = custom_eval(args[1], merge(vars, Dict(:x => x, :y => y)); sampler)
            return grad_dir(wrapped_f, evaluated_args[1], evaluated_args[2])
        elseif func == :atan
            return atan(evaluated_args[1], evaluated_args[2])
        else
            error("Unknown function: $func")
        end
    end
end

function threshold(x, t=0.5)
    return x >= t ? true : false
end