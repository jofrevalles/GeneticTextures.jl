using CoherentNoise: sample, perlin_2d

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



function arity(f::Function)
    m = first(methods(f))
    return length(m.sig.parameters) - 2 # Subtract 1 for `typeof(f)` and 1 for `#self#`
end

function custom_eval(expr, vars)
    if expr isa Symbol
        return vars[expr]
    elseif expr isa Number
        return expr
    else
        # Assume expr is an Expr with head :call
        func = expr.args[1]
        args = expr.args[2:end]
        evaluated_args = custom_eval.(args, Ref(vars))

        # Check for infinite values in the arguments
        for arg in evaluated_args
            if isinf(arg) || isnan(arg)
                return 0.0
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
        elseif func == :perlin_2d
            sampler = perlin_2d()
            return sample(sampler, evaluated_args[1], evaluated_args[2])
        else
            error("Unknown function: $func")
        end
    end
end
