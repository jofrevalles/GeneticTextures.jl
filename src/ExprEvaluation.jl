using CoherentNoise: sample, perlin_2d

function custom_eval(ce::CustomExpr, vars; sampler = nothing, primitives_with_arity = primitives_with_arity)
    return custom_eval(ce.expr, vars; sampler, primitives_with_arity)
end

function custom_eval(expr, vars; sampler = nothing, primitives_with_arity = primitives_with_arity)
    if expr isa Symbol
        if primitives_with_arity[expr] == 0
            return vars[expr]
        else
            # TODO: this now currently won't work for functions defined in custom_eval, for example
            return getfield(Main, expr)  # Return the function associated with the symbol
        end
    elseif expr isa Number || expr isa Color
        return expr
    else
        # Assume expr is an Expr with head :call
        func = expr.args[1]
        args = expr.args[2:end]
        evaluated_args = custom_eval.(args, Ref(vars); sampler)

        # Check for infinite values in the arguments
        for i in eachindex(evaluated_args)
            arg = evaluated_args[i]

            if arg isa Number || arg isa Color
                mask_inf = isinf.(arg) .| isnan.(arg)
                mask_large = arg .> 1e6
                mask_small = arg .< -1e6

                new_arg = map((m, a) -> ifelse(m, 0.0, a), mask_inf, arg)
                new_arg = map((m, a) -> ifelse(m, 1.0, a), mask_large, new_arg)
                new_arg = map((m, a) -> ifelse(m, -1.0, a), mask_small, new_arg)

                if new_arg isa Number
                    evaluated_args[i] = new_arg
                else
                    evaluated_args[i] = Color(new_arg...)
                end
            end
        end

        if func == :+
            return evaluated_args[1] .+ evaluated_args[2]
        elseif func == :-
            return evaluated_args[1] .- evaluated_args[2]
        elseif func == :*
            return evaluated_args[1] .* evaluated_args[2]
        elseif func == :/
            return evaluated_args[1] ./ evaluated_args[2]
        elseif func == :sin
            return sin.(evaluated_args[1])
        elseif func == :cos
            return cos.(evaluated_args[1])
        elseif func == :sinh
            return sinh.(evaluated_args[1])
        elseif func == :cosh
            return cosh.(evaluated_args[1])
        elseif func == :abs
            return abs.(evaluated_args[1])
        elseif func == :sqrt
            return sqrt.(abs.(evaluated_args[1]))
        elseif func == :mod
            return mod.(evaluated_args[1], evaluated_args[2])
        elseif func == :log
            return log.(abs.(evaluated_args[1]))
        elseif func == :exp
            return exp.(evaluated_args[1])
        elseif func == :or
            return apply_elementwise((x, y) -> x | y, threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :and
            return apply_elementwise((x, y) -> x & y, threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :xor
            return apply_elementwise((x, y) -> xor(x, y), threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :perlin_2d
            # sampler = perlin_2d()
            return sample(sampler, evaluated_args[1], evaluated_args[2])
        elseif func == :perlin_color
            # sampler = perlin_3d()
            offset_1 = evaluated_args[3]
            offset_2 = evaluated_args[4]
            offset_3 = evaluated_args[5]
            r = sample(sampler, evaluated_args[1]+offset_1, evaluated_args[2]+offset_1)
            g = sample(sampler, evaluated_args[1]+offset_2, evaluated_args[2]+offset_1)
            b = sample(sampler, evaluated_args[1]+offset_3, evaluated_args[2]+offset_3)
            return Color(r, g, b)
        elseif func == :grad_dir
            # TODO: Modify condition so it can take functions that have different
            #       number of arguments (not just 2)

            # Define a wrapper function for custom_eval to pass to grad_dir
            wrapped_f(x, y) = custom_eval(args[1], vars; sampler)
            return grad_dir.(getfield(Main, args[1]), evaluated_args[2], evaluated_args[3])
        elseif func == :atan
            return atan.(evaluated_args[1], evaluated_args[2])
        elseif func == :Color
            return Color(evaluated_args...)
        else
            error("Unknown function: $func")
        end
    end
end
