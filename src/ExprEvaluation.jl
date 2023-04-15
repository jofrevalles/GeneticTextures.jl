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
            return apply_elementwise((x, y) -> convert(Float64, x | y), threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :and
            return apply_elementwise((x, y) -> convert(Float64, x & y), threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :xor
            return apply_elementwise((x, y) -> convert(Float64, xor(x, y)), threshold.(evaluated_args[1]), threshold.(evaluated_args[2]))
        elseif func == :perlin_2d
            return sample.(sampler, evaluated_args[1], evaluated_args[2])
        elseif func == :perlin_color
            offset = evaluated_args[3]
            return sample.(sampler, evaluated_args[1] .+ offset, evaluated_args[2] .+ offset)
        elseif func == :grad_dir
            if primitives_with_arity[args[1]] == 1
                return grad_dir.(getfield(Main, args[1]), evaluated_args[2])
            elseif primitives_with_arity[args[1]] == 2
                return grad_dir.(getfield(Main, args[1]), evaluated_args[2], evaluated_args[3])
            else
                error("Invalid number of arguments for grad_dir")
            end
        elseif func == :atan
            return atan.(evaluated_args[1], evaluated_args[2])
        elseif func == :dissolve
            return dissolve.(evaluated_args[1], evaluated_args[2], evaluated_args[3])
        elseif func == :Color
            return Color(evaluated_args...)
        else
            error("Unknown function: $func")
        end
    end
end

