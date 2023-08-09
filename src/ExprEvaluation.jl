using CoherentNoise: sample, perlin_2d
using Random: seed!

function custom_eval(ce::CustomExpr, vars, width, height; samplers = Dict(), primitives_with_arity = primitives_with_arity)
    return custom_eval(ce.expr, vars, width, height; samplers, primitives_with_arity)
end

ternary(cond, x, y) = cond ? x : y
ternary(cond::Float64, x, y) = Bool(cond) ? x : y # If cond is a float, convert the float to a boolean

function custom_eval(expr, vars, width, height; samplers = Dict(), primitives_with_arity = primitives_with_arity)
    if expr isa Symbol
        if primitives_with_arity[expr] == 0
            if vars[expr] isa Number
                return vars[expr]
            elseif vars[expr] isa Matrix
                idx_x = (vars[:x]+0.5) * (width-1) + 1 |> trunc |> Int
                idx_y = (vars[:y]+0.5) * (height-1) + 1 |> trunc |> Int
                return vars[expr][idx_x, idx_y]
            else
                throw(ArgumentError("Invalid type for variable $expr: $(typeof(vars[expr]))"))
            end
        else
            return safe_getfield(expr) # Return the function associated with the symbol
        end
    elseif expr isa Number || expr isa Color
        return expr
    else
        # Assume expr is an Expr with head :call
        func = expr.args[1]
        args = expr.args[2:end]
        evaluated_args = custom_eval.(args, Ref(vars), width, height; samplers, primitives_with_arity)

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
        elseif func == :^
            return evaluated_args[1] .^ evaluated_args[2]
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
        elseif func == :perlin_2d || func == :perlin_color
            seed = expr.args[2]
            if !haskey(samplers, seed)
                samplers[seed] = perlin_2d(seed=hash(seed))
            end
            sampler = samplers[seed]
            noise_args = evaluated_args[2:end]

            if func == :perlin_2d
                return sample.(sampler, noise_args[1], noise_args[2])
            else
                offset = noise_args[3]
                return sample.(sampler, noise_args[1] .+ offset, noise_args[2] .+ offset)
            end
        elseif func == :grad_dir
            return grad_dir.(safe_getfield(args[1]), evaluated_args[2], evaluated_args[3])
        elseif func == :grad_mag
            return grad_mag.(safe_getfield(args[1]), evaluated_args[2:end]...)
        elseif func == :blur
            return blur.(safe_getfield(args[1]), evaluated_args[2], evaluated_args[3])
        elseif func == :atan
            return atan.(evaluated_args[1], evaluated_args[2])
        elseif func == :dissolve
            return dissolve.(evaluated_args[1], evaluated_args[2], evaluated_args[3])
        elseif func == :Color
            return Color(evaluated_args...)
        elseif func == :rand_scalar
            seed!(trunc(Int, evaluated_args[1] * 1000))
            return rand(1) |> first
        elseif func == :ifs
            # TODO: maybe check the case with Color in the conditional
            return ternary.(evaluated_args[1], evaluated_args[2], evaluated_args[3])
        elseif func == :max
            return max.(evaluated_args[1], evaluated_args[2])
        elseif func == :min
            return min.(evaluated_args[1], evaluated_args[2])
        elseif func == :<
            return evaluated_args[1] .< evaluated_args[2]
        elseif func == :>
            return evaluated_args[1] .> evaluated_args[2]
        elseif func == :<=
            return evaluated_args[1] .<= evaluated_args[2]
        elseif func == :>=
            return evaluated_args[1] .>= evaluated_args[2]
        elseif func == :laplacian
            return laplacian(args[1], vars, width, height)
        elseif func == :neighbor_min
            return neighbor_min(args[1], vars, width, height)
        else
            error("Unknown function: $func")
        end
    end
end

