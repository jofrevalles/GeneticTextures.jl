function update_parent_expr!(parent::Expr, idx::Int, new_expr)
    parent.args[idx] = new_expr
    return parent
end

# TODO: create a `f` type struct that contains functions that are not evaluated?
function mutate!(expr, mutation_probs, primitives_with_arity, parent::Union{Expr, Nothing}=nothing, idx::Int=0, max_mutations::Int=5)
    if max_mutations <= 0
        return expr
    end

    mutated = false
    # Mutation type 1: any node can mutate into a new random expression
    if rand() < mutation_probs[:rand_expr] && (parent === nothing || !(parent.args[1] ∈ special_funcs && idx == 2))
        f =  random_function(primitives_with_arity, depth_of_expr(expr))

        if parent !== nothing
            parent = update_parent_expr!(parent, idx, f)
            mutated = true
            return f
        else
            expr = f
        end
    end

    # Mutation type 2: adjust scalar value (constant) by adding a random amount
    if expr isa Number && rand() < mutation_probs[:adjust_scalar]
        expr += randn() * 0.1
    end

    # Mutation type 3: adjust vector value (color) by adding random amounts to each element
    if expr isa Color && rand() < mutation_probs[:adjust_color]
        expr += randn(3) * 0.1
    end

    # Mutation type 4: mutate function into a different function, adjusting arguments as necessary
    # TODO: Fix: color could be added if only the arguments do not return a color
    if expr isa Expr && rand() < mutation_probs[:rand_func] && (parent === nothing || !(parent.args[1] ∈ special_funcs && idx == 2)) && (parent === nothing || parent != :color)
        if expr.args[1] ∉ special_funcs # TODO: For now, special_funcs are excluded from this mutation type
            prim_keys = collect(keys(primitives_with_arity))
            compatible_primitives = filter(p -> primitives_with_arity[p] == length(expr.args) - 1, prim_keys)
            compatible_primitives = filter(p -> p ∉ (:grad_dir, :grad_mag), compatible_primitives) # For now, exclude grad_dir and grad_mag")

            new_func = rand(compatible_primitives)
            if new_func == :perlin_2d || new_func == :perlin_color
                seed = round(rand() * 100, digits=4)
                new_args = [new_func, seed]
                append!(new_args, expr.args[2:end])
                expr = Expr(:call, new_args...)
            else
                expr.args[1] = new_func
            end
        end
    end

    # Mutation type 5: make a node the argument of a new random function, generating other arguments randomly if necessary
    if rand() < mutation_probs[:add_argument] && (parent === nothing || !(parent.args[1] ∈ special_funcs && idx == 2)) && (parent === nothing || parent != :color)
        # Select a random function from the primitives
        compatible_primitives = filter(p -> primitives_with_arity[p] > 0, collect(keys(primitives_with_arity)))
        new_func = rand(compatible_primitives)

        # Get the arity of the selected function
        n_args = primitives_with_arity[new_func]

        if new_func == :perlin_2d || new_func == :perlin_color
            seed = round(rand() * 100, digits=4)
            args = Vector{Any}(undef, n_args-2)
            for i in 1:(n_args-2)
                args[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            insert!(args, rand(1:n_args-1), expr)

            expr = Expr(:call, new_func, seed, args...)
        elseif new_func == :grad_dir
            op = rand((x -> x[1]).(filter(x -> x.second ∈ [2] && x.first ∉ special_funcs ∪ boolean_funcs, collect(primitives_with_arity))))
            n_args_op = primitives_with_arity[op]
            args_op = Vector{Any}(undef, n_args_op)
            for i in 1:n_args_op
                args_op[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            insert!(args_op, rand(1:n_args_op), expr)

            expr = Expr(:call, new_func, op, args_op...)
        elseif new_func == :grad_mag
            op = rand((x -> x[1]).(filter(x -> x.second != 0 && x.first ∉ special_funcs ∪ boolean_funcs, collect(primitives_with_arity))))
            n_args_op = primitives_with_arity[op]
            args_op = Vector{Any}(undef, n_args_op)
            for i in 1:n_args_op
                args_op[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            insert!(args_op, rand(1:n_args_op), expr)

            expr = Expr(:call, new_func, op, args_op...)
        else            args = Vector{Any}(undef, n_args-1)
            for i in 1:(n_args-1)
                args[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            insert!(args, rand(1:n_args), expr)

            expr = Expr(:call, new_func, args...)
        end
    end

    # Mutation type 6: remove an expression from its parent function (inverse of mutation type 5)
    if expr isa Expr && rand() < mutation_probs[:become_argument]
        if expr.args[1] ∈ special_funcs
            pos = rand(3:length(expr.args)) # Exclude the first argument
        else
            pos = rand(2:length(expr.args)) # Start from 2 because the function itself is at index 1
        end
        new_expr = expr.args[pos]

        if parent !== nothing
            parent = update_parent_expr!(parent, idx, new_expr)
            mutated = true
            return new_expr
        else
            expr = new_expr
        end
    end

    # Mutation type 7: duplicate a node within the expression (like mating an expression with itself)
    # TODO: Try to implement a custom node duplication for grad_dir
    if rand() < mutation_probs[:duplicate_node] && parent !== nothing && !(parent.args[1] ∈ special_funcs && idx == 2)
        target_pos = rand(2:length(parent.args))
        if idx != target_pos
            parent = update_parent_expr!(parent, target_pos, deepcopy(expr))
            mutated = true
        end
    end

    # Recursively mutate child nodes
    if !mutated && expr isa Expr
        for i in 2:length(expr.args) # Start from 2 because the function itself is at index 1
            expr.args[i] = mutate!(expr.args[i], mutation_probs, primitives_with_arity, expr, i, max_mutations - 1)
        end
    end

    return expr
end

function mutate!(ce::CustomExpr, mutation_probs, primitives_with_arity, max_mutations::Int=5)
    mutated_expr = mutate!(ce.expr, mutation_probs, primitives_with_arity, nothing, 0, max_mutations)
    return CustomExpr(mutated_expr)
end

function mutate(e, mutation_probs, primitives_with_arity, max_mutations::Int=5)
    if e isa CustomExpr
        mutated_expr = mutate!(deepcopy(e.expr), mutation_probs, primitives_with_arity, nothing, 0, max_mutations)
    else
        mutated_expr = mutate!(deepcopy(e), mutation_probs, primitives_with_arity, nothing, 0, max_mutations)
    end
    return CustomExpr(mutated_expr)
end