function update_parent_expr!(parent::Expr, idx::Int, new_expr)
    parent.args[idx] = new_expr
    return parent
end

function mutate!(expr, mutation_probs, primitives_with_arity, parent::Union{Expr, Nothing}=nothing, idx::Int=0)
    # TODO: if mutation_probs is large, maybe apply a depth_factor to the probabilities
    #       so that deeper expressions are less likely to be mutated so
    #       that the tree doesn't get too big and find recursive problems

    mutated = false
    # Mutation type 1: mutate node into a new random expression
    if rand() < mutation_probs[:new_rand_expr]
        println("Mutation type 1")
        f =  random_function(primitives_with_arity, depth_of_expr(expr))

        if parent !== nothing
            parent = update_parent_expr!(parent, idx, f)
            mutated = true
            return f
        else
            expr = f
        end
    end

    # Mutation type 2: adjust scalar value (constant)
    if expr isa Number && rand() < mutation_probs[:adjust_scalar]
        println("Mutation type 2")
        expr += randn() * 0.1
    end

    # Mutation type 3: adjust vector value (color)
    if expr isa Color && rand() < mutation_probs[:adjust_color]
        println("Mutation type 3")
        expr += randn(3) * 0.1
    end

    # Mutation type 4: mutate function into a different function
    if expr isa Expr && rand() < mutation_probs[:change_func]
        println("Mutation type 4")
        prim_keys = collect(keys(primitives_with_arity))
        compatible_primitives = filter(p -> primitives_with_arity[p] == length(expr.args) - 1, prim_keys)

        # Exclude grad_dir from being changed to a different function
        compatible_primitives = filter(p -> p != :grad_dir, compatible_primitives)

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

    # Mutation type 5: make a node the argument of a new random function
    if rand() < mutation_probs[:add_branch] && (parent === nothing || !(parent.args[1] == :grad_dir && idx == 2))
        println("Mutation type 5")

        # Select a random function from the primitives
        compatible_primitives = filter(p -> primitives_with_arity[p] > 0, collect(keys(primitives_with_arity)))
        new_func = rand(compatible_primitives)

        # Get the arity of the selected function
        n_args = primitives_with_arity[new_func]

        println("New function: $new_func, arity: $n_args")
        if new_func == :perlin_2d || new_func == :perlin_color
            seed = round(rand() * 100, digits=4)
            args = Vector{Any}(undef, n_args-2)
            for i in 1:(n_args-2)
                args[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            insert!(args, rand(1:n_args-1), expr)
            expr = Expr(:call, new_func, seed, args...)
        elseif new_func == :grad_dir
            op = rand((x -> x[1]).(filter(x -> x.second ∈ [1, 2] && x.first ∉ [:or, :and, :xor, :perlin_2d], collect(primitives_with_arity))))
            n_args_op = primitives_with_arity[op]
            args_op = Vector{Any}(undef, n_args_op)
            for i in 1:n_args_op
                args_op[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            insert!(args_op, rand(1:n_args_op), expr)
            expr = Expr(:call, new_func, op, args_op...)
        else
            args = Vector{Any}(undef, n_args-1)
            for i in 1:(n_args-1)
                args[i] = random_function(primitives_with_arity, depth_of_expr(expr))
            end
            println("args: $args")
            println("expr: $expr")

            insert!(args, rand(1:n_args), expr)
            println("args after insert: $args")

            expr = Expr(:call, new_func, args...)
        end
    end

    if expr isa Expr
        # Mutation type 6: remove an expression from its parent function
        if rand() < mutation_probs[:delete_branch]
            println("Mutation type 6")
            pos = rand(1:length(expr.args))
            expr.args = filter(i -> i != pos, expr.args)
        end

        # Mutation type 7: duplicate a node within the expression
        if rand() < mutation_probs[:duplicate_branch]
            println("Mutation type 7")
            source_pos = rand(1:length(expr.args))
            target_pos = rand(1:length(expr.args))
            expr.args[target_pos] = deepcopy(expr.args[source_pos])
        end

        # Recursively mutate child nodes
        if !mutated
            for i in 2:length(expr.args) # Start from 2 because the function itself is at index 1
                expr.args[i] = mutate!(expr.args[i], mutation_probs, primitives_with_arity, expr, i)
            end
        end
    end

    return expr
end

function mutate!(ce::CustomExpr, mutation_probs, primitives_with_arity)
    mutated_expr = mutate!(ce.expr, mutation_probs, primitives_with_arity, nothing, 0)
    return CustomExpr(mutated_expr)
end

function mutate(ce::CustomExpr, mutation_probs, primitives_with_arity)
    mutated_expr = mutate!(deepcopy(ce.expr), mutation_probs, primitives_with_arity, nothing, 0)
    return CustomExpr(mutated_expr)
end