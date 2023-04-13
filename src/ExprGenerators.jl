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
    :perlin_color => 4,
    :grad_dir => 3, # grad_dir takes 3 arguments: the expression and the x, y coordinates
    :atan => 2,
    :log => 1,
    :exp => 1,
    :or => 2,
    :and => 2,
    :xor => 2,
    :x => 0,
    :y => 0,
    :rand_scalar => 0,
    :rand_color => 0
)

function random_expr(primitives_with_arity, max_depth; kwargs...)
    return CustomExpr(random_function(primitives_with_arity, max_depth; kwargs...))
end

function random_function(primitives_with_arity, max_depth; boolean_functions_depth_threshold = 1)
    """
    Function that creates random texture description functions using the primitive
    functions and the `Expr` type. This function should take the maximum depth of the
    expression tree as an input and return an `Expr`` object.
    """
    if max_depth == 0
        # Select a random primitive function with arity 0 (constant or variable)
        f = rand([k for (k, v) in primitives_with_arity if v == 0])

        if f == :rand_scalar
            return rand()
        elseif f == :rand_color
            return Color(rand(3))
        else
            return f
        end
    else
        if max_depth > boolean_functions_depth_threshold # only allow boolean functions deep in the function graph
            available_funcs = [k for k in keys(primitives_with_arity) if k ∉ [:or, :xor, :and]]
        else
            available_funcs = keys(primitives_with_arity)
        end

        # Select a random primitive function
        f = rand(available_funcs)
        n_args = primitives_with_arity[f]

        #TODO: refactor the code so looks cleaner, I don't like how we are calling `rand()/5` etc
        #TODO:  make that the random numbers in the print only have 2 digits after the decimal point

        # TODO: check if this is the best way to handle perlin_2d
        if f == :perlin_2d
            args = Expr(:call, f, :x, :y)
            # f = :perlin_2d
            # n_args = 2
            # args = [:x, :y]
        elseif f == :perlin_color # TODO: maybe we would like to disable this part
            args = Expr(:call, f, :x, :y, rand()/4, rand()/4, rand()/4)

            # TODO: change this so it just has arguments that are not of type `Color`anyomore
            # Maybe we can remove that from the `primitives_with_arity` dict
            # Also maybe we can test if we want low depth of other functions inside this arguments

            # available_funcs = [k for k in keys(primitives_with_arity) if k ∉ [:perlin_color, :rand_color]]
            # f = :perlin_color
            # n_args = 4
            # args = [:x, :y, rand()/5, rand()/5]
        elseif f == :rand_scalar
            return rand()
        elseif f == :rand_color
            return Color(rand(3))
        elseif f == :grad_dir  # ??remove the Color maker functions from primitives_with_arity
            op = rand(keys(primitives_with_arity)) #maybe disable boolean functions here?
            args = [op, [random_function(primitives_with_arity, max_depth - 1) for _ in 2:n_args]...]

            return Expr(:call, f, args...)
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