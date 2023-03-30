primitives = [:+, :-, :*, :/, :^, :sin, :cos, :tan]

function random_function(primitives, max_depth)
    """
    Function that creates random texture description functions using the primitive
    functions and the `Expr` type. This function should take the maximum depth of the
    expression tree as an input and return an `Expr`` object.
    """
    if max_depth == 0
        return rand(0.0:0.1:1.0) # return a random constant between 0 and 1
    end

    op = rand(primitives)
    num_args = arity(op)
    args = [random_function(primitives, max_depth - 1) for _ in 1:num_args]
    return Expr(:call, op, args...)
end