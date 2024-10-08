const primitives_with_arity = Dict(
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :^ => 2,
    :sin => 1,
    :cos => 1,
    :sinh => 1,
    :cosh => 1,
    :abs => 1,
    :sqrt => 1,
    :mod => 2,
    :perlin_2d => 3,
    :perlin_color => 4,
    :grad_mag => 1, # grad_mag takes 1 argument, but it can be a function with variable number of arguments
    :grad_dir => 3,
    :blur => 3, # blur takes 3 arguments, since currently the first argument has to be a function with 2 arguments
    :atan => 2,
    :log => 1,
    :exp => 1,
    :round => 1,
    :Int => 1,
    :or => 2,
    :and => 2,
    :xor => 2,
    :x => 0,
    :y => 0,
    :rand_scalar => 0,
    :rand_color => 0,
    :dissolve => 3,
    :laplacian => 1,
    :x_grad => 1,
    :y_grad => 1,
    :grad_magnitude => 1,
    :grad_direction => 1,
    :neighbor_min => 1,
    :neighbor_max => 1,
    :neighbor_ave => 1,
    :ifs => 3,
    :max => 2,
    :min => 2,
    :real => 1,
    :imag => 1,
    :A => -1,
    :B => -1,
    :C => -1,
    :D => -1,
    :t => -1
)

# special_funcs take not only numbers as arguments
const special_funcs = (
    :blur,
    :perlin_2d,
    :perlin_color
)

# boolean_funcs are boolean functions
const boolean_funcs = (
    :or,
    :and,
    :xor,
    :ifs,
)

# variables
const variables = (
    :A,
    :B,
    :C,
    :D,
    :t
)

# color_funcs are functions that can return a color
const color_funcs = (
    :perlin_color,
    :rand_color,
    :Color
)

function random_expr(primitives_with_arity, max_depth; kwargs...)
    return GeneticExpr(random_function(primitives_with_arity, max_depth; kwargs...))
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
            available_funcs = [k for k in keys(primitives_with_arity) if k ∉ boolean_funcs && k ∉ variables]
        else
            available_funcs = filter(x -> x ∉ variables, keys(primitives_with_arity))
        end
        # Select a random primitive function
        f = rand(available_funcs)
        n_args = primitives_with_arity[f]

        #TODO: refactor the code so looks cleaner, I don't like how we are calling `rand()/5` etc
        #TODO: make that the random numbers in the print only have 2 digits after the decimal point

        # TODO: check if this is the best way to handle perlin_2d
        if f == :perlin_2d
            seed = round(rand() * 100, digits = 4)
            limited_depth = min(3, max_depth) # Set a limit to the depth of the functions for the arguments

            args = [random_function(primitives_with_arity, limited_depth - 1) for _ in 1:n_args-1]

            return Expr(:call, f, seed, args...)
        elseif f == :perlin_color
            seed = round(rand() * 100, digits = 4)
            limited_depth = min(3, max_depth) # Set a limit to the depth of the functions for the arguments

            args = [random_function(primitives_with_arity, limited_depth - 1) for _ in 1:n_args-2]

            return Expr(:call, f, seed, args..., Color(rand(3)))
        elseif f == :rand_scalar
            return rand()
        elseif f == :rand_color
            return Color(rand(3))
        elseif f == :blur
            op = rand((x -> x[1]).(filter(x -> x.second ∈ [2] && x.first ∉ [:or, :and, :xor, :perlin_2d, :perlin_color, :grad_dir, :grad_mag], collect(primitives_with_arity)))) #maybe disable boolean functions here?
            n_args = primitives_with_arity[op]

            args = [op, [random_function(primitives_with_arity, max_depth - 1) for _ in 1:n_args]...]

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

function depth_of_expr(expr)
    if expr isa Expr
        max_child_depth = 0
        for arg in expr.args[2:end]
            child_depth = depth_of_expr(arg)
            max_child_depth = max(max_child_depth, child_depth)
        end
        return 1 + max_child_depth
    else
        return 0 # If the input is not an Expr (i.e., a Number, Color, or Symbol), return 0 as the depth
    end
end