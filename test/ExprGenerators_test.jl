@testset "ExprGenerators" begin
    using GeneticTextures: CustomExpr, random_expr, random_function, grad_dir, primitives_with_arity, depth_of_expr
    max_depth = 5

    @testset "random_expr" begin
        for _ in 1:20
            c_expr = random_expr(primitives_with_arity, max_depth)

            if c_expr isa CustomExpr
                @test c_expr.expr isa Union{Expr, Number, Color, Symbol}
            else
                @test c_expr isa Union{Number, Color, Symbol}
            end
        end
    end

    function check_expr(expr, primitives_with_arity, depth, max_depth)
        if expr isa Expr
            @test expr.head == :call # Check if the expression is a function call
            @test length(expr.args) > 0 # Check if the expression has arguments

            # Check if the called function is in the list of primitive functions
            f = expr.args[1]
            @test f in keys(primitives_with_arity)

            # Check the number of arguments for the function call.
            # For the `:grad_mag` function, the arity depends on the input function it takes,
            # so we need to handle it differently.
            if f == :grad_mag
                n_args = primitives_with_arity[expr.args[2]]
                @test length(expr.args) == n_args + 2
            else
                n_args = primitives_with_arity[f]
                @test length(expr.args) == n_args + 1
            end

            if depth < max_depth
                for arg in expr.args[2:end]
                    if arg isa Expr || arg isa Number || arg isa Color
                        check_expr(arg, primitives_with_arity, depth + 1, max_depth)
                    end
                end
            end
        else
            @test expr isa Number || expr isa Color || expr isa Symbol
        end
    end

    @testset "random_function" begin
        for _ in 1:6
            expr = random_function(primitives_with_arity, max_depth)
            check_expr(expr, primitives_with_arity, 1, max_depth)
        end

        @testset "max_depth" begin
            for max_depth in 0:5
                for _ in 1:20
                    expr = random_function(primitives_with_arity, max_depth)
                    actual_depth = depth_of_expr(expr)
                    @test actual_depth <= max_depth
                end
            end
        end
    end
end
