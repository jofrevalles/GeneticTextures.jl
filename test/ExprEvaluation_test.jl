@testset "ExprEvaluation" begin
    using GeneticTextures: Color, GeneticExpr, custom_eval, primitives_with_arity
    using CoherentNoise: sample, perlin_2d

    vars = Dict(:x => 0.5, :y => 0.5)

    @testset "basic operations" begin
        add_expr = :(+, x, y)
        ge_add = GeneticExpr(add_expr)
        @test custom_eval(ge_add, vars) ≈ 1.0

        sub_expr = :(-, x, y)
        ge_sub = GeneticExpr(sub_expr)
        @test custom_eval(ge_sub, vars) ≈ 0.0

        mul_expr = :(*, x, y)
        ge_mul = GeneticExpr(mul_expr)
        @test custom_eval(ge_mul, vars) ≈ 0.25

        div_expr = :(/, x, y)
        ge_div = GeneticExpr(div_expr)
        @test custom_eval(ge_div, vars) ≈ 1.0
    end

    @testset "unary functions" begin
        sin_expr = :(sin(x))
        ge_sin = GeneticExpr(sin_expr)
        @test custom_eval(ge_sin, vars) ≈ sin(0.5)

        cos_expr = :(cos(x))
        ge_cos = GeneticExpr(cos_expr)
        @test custom_eval(ge_cos, vars) ≈ cos(0.5)

        abs_expr = :(abs(-1 * x))
        ge_abs = GeneticExpr(abs_expr)
        @test custom_eval(ge_abs, vars) ≈ 0.5

        sqrt_expr = :(sqrt(x))
        ge_sqrt = GeneticExpr(sqrt_expr)
        @test custom_eval(ge_sqrt, vars) ≈ sqrt(0.5)

        exp_expr = :(exp(x))
        ge_exp = GeneticExpr(exp_expr)
        @test custom_eval(ge_exp, vars) ≈ exp(0.5)
    end

    @testset "binary functions" begin
        mod_expr = :(mod(3, 2))
        ge_mod = GeneticExpr(mod_expr)
        @test custom_eval(ge_mod, vars) ≈ 1

        atan_expr = :(atan(x, y))
        ge_atan = GeneticExpr(atan_expr)
        @test custom_eval(ge_atan, vars) ≈ atan(0.5, 0.5)
    end

    @testset "Color" begin
        color_expr = :(Color(0.5, 0.5, 0.5))
        ge_color = GeneticExpr(color_expr)
        @test custom_eval(ge_color, vars) ≈ Color(0.5, 0.5, 0.5)
    end

    @testset "perlin_2d" begin
        perlin_expr = :(perlin_2d(1, x, y))
        ge_perlin = GeneticExpr(perlin_expr)
        sampler = perlin_2d(seed=hash(1))
        @test custom_eval(ge_perlin, vars) ≈ sample(sampler, 0.5, 0.5)
    end

    @testset "complex expressions" begin
        complex_expr = :(sin(cos(sqrt(/(x, y)))))
        ge_complex = GeneticExpr(complex_expr)
        @test custom_eval(ge_complex, vars) ≈ sin(cos(sqrt(0.5 / 0.5)))

        nested_expr = :(*, Color(0.5, 0.5, 0.5), +(x, y))
        ge_nested = GeneticExpr(nested_expr)
        @test custom_eval(ge_nested, vars) ≈ Color(0.5, 0.5, 0.5) * (0.5 + 0.5)
    end

    @testset "edge cases" begin
        zero_expr = :(/, x, 0)
        ge_zero = GeneticExpr(zero_expr)
        @test custom_eval(ge_zero, vars) ≈ Inf

        large_expr = :(*, 1e7, x)
        ge_large = GeneticExpr(large_expr)
        @test custom_eval(ge_large, vars) ≈ 0.5

        negative_large_expr = :(*, -1e7, x)
        ge_negative_large = GeneticExpr(negative_large_expr)
        @test custom_eval(ge_negative_large, vars) ≈ -0.5

        nan_expr = :(+, x, (0 / 0))
        ge_nan = GeneticExpr(nan_expr)
        @test custom_eval(ge_nan, vars) ≈ 0.5
    end

end