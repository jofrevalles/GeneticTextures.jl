@testset "ExprEvaluation" begin
    using GeneticTextures: Color, CustomExpr, custom_eval, primitives_with_arity
    using CoherentNoise: sample, perlin_2d

    vars = Dict(:x => 0.5, :y => 0.5)

    @testset "basic operations" begin
        add_expr = :(+, x, y)
        ce_add = CustomExpr(add_expr)
        @test custom_eval(ce_add, vars) ≈ 1.0

        sub_expr = :(-, x, y)
        ce_sub = CustomExpr(sub_expr)
        @test custom_eval(ce_sub, vars) ≈ 0.0

        mul_expr = :(*, x, y)
        ce_mul = CustomExpr(mul_expr)
        @test custom_eval(ce_mul, vars) ≈ 0.25

        div_expr = :(/, x, y)
        ce_div = CustomExpr(div_expr)
        @test custom_eval(ce_div, vars) ≈ 1.0
    end

    @testset "unary functions" begin
        sin_expr = :(sin(x))
        ce_sin = CustomExpr(sin_expr)
        @test custom_eval(ce_sin, vars) ≈ sin(0.5)

        cos_expr = :(cos(x))
        ce_cos = CustomExpr(cos_expr)
        @test custom_eval(ce_cos, vars) ≈ cos(0.5)

        abs_expr = :(abs(-1 * x))
        ce_abs = CustomExpr(abs_expr)
        @test custom_eval(ce_abs, vars) ≈ 0.5

        sqrt_expr = :(sqrt(x))
        ce_sqrt = CustomExpr(sqrt_expr)
        @test custom_eval(ce_sqrt, vars) ≈ sqrt(0.5)

        exp_expr = :(exp(x))
        ce_exp = CustomExpr(exp_expr)
        @test custom_eval(ce_exp, vars) ≈ exp(0.5)
    end

    @testset "binary functions" begin
        mod_expr = :(mod(3, 2))
        ce_mod = CustomExpr(mod_expr)
        @test custom_eval(ce_mod, vars) ≈ 1

        atan_expr = :(atan(x, y))
        ce_atan = CustomExpr(atan_expr)
        @test custom_eval(ce_atan, vars) ≈ atan(0.5, 0.5)
    end

    @testset "Color" begin
        color_expr = :(Color(0.5, 0.5, 0.5))
        ce_color = CustomExpr(color_expr)
        @test custom_eval(ce_color, vars) ≈ Color(0.5, 0.5, 0.5)
    end

    @testset "perlin_2d" begin
        perlin_expr = :(perlin_2d(1, x, y))
        ce_perlin = CustomExpr(perlin_expr)
        sampler = perlin_2d(seed=hash(1))
        @test custom_eval(ce_perlin, vars) ≈ sample(sampler, 0.5, 0.5)
    end

    @testset "complex expressions" begin
        complex_expr = :(sin(cos(sqrt(/(x, y)))))
        ce_complex = CustomExpr(complex_expr)
        @test custom_eval(ce_complex, vars) ≈ sin(cos(sqrt(0.5 / 0.5)))

        nested_expr = :(*, Color(0.5, 0.5, 0.5), +(x, y))
        ce_nested = CustomExpr(nested_expr)
        @test custom_eval(ce_nested, vars) ≈ Color(0.5, 0.5, 0.5) * (0.5 + 0.5)
    end

    @testset "edge cases" begin
        zero_expr = :(/, x, 0)
        ce_zero = CustomExpr(zero_expr)
        @test custom_eval(ce_zero, vars) ≈ Inf

        large_expr = :(*, 1e7, x)
        ce_large = CustomExpr(large_expr)
        @test custom_eval(ce_large, vars) ≈ 0.5

        negative_large_expr = :(*, -1e7, x)
        ce_negative_large = CustomExpr(negative_large_expr)
        @test custom_eval(ce_negative_large, vars) ≈ -0.5

        nan_expr = :(+, x, (0 / 0))
        ce_nan = CustomExpr(nan_expr)
        @test custom_eval(ce_nan, vars) ≈ 0.5
    end

end