@testset "Color" begin
    using GeneticTextures: Color

    @testset "constructor" begin
        array = rand(3)
        c = Color(array)

        @test c == Color(array[1], array[2], array[3])
        @test c.r == array[1]
        @test c.g == array[2]
        @test c.b == array[3]

        @test Color(1, 2, 3) == Color([1, 2, 3])

        @test_throws ArgumentError Color(rand(2))
        @test_throws ArgumentError Color(rand(4))
    end

    @testset "indexing" begin
        c = Color(0.1, 0.2, 0.3)

        @test c[1] == 0.1
        @test c[2] == 0.2
        @test c[3] == 0.3

        @test_throws BoundsError c[0]
        @test_throws BoundsError c[4]

        c[1] = 0.4
        c[2] = 0.5
        c[3] = 0.6

        @test c == Color(0.4, 0.5, 0.6)
    end

    @testset "iterating" begin
        c = Color(0.1, 0.2, 0.3)
        values = [0.1, 0.2, 0.3]

        for (i, val) in enumerate(c)
            @test val == values[i]
        end
    end

    @testset "broadcasting" begin
        array1 = rand(3)
        array2 = rand(3)
        c1 = Color(array1)
        c2 = Color(array2)

        function f(x...)
            return mapreduce(c -> Vector(c), +, x)
        end

        @test f(c1, c2) == Color(f(array1, array2))

        @test c1 .+ 1 == Color(array1 .+ 1)
        @test 1 .+ c1 == Color(1 .+ array1)
        @test c1 .+ c2 == Color(array1 .+ array2)
    end

end