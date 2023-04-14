@testset "Color" begin
    using GeneticTextures: Color

    # test Color constructor
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
end