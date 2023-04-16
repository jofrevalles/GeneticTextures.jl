using Test
using GeneticTextures

@testset "Unit tests" verbose = true begin
    include("Color_test.jl")

    include("ExprGenerators_test.jl")
end
