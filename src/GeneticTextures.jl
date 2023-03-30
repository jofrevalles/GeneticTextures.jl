module GeneticTextures

include("expressions.jl")
export primitives_with_arity, random_function, custom_eval

include("renderer.jl")
export render, substitute, save_population

# include("genetic_algorithm.jl")

# include("user_interface.jl")

end