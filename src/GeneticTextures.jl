module GeneticTextures

include("Color.jl")
export Color, ColorStyle

include("expressions.jl")
export primitives_with_arity, random_function, random_function_v2, custom_eval, grad_dir

include("renderer.jl")
export render, substitute, save_population, generate_image



# include("genetic_algorithm.jl")

# include("user_interface.jl")

end