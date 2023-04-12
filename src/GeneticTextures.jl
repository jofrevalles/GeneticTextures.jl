module GeneticTextures

include("Utils.jl")

include("Color.jl")
export Color, ColorStyle

include("Expressions.jl")
export primitives_with_arity, random_expr, custom_eval, grad_dir, save_image_and_expr

include("Renderer.jl")
export render, substitute, save_population, generate_image

# include("Genetic.jl")

# include("User.jl")

end