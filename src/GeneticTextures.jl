module GeneticTextures

include("Utils.jl")

include("Color.jl")
export Color, ColorStyle

include("PerlinBroadcasting.jl")
include("CustomExpr.jl")
include("ExprGenerators.jl")
export primitives_with_arity, random_expr

include("MathOperations.jl")

include("ExprEvaluation.jl")
export custom_eval

include("Renderer.jl")
export generate_image, save_image_and_expr

include("Genetic.jl")
export mutate, mutate!

include("UI.jl")
export generate_population, display_images, get_user_choice, create_variations

# include("DynamicalSystems.jl")
# export DynamicalSystem, animate_system

end