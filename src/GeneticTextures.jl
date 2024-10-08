module GeneticTextures

include("Utils.jl")

include("Color.jl")
export Color, red, blue, green, ColorStyle

include("PerlinBroadcasting.jl")
include("GeneticExpr.jl")
export GeneticExpr

include("ExprGenerators.jl")
export primitives_with_arity, random_expr

include("MathOperations.jl")
export gradient_functions

include("ExprEvaluation.jl")
export compile_expr

include("Renderer.jl")
export generate_image, save_image_and_expr

include("Genetic.jl")
export mutate, mutate!

include("UI.jl")
export generate_population, display_images, get_user_choice, create_variations

include("DynamicalSystems.jl")
export DynamicalSystem, VariableDynamics, animate_system

end