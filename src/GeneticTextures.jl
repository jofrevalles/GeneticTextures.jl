module GeneticTextures

include("Utils.jl")

include("Color.jl")
export Color, ColorStyle

include("CustomExpr.jl")
include("ExprGenerators.jl")
export primitives_with_arity, random_expr

include("MathOperations.jl")

include("ExprEvaluation.jl")
export custom_eval

include("Renderer.jl")
export generate_image, save_image_and_expr

# include("Genetic.jl")

# include("User.jl")

end