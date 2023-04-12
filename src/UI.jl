using GeneticTextures
# Set the desired image dimensions
width, height = 128, 128

population = [random_function(primitives_with_arity, 5) for _ in 1:1]

# Create an empty array to store the rendered images
rendered_images = Array{Array{Float64, 2}, 1}(undef, length(population))

# Render each expression in the population
for (i, expr) in enumerate(population)
    rendered_images[i] = GeneticTextures.render(expr, width, height)
end
