# Initial setup
n_images = 10  # Set this to the number of images you want to generate
width = 128
height = 128
max_depth = 3
max_changes = 1

# Define mutation probabilities
mutation_probs = Dict(
    :rand_expr => 0.02,
    :adjust_scalar => 0.05,
    :adjust_color => 0.05,
    :rand_func => 0.01,
    :add_argument => 0.02,
    :become_argument => 0.04,
    :duplicate_node => 0.03,
)

# let block begins
let
    # Initialize the population and generate the first image
    population, image = generate_population(1, primitives_with_arity, max_depth, width, height)
    populaion = population[1]
    population = GeneticTextures.CustomExpr(:(cosh(sinh(grad_dir(/, blur(-, cosh(atan(atan(Color(0.38, 0.1, 0.57), y), sin(grad_mag(-, x, Color(0.58, 0.06, 0.85)) + grad_mag(sqrt, x))) * (sin(perlin_color(59.6018, 0.1509, y, Color(0.72, 0.84, 0.84)) + 0.777) + grad_mag(sin, sqrt(Color(0.63, 0.65, 0.15)) + (0.08 - Color(0.18, 0.45, 0.15))))), grad_dir(+, sinh(sinh(y)), (perlin_color(41.4285, Color(0.24, 0.94, 0.41), atan(perlin_2d(4.6597, Color(0.49, 0.2, 0.47), Color(0.98, 0.8, 0.81)), Color(0.91, 0.44, 0.72)), Color(0.48, 0.58, 0.72)) - grad_dir(/, sqrt(xor(y, y)), 0.0361)) * mod(perlin_color(29.0295, x, Color(0.76, 0.62, 0.15), Color(0.87, 0.6, 0.13)), grad_mag(+, mod(grad_mag(+, Color(0.6, 0.99, 0.79), x), x), exp(xor(y, Color(0.98, 0.93, 0.16))))))), x)))))

    images = [generate_image(population, width, height)]
    save_image_and_expr(images[1], population; folder = "frames")

    # Main loop
    for i in 2:n_images
        # Evolve the population and generate a new variation
        population = mutate(population, mutation_probs, primitives_with_arity, max_changes)
        # new_image = generate_image(population[1], width, height)
        try
            new_image = generate_image(population, width, height)

            push!(images, new_image)

            # Save the variation
            save_image_and_expr(map(clamp01nan, images[end]), population; folder = "frames")
        catch
            # If generation fails, try to create variations again
            population = mutate(population, mutation_probs, primitives_with_arity, max_changes)
            new_image = generate_image(population, width, height)

            push!(images, new_image)

            # Save the variation
            save_image_and_expr(map(clamp01nan, images[end]), population; folder = "frames")
        end
        # push!(images, new_image)

        # # Save the variation
        # save_image_and_expr(images[end], population; folder = "frames")
    end
end
