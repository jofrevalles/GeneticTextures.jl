using Plots

# Main loop
n = 6
width = 128
height = 128
max_depth = 3

# Define mutation probabilities
mutation_probs = Dict(
    :rand_expr => 0.06,
    :adjust_scalar => 0.1,
    :adjust_color => 0.1,
    :rand_func => 0.08,
    :add_argument => 0.1,
    :become_argument => 0.08,
    :duplicate_node => 0.03,
)

original_population, original_image = generate_population(1, primitives_with_arity, max_depth, width, height)
# original_population = [GeneticTextures.CustomExpr(:(cosh(mod(exp(sqrt(or(Color(0.57, 0.22, 0.0), 0.0693))), grad_dir(atan, y / x, sinh(0.4493), 0.4074)))))]
# original_population = [GeneticTextures.CustomExpr(:(cos(perlin_color(84.2126, grad_mag(mod, exp(x), 0.91 - 0.9128), abs(x), dissolve(y, Color(0.28, 0.47, 0.86), Color(0.28, 0.47, 0.86))))))]
original_image = [generate_image(original_population[1], width, height)]
population, images = create_variations(1, original_population, mutation_probs, primitives_with_arity, max_depth, width, height)

display_images(original_image[1], images)

push!(images, original_image[1])
push!(population, original_population[1])

println("Original population: $original_population")
# savefig("example_UI.svg")

# Main loop
let population = population, images = images
    while true
        best_choice = get_user_choice(length(images))
        println("Chosen Image: $(population[best_choice])")
        chosen_image = images[best_choice]
        chosen_population = population[best_choice]

        if best_choice === nothing
            break
        end

        population, images = create_variations(best_choice, population, mutation_probs, primitives_with_arity, max_depth, width, height)
        display_images(chosen_image, images)

        push!(images, chosen_image)
        push!(population, chosen_population)
        # savefig("example_UI.svg")
    end
end
