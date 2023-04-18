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

original_population, original_image = generate_population(1, primitives_with_arity, max_depth)
population, images = create_variations(1, original_population, mutation_probs, primitives_with_arity, max_depth)
display_images(original_image[1], images)
println("Original population: $original_population")

# Main loop
let population = population, images = images
    while true
        best_choice = get_user_choice(n)
        println("Chosen Image: $(population[best_choice])")
        chosen_image = images[best_choice]

        if best_choice === nothing
            break
        end

        population, images = create_variations(best_choice, population, mutation_probs, primitives_with_arity, max_depth)
        display_images(chosen_image, images)
    end
end
