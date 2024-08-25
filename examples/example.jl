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


primitives_with_arity2 = Dict(
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :^ => 2,
    :sin => 1,
    :cos => 1,
    :sinh => 1,
    :cosh => 1,
    :abs => 1,
    :sqrt => 1,
    :mod => 2,
    :perlin_2d => 3,
    :perlin_color => 4,
    :grad_mag => 1, # grad_mag takes 1 argument, but it can be a function with variable number of arguments
    :grad_dir => 3,
    :blur => 3, # blur takes 3 arguments, since currently the first argument has to be a function with 2 arguments
    :atan => 2,
    :log => 1,
    :exp => 1,
    :round => 1,
    :Int => 1,
    :or => 2,
    :and => 2,
    :xor => 2,
    :x => 0,
    :y => 0,
    :rand_scalar => 0,
    :rand_color => 0,
    :dissolve => 3,
    :laplacian => 1,
    :x_grad => 1,
    :y_grad => 1,
    :grad_magnitude => 1,
    :grad_direction => 1,
    :neighbor_min => 1,
    :neighbor_max => 1,
    :neighbor_ave => 1,
    :ifs => 3,
    :max => 2,
    :min => 2,
    :real => 1,
    :imag => 1,
    # :A => 0,
    # :B => 0,
    # :C => 0,
    # :t => 0
)

original_population, original_image = generate_population(1, primitives_with_arity2, max_depth, width, height)
# original_population = [GeneticTextures.CustomExpr(:(cosh(mod(exp(sqrt(or(Color(0.57, 0.22, 0.0), 0.0693))), grad_dir(atan, y / x, sinh(0.4493), 0.4074)))))]
# original_population = [GeneticTextures.CustomExpr(:(cos(perlin_color(84.2126, grad_mag(mod, exp(x), 0.91 - 0.9128), abs(x), dissolve(y, Color(0.28, 0.47, 0.86), Color(0.28, 0.47, 0.86))))))]
original_image = [generate_image(original_population[1], width, height)]
population, images = create_variations(1, original_population, mutation_probs, primitives_with_arity2, max_depth, width, height)

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

        population, images = create_variations(best_choice, population, mutation_probs, primitives_with_arity2, max_depth, width, height)
        display_images(chosen_image, images)

        push!(images, chosen_image)
        push!(population, chosen_population)
        # savefig("example_UI.svg")
    end
end
