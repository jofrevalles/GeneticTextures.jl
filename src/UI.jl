using Images
using GeneticTextures
using Plots
gr(size=(800, 800))

# Function to generate initial population
function generate_population(n::Int, primitives_with_arity, max_depth)
    population = [random_expr(primitives_with_arity, max_depth) for _ in 1:n]
    println("population: $population")
    images = [generate_image(expr, width, height) for expr in population]
    return population, images
end

# Function to display images with their index
function display_images(original_image, mutated_images)
    p1 = plot(original_image, aspect_ratio=:equal, axis=false, title="Original Image")
    p_mutated = [plot(img, aspect_ratio=:equal, axis=false, title="Mutation $i") for (i, img) in enumerate(mutated_images)]

    l = @layout([a{0.5h}; grid(2, 3)])
    p = plot(p1, p_mutated..., layout=l, size=(800, 600), legend=false)

    display(p)
end

function get_user_choice(n::Int)
    println("Enter the index of your favorite image:")
    choice = parse(Int, chomp(readline()))

    if 1 <= choice <= n
        return choice
    else
        println("Invalid input. Please try again.")
        return get_user_choice(n)
    end
end

# Function to create new variations based on the user-selected image
function create_variations(best_choice, population, mutation_probs, primitives_with_arity, max_depth)
    new_population = []
    new_images = []

    for i in 1:6
        mutated_expr = mutate(population[best_choice], mutation_probs, primitives_with_arity, max_depth)
        mutated_img = generate_image(mutated_expr, width, height)
        println("Mutation $i: $mutated_expr")

        push!(new_population, mutated_expr)
        push!(new_images, mutated_img)
    end

    return new_population, new_images
end

# Main loop
n = 6
width = 128
height = 128
max_depth = 3

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

let population = population
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
