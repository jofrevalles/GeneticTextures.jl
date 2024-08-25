using BenchmarkTools, CSV, DataFrames, Plots

function evolve_normal!(vals, dynamics::DynamicalSystem3, custom_exprs, width, height, t, dt, complex_func::Function)
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    vars = merge(Dict(:t => t), Dict(name(ds) => vals[i] for (i, ds) in enumerate(dynamics)))

    for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            vars[:x] = x
            vars[:y] = y

            for (i, ds) in enumerate(dynamics)

                val =  dt .* invokelatest(custom_exprs[i].func, vars)

                if val isa Color
                    δvals[i][y_pixel, x_pixel] = val
                elseif isreal(val)
                    δvals[i][y_pixel, x_pixel] = Color(val, val, val)
                else
                    δvals[i][y_pixel, x_pixel] = Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                end
            end
        end
    end

    # Update vals
    for (i, ds) in enumerate(dynamics)
        vals[i] += δvals[i]
    end

    return vals
end

function evolve_threaded!(vals, dynamics::DynamicalSystem3, custom_exprs, width, height, t, dt, complex_func::Function)
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    vars = merge(Dict(:t => t), Dict(name(ds) => vals[i] for (i, ds) in enumerate(dynamics)))

    Threads.@threads for x_pixel in 1:width
        for y_pixel in 1:height
            x = (x_pixel - 1) / (width - 1) - 0.5
            y = (y_pixel - 1) / (height - 1) - 0.5

            vars[:x] = x
            vars[:y] = y

            for (i, ds) in enumerate(dynamics)

                val =  dt .* invokelatest(custom_exprs[i].func, vars)

                if val isa Color
                    δvals[i][y_pixel, x_pixel] = val
                elseif isreal(val)
                    δvals[i][y_pixel, x_pixel] = Color(val, val, val)
                else
                    δvals[i][y_pixel, x_pixel] = Color(invokelatest(complex_func, val), invokelatest(complex_func, val), invokelatest(complex_func, val))
                end
            end
        end
    end

    # Update vals
    for (i, ds) in enumerate(dynamics)
        vals[i] += δvals[i]
    end

    return vals
end

function vectorize_color_decision!(results, δvals, complex_func, i)
    # Determine the type of each element in results
    is_color = [r isa Color for r in results]
    is_real_and_not_color = [isreal(r) && !(r isa Color) for r in results]

    # Assign directly where result elements are Color
    δvals[i][is_color] = results[is_color]

    # Where the results are real numbers, create Color objects
    real_vals = results[is_real_and_not_color]
    δvals[i][is_real_and_not_color] = Color.(real_vals, real_vals, real_vals)

    # For remaining cases, apply the complex function and create Color objects
    needs_complex = .!(is_color .| is_real_and_not_color)
    complex_results = results[needs_complex]
    processed_complex = complex_func.(complex_results)
    δvals[i][needs_complex] = Color.(processed_complex, processed_complex, processed_complex)
end


function evolve_vectorized!(vals, dynamics::DynamicalSystem3, custom_exprs, width, height, t, dt, complex_func::Function)
    # Precompute coordinate grids
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    # Prepare δvals to accumulate changes
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    vars = merge(Dict(:t => t), Dict(name(ds) => vals[i] for (i, ds) in enumerate(dynamics)))

    # Loop through each dynamical system
    for (i, ds) in enumerate(dynamics)
        # Evaluate the function for all pixels in a vectorized manner
        result = broadcast((x, y) -> dt .* invokelatest(custom_exprs[i].func, merge(vars, Dict(:x => x, :y => y))), X, Y)

        # After obtaining results for each dynamic system
        vectorize_color_decision!(result, δvals, complex_func, i)
    end

    # Update vals by adding δvals
    for (i, ds) in enumerate(dynamics)
        vals[i] .+= δvals[i]
    end

    return vals
end

# Helper function to create meshgrid equivalent to MATLAB's
function meshgrid(x, y)
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(reshape(y, :, 1), 1, length(x))
    return X, Y
end

function evolve_threadandvectorized!(vals, dynamics::DynamicalSystem3, custom_exprs, width, height, t, dt, complex_func::Function; n_blocks=Threads.nthreads()*4)
    # Prepare δvals to accumulate changes
    δvals = [Matrix{Color}(undef, height, width) for _ in 1:length(dynamics)]

    # Compute coordinate grids once
    x_grid = ((1:width) .- 1) ./ (width - 1) .- 0.5
    y_grid = ((1:height) .- 1) ./ (height - 1) .- 0.5
    X, Y = meshgrid(x_grid, y_grid)

    vars = merge(Dict(:t => t), Dict(name(ds) => vals[i] for (i, ds) in enumerate(dynamics)))

    # Multithreading across either columns or blocks of columns
    Threads.@threads for block in 1:n_blocks
        x_start = 1 + (block - 1) * Int(width / n_blocks)
        x_end = block * Int(width / n_blocks)
        X_block = X[:, x_start:x_end]
        Y_block = Y[:, x_start:x_end]
        δvals_block = [Matrix{Color}(undef, height, x_end - x_start + 1) for _ in 1:length(dynamics)]


        # Vectorized computation within each block
        for (i, ds) in enumerate(dynamics)
            result_block = broadcast((x, y) -> dt .* invokelatest(custom_exprs[i].func, merge(vars, Dict(:x => x, :y => y))), X_block, Y_block)

            # Use a vectorized color decision
            vectorize_color_decision!(result_block, δvals_block, complex_func, i)
        end

        # Update the global δvals with the block's results
        for (i, ds) in enumerate(dynamics)
            δvals[i][:, x_start:x_end] .= δvals_block[i]
        end
    end

    # Update vals by adding δvals
    for (i, ds) in enumerate(dynamics)
        vals[i] .+= δvals[i]
    end

    return vals
end

# Example sizes and parameters
sizes = [64, 128, 256, 512, 1024]
n_blocks = Threads.nthreads() .* [1, 2, 4, 8, 16]
T = 1.0  # Total time
dt = 0.2  # Time step

F_A0 = :(-1*rand_scalar()*1.27-0.06)
F_dA = :(neighbor_min(A; Δx=2, Δy=2))

F_B0 = :(-0.032+0)
F_dB = :(laplacian(A*4.99))

color_expr = :((a, b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))
complex_expr = :((c) -> real(c))

color_func = eval(color_expr)
complex_func = eval(complex_expr)

A = VariableDynamics3(:A, F_A0, F_dA)
B = VariableDynamics3(:B, F_B0, F_dB)

dynamics = DynamicalSystem3([A, B])

# Define the DataFrame to capture benchmark results
df = DataFrame(
    Size = Int[],
    Blocks = Int[],
    Time = Float64[],
    MinTime = Float64[],
    MaxTime = Float64[],
    Memory = String[],
    nthreads = Int[]
)

# Loop through each size and block configuration
for size in sizes
    global width = global height = size
    vals = [Color.(rand(height, width), rand(height, width), rand(height, width)) for _ in 1:length(dynamics)]
    custom_exprs = [CustomExpr(compile_expr(δF(ds), custom_operations, primitives_with_arity, gradient_functions, width, height, Dict())) for ds in dynamics]

    for blocks in n_blocks
        bench_result = @benchmark evolve_threadandvectorized!($vals, $dynamics, $custom_exprs, $width, $height, $T, $dt, $complex_func; n_blocks=$blocks)

        # Append results to DataFrame
        push!(df, (
            Size = size,
            Blocks = blocks,
            Time = mean(bench_result.times) / 1e6,  # Convert to milliseconds
            MinTime = minimum(bench_result.times) / 1e6,
            MaxTime = maximum(bench_result.times) / 1e6,
            Memory = Base.format_bytes(memory(bench_result)),
            nthreads = Threads.nthreads()
        ))

        @info "Benchmark completed for size $size with $blocks blocks"
    end
end


# Save the DataFrame to a CSV file
CSV.write("benchmark_evolve_threadandvector_results.csv", df)