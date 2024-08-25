using CSV, DataFrames, Plots

function normalize_memory_(mem_str)
    if occursin("GiB", mem_str)
        val = parse(Float64, replace(mem_str, " GiB" => ""))
        return val * 1024  # Convert GiB to MiB if you want everything in MiB
    elseif occursin("MiB", mem_str)
        return parse(Float64, replace(mem_str, " MiB" => ""))
    else
        error("Unknown memory unit in the string: $mem_str")
    end
end

line_styles = [:solid, :dash, :dot, :dashdot, :dashdash]
markers = [:circle, :square, :diamond, :cross, :utriangle]
labels = ["Normal", "Threaded", "Vectorized", "ThreadVector"]
sizes = [64, 128, 256, 512, 1024]

# Load the data
df = CSV.read("benchmark_evolve_threadandvector_results.csv", DataFrame)

# Normalize memory usage
df.Memory_MiB = [normalize_memory_(mem) for mem in df.Memory]

# Group by Size and Blocks to aggregate data if necessary or prepare for plotting directly
grouped = groupby(df, [:Blocks])

# Initialize the plot for time and memory usage
p1 = plot(title = "Runtime Difference compared with best",
          xlabel = "Image Size (pixels)", ylabel = "Average Time (ms)", legend = :outertopright, xscale = :log2, yscale = :log)

p2 = plot(title = "Memory Usage Difference compared with best",
          xlabel = "Image Size (pixels)", ylabel = "Memory Usage (MiB)", legend = :outertopright, xscale = :log2, yscale = :log)

min_times = zeros(length(sizes)) # Get the minimum value at each size between  df.Normal_Time, df.Threaded_Time, df.Vectorized_Time, df.ThreadVector_Time
min_memory = zeros(length(sizes)) # Get the minimum value at each size between  df.Normal_Mem, df.Threaded_Mem, df.Vectorized_Mem, df.ThreadVector_Mem
for (i, (key, group)) in enumerate(pairs(grouped))
    min_times[i] = minimum([group.Time[i] for (key, group) in pairs(grouped)]) - 1
    min_memory[i] = minimum([group.Memory_MiB[i] for (key, group) in pairs(grouped)]) - 1
end

# Iterate through each group and plot
for (i, (key, group)) in enumerate(pairs(grouped))
    blocks = key.Blocks[1] #blocks is a GroupKey
    println("blocks: $blocks")


    plot!(p1, sizes, group.Time .- min_times, label = "Blocks=$blocks",
          line = line_styles[i], marker = markers[i], markersize = 4, lw = 2)

    plot!(p2, sizes, group.Memory_MiB .- min_memory, label = "Blocks=$blocks",
          line = line_styles[i], marker = markers[i], markersize = 4, lw = 2)
end

# Create a combined plot layout
combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 800))
display(combined_plot)

# Optionally save the plot to a file
savefig(combined_plot, "evolve_theadandvec_performance_by_blocks.png")
