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

line_styles = [:solid, :dash, :dot, :dashdot]
markers = [:circle, :square, :diamond, :cross]
labels = ["Normal", "Threaded", "Vectorized", "ThreadVector"]

# Load the data
df = CSV.read("benchmark_results_new.csv", DataFrame)

# Correcting the ribbon data calculation
ribbon_lower = [(df.Normal_Time .- df.Normal_MinTime),
                (df.Threaded_Time .- df.Threaded_MinTime),
                (df.Vectorized_Time .- df.Vectorized_MinTime),
                (df.ThreadVector_Time .- df.ThreadVector_MinTime)]
ribbon_upper = [(df.Normal_MaxTime .- df.Normal_Time),
                (df.Threaded_MaxTime .- df.Threaded_Time),
                (df.Vectorized_MaxTime .- df.Vectorized_Time),
                (df.ThreadVector_MaxTime .- df.ThreadVector_Time)]

# You should use ribbon as a vector of tuples, each tuple corresponds to lower and upper for each series
ribbons = [(lower, upper) for (lower, upper) in zip(ribbon_lower, ribbon_upper)]

# Initialize the plot
p1 = plot(title = "Runtime Performance",
          xlabel = "Image Size (pixels)", ylabel = "Average Time (ms)",
          legend = :right, size = (800, 600))

# Add each series individually
labels = ["Normal", "Threaded", "Vectorized", "ThreadVector"]
times = [df.Normal_Time, df.Threaded_Time, df.Vectorized_Time, df.ThreadVector_Time]

min_times = zeros(length(df.Size)) # Get the minimum value at each size between  df.Normal_Time, df.Threaded_Time, df.Vectorized_Time, df.ThreadVector_Time
min_memory = zeros(length(df.Size)) # Get the minimum value at each size between  df.Normal_Mem, df.Threaded_Mem, df.Vectorized_Mem, df.ThreadVector_Mem
for (i, s) in enumerate(df.Size)
    min_times[i] = minimum([df.Normal_Time[i], df.Threaded_Time[i], df.Vectorized_Time[i], df.ThreadVector_Time[i]]) - 1
    min_memory[i] = minimum([normalize_memory_(df.Normal_Mem[i]), normalize_memory_(df.Threaded_Mem[i]), normalize_memory_(df.Vectorized_Mem[i]), normalize_memory_(df.ThreadVector_Mem[i])]) - 1
end


for i in eachindex(labels)
    plot!(df.Size, times[i],
          label = labels[i],
          lw = 2, line = line_styles[i], marker = markers[i], markersize = 4)
end

# Adjust memory normalization and plot memory usage
mem_usage = [normalize_memory_.(df.Normal_Mem), normalize_memory_.(df.Threaded_Mem), normalize_memory_.(df.Vectorized_Mem), normalize_memory_.(df.ThreadVector_Mem)]

# Initialize the plot
p2 = plot(title = "Memory Usage",
          xlabel = "Image Size (pixels)", ylabel = "Memory Usage (MiB)",
          legend = :right, size = (800, 600))

# Add each series individually
for i in eachindex(labels)
    plot!(df.Size, mem_usage[i],
          label = labels[i],
          lw = 2, line = line_styles[i], marker = markers[i], markersize = 4)
end
# Create a combined plot layout
combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 800))
display(combined_plot)

# Optionally save the plot to a file
savefig(combined_plot, "combined_performance_analysis.png")