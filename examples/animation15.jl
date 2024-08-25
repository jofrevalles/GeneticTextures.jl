using GeneticTextures
using Colors

# Define a colormap
colormap_ = colormap("RdBu", 100)  # Choose any available colormap

# Color mapping function
function map_to_color(b, colormap)
    idx = min(floor(Int, abs(b) * length(colormap)) + 1, length(colormap))
    return colormap[idx]
end

F_A0 = :(Complex(4*x, 4*y))
# F_dA = :(A^2 + Complex(0.355, 0.355))
F_da = :(A^2 + Complex(0.74543, 0.11301))

F_B0 = :(0)
F_dB = :(ifs(abs(A) > 2, B + 0.1, B))

color_expr = :(begin
    using ColorSchemes

    # Access the viridis color scheme
    color_scheme = ColorSchemes.viridis

    # Function to map a value between 0 and 1 to a color in the viridis scheme
    function map_to_color(value)
        return get(color_scheme, value)
    end

    (a, b) -> map_to_color(b.r)
    end)
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
ds = DynamicalSystem([A,B])

w = h = 128
animate_system_2(ds, w, h, 200.0, 0.02; color_expr, complex_expr)