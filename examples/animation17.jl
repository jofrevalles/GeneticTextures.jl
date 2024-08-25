using GeneticTextures

F_A0 = :(Complex(4*x, 4*y))
F_dA = :(A^2 + Complex(0.355, 0.355))
# F_dA = :(A^2 + Complex(0.74543, 0.11301))

F_B0 = :(0)
F_dB = :(ifs(abs(A) > 2, B + 0.1, B))

F_C0 = :(0) # C will be used as a counter
F_dC = :(C + 0.1)


# Define the color expression
color_expr = :(begin
    using ColorSchemes
    smooth_modifier(k, z) = abs(z) > 2 ? k - log2(log2(abs(z))) : k

    angle_modifier(k, z) =  abs(z) > 2 ? angle(z)/2 * pi : k
    # Access the viridis color scheme
    color_scheme = ColorSchemes.viridis

    # Function to map a value between 0 and 1 to a color in the viridis scheme
    function map_to_color(value)
        return get(color_scheme, value)
    end

    (a, b, c) -> begin
        # Calculate the modified iteration count
        modified_k = smooth_modifier(c.r, a.r)

        # Normalize using the counter C
        normalized_k = modified_k / c.r

        # Map to color
        map_to_color(normalized_k)
    end
end)

complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C0, F_dC)
ds = DynamicalSystem([A,B,C])

w = h = 1024
animate_system_2(ds, w, h, 2.0, 0.01;
color_expr,
complex_expr)

