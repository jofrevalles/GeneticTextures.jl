using GeneticTextures

F_A0 = :(Complex(4*x, 4*y))
# F_dA = :(A^2 + Complex(0.355, 0.355))
F_da = :(A^2 + Complex(0.74543, 0.11301))

F_B0 = :(0)
F_dB = :(ifs(abs(A) > 2, B + 1, B))

F_C0 = :(0) # C will be used as a counter
F_dC = :(C + 1)

color_expr = :(begin
    using ColorSchemes

    # Access the viridis color scheme
    color_scheme = ColorSchemes.viridis

    # Function to map a value between 0 and 1 to a color in the viridis scheme
    function map_to_color(value)
        return get(color_scheme, value)
    end

    (a, b, c) -> map_to_color(b.r/c.r)
    end)
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C0, F_dC)
ds = DynamicalSystem([A,B,C])

w = h = 128
animate_system_2(ds, w, h, 200.0, 0.3; color_expr, complex_expr)