using GeneticTextures
F_A0 = :(Complex(4*x, y))
# F_dA = :(A^2 + Complex(0.355, 0.355))
F_dA = :(A^2 + Complex(0.74543, 0.11301))

F_B0 = :(0)
F_dB = :(ifs(abs(A) > 2, B + 1, B))

F_C0 = :(0) # C will be used as a counter
F_dC = :(C + 1)


# Define the value expression
value_expr = :(begin
    smooth_modifier(k, z) = abs(z) > 2 ? k - log2(log2(abs(z))) : k

    angle_modifier(k, z) =  abs(z) > 2 ? angle(z)/2 * pi : k

    (a, b, c) -> begin
        # Calculate the modified iteration count
        modified_k = angle_modifier(c.r, a.r)
    end
end)

complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C0, F_dC)
ds = DynamicalSystem([A,B,C])

w = h = 32
GeneticTextures.animate_system_3(ds, w, h, 200.0, 0.01;
cmap = :viridis,
value_expr,
complex_expr)

