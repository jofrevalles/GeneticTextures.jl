using GeneticTextures

F_A0 = :(Complex(4*x, 4*y))
F_dA = :(A^2 + Complex(0.355, 0.355))
# F_dA = :(A^2 + Complex(0.14, 0.13))

F_B0 = :(0)
F_dB = :(ifs(abs(A) > 2, B + 0.1, B))


color_expr = :((a, b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
ds = DynamicalSystem([A,B])

w = h = 512
animate_system_2(ds, w, h, 200.0, 0.02; color_expr, complex_expr)