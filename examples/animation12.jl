using GeneticTextures
F_A0 = :(Complex(x, y))
# F_dA = :(A^2 + Complex(0.355, 0.355))
F_dA = :(A^2 + Complex(0.14, 0.13))


color_expr = :((a) -> RGB(abs(a.r), abs(a.g), abs(a.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
# B = VariableDynamics(:B, F_B0, F_dB)
# ds = DynamicalSystem([A,B])

ds = DynamicalSystem([A])
w = h = 512
animate_system_2(ds, w, h, 200.0, 0.04; color_expr, complex_expr)