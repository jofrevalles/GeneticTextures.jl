using GeneticTextures
F_A0 = :(Complex(x, y))
F_dA = :(x_grad(A) + y_grad(A) + 0.1 * laplacian(A))


color_expr = :((a) -> RGB(abs(a.r), abs(a.g), abs(a.b)))
complex_expr = :((c) -> abs(c))

A = VariableDynamics(:A, F_A0, F_dA)
# B = VariableDynamics(:B, F_B0, F_dB)
# ds = DynamicalSystem([A,B])

ds = DynamicalSystem([A])
w = h = 64
animate_system_2(ds, w, h, 100.0, 0.2; color_expr, complex_expr)