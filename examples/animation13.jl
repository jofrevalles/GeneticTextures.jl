using GeneticTextures
F_A0 = :(y)
F_dA = :(C)

F_B0 = :(1.0)
F_dB = :(x_grad(C))

F_C = :(1 - ifs(rand_scalar() > 0.97, 0.0, 1.0) * y)
F_dC = :(neighbor_ave(grad_direction(B * 0.25)))


color_expr = :((a, b, c) -> RGB(abs(a.r), abs(b.g), abs(c.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C, F_dC)

ds = DynamicalSystem([A, B, C])
w = h = 32
animate_system_2(ds, w, h, 10.0, 0.01; color_expr, complex_expr)