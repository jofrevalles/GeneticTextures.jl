using GeneticTextures
F_A0 = :(y)
F_dA = :(neighbor_max(neighbor_max(C; Δx=4, Δy=4); Δx=4, Δy=4))

F_B0 = :(1.0)
F_dB = :(x_grad(C))

F_C = :((1 - rand_scalar()*1.68+0.12) + y)
F_dC = :(neighbor_ave(grad_direction(B * 0.25; Δx=4, Δy=4)))


color_expr = :((a, b, c) -> RGB(abs(c.r), abs(c.g), abs(c.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C, F_dC)

ds = DynamicalSystem([A, B, C])
w = h = 64
animate_system_2threaded(ds, w, h, 10.0, 0.1; color_expr, complex_expr)