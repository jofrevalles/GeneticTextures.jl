using GeneticTextures


# F_A0= :(Color(Complex(x, y), Complex(y, x), Complex(abs(x - y), abs(x + y))))
# F_dA= :(min(A, Complex(1.0, 0.2)) * A + exp(laplacian(A * Complex(-1.2, 1.0))))

F_A0 = :(1 * ifs(rand_scalar() > 0.9994, 0, 1))
F_dA = :(neighbor_min(A; Δx = 2 + Int(round(t*200)), Δy = 2 + Int(round(t*200))) + exp(laplacian(B)))

F_B0 = :(-0.24 * x)
F_dB = :(4.99 * laplacian(A * 10 * t)* 10 * t - x_grad(A * 10 * t)^2)

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A, B])
complex_expr = :((c) -> abs(c))
# color_expr = :((a) -> RGB(a.r, a.g, a.b))
color_expr = :((a,b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))

animate_system_2(ds, 720, 720, 2.0, 0.0005; color_expr, complex_expr)