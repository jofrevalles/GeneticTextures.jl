using GeneticTextures

F_A0 = :(sqrt(perlin_color(67.9588, x, x, Color(0.26, 0.2, 0.13))))
F_dA = :(-1.0 * laplacian(A) * B + 0.4 * neighbor_min(A; Δx = 2, Δy = 2) * perlin_color(24.2126, 2.4, 3.2-t*90, 2.4+t*100))
F_B0 = :(ifs(and(x ^ 2.0 + y ^ 2.0 < 0.1, x ^ 2.0 + y ^ 2.0 > 0.09), 0.0, 1.0) * Color(0.7, 1.4, 0.9)* perlin_color(84.2126, 2.4, 3.2-t*10, 2.4+t*100))
F_dB = :(-1.0 * laplacian(B) * A * Color(0.7, 1.4, 0.9) * perlin_color(24.2126, 2.4, 3.2-t*80, 2.4+t*90)- 0.4 * neighbor_min(B))



color_expr = :((a, b) -> RGB(abs(a.r), abs(a.g), abs(a.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
ds = DynamicalSystem([A,B])

w = h = 64
animate_system_2(ds, w, h, 200.0, 0.1; color_expr, complex_expr)
