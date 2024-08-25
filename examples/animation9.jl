using GeneticTextures
F_A0 = :(perlin_color(84.2126, grad_mag(mod, 0.4672, 0.8845 - 0.9128), perlin_2d(66.4615, dissolve(grad_dir(mod, or(x, x) + and(x, y), exp(y)), sinh(atan(x, y)), and(exp(y), x)), sqrt(sinh(dissolve(dissolve(y, 0.0344, x) - and(x, 0.9832), cosh(y) - abs(y), sin(y))))), Color(0.28, 0.4202, 0.86)))
F_dA = :(-1.0 * laplacian(A) * B + 0.4 * neighbor_min(A; Δx = 2, Δy = 2) * perlin_color(24.2126, 2.4, 3.2-t*90, 2.4+t*100))
F_B0 = :(ifs(and(x ^ 2.0 + y ^ 2.0 < 0.1, x ^ 2.0 + y ^ 2.0 > 0.09), 0.0, 1.0) * Color(0.7, 1.4, 0.9)* perlin_color(84.2126, 2.4, 3.2-t*10, 2.4+t*100))
F_dB = :(-1.0 * laplacian(B) * A * Color(0.7, 1.4, 0.9) * perlin_color(24.2126, 2.4, 3.2-t*80, 2.4+t*90)- 0.4 * neighbor_min(B))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
ds = DynamicalSystem([A,B])
complex_expr = :((c) -> abs(c))
# color_expr = :((a) -> RGB(a.r, a.g, a.b))
color_expr = :((a, b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))

animate_system_2(ds, 128, 128, 4.0, 0.01; color_expr, complex_expr)