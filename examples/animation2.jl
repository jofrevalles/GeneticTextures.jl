
# Example usage:
using GeneticTextures

F_A0 = :(ifs(rand_scalar() > 0.97, 0.0, 1))
F_A0 = :(cos(perlin_color(84.2126, grad_mag(mod, exp(x), 0.91 - 0.9128), abs(x), dissolve(y, Color(0.28, 0.47, 0.86), Color(0.28, 0.47, 0.86)))))
F_dA = :(-1.0 * laplacian(A)*B+0.4*neighbor_min(A; Δx=4, Δy=4))

F_B0 = :(ifs(and((x^2 + y^2) < 0.1, (x^2 + y^2) > 0.09), 0.0, 1.0))
F_dB = :(-1.0 * laplacian(B)*A-0.4*neighbor_min(B))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A, B])
complex_expr = :((c) -> abs(c))

animate_system2(ds, 64, 64, 2.0, 0.1; complex_expr)