
# Example usage:
using GeneticTextures

F_A0= :(ifs(rand_scalar() > 0.97, 0.0, 1.0))
F_B0= :(Complex(0.4, 1.0))
F_dA= :(neighbor_min(A; Δx = 2, Δy = 2))
F_dB= :(4.99 * laplacian(A))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A, B])
complex_expr = :((c) -> abs(c))
color_expr = :((a, b) -> RGB(a.r * 0.8, abs(b.g - a.g), ((a.b * b.b) * 0.2) % 1.0))
animate_system_2(ds, 256, 256, 2.0, 0.002; color_expr, complex_expr)