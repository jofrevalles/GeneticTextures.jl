
# Example usage:
using GeneticTextures

F_A0 = GeneticTextures.CustomExpr(:(ifs(rand_scalar() > 0.97, 0.0, 1)))
F_dA = GeneticTextures.CustomExpr(:(-1.0 * laplacian(A)*B+0.4*neighbor_min(A; Δx=4, Δy=4)))

F_B0 = GeneticTextures.CustomExpr(:(ifs(and((x^2 + y^2) < 0.1, (x^2 + y^2) > 0.09), 0.0, 1.0)))
F_dB = GeneticTextures.CustomExpr(:(-1.0 * laplacian(B)*A-0.4*neighbor_min(B)))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A, B])
complex_expr = :((c) -> abs(c))

animate_system2(ds, 64, 64, 2.0, 0.1; complex_expr)