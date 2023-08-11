
# Example usage:
using GeneticTextures

F_dA = :(-1.0 * laplacian(A)*B+0.4*neighbor_min(A; Δx=4, Δy=4))
F_dB = :(-1.0 * laplacian(B)*A-0.4*neighbor_min(B))
# F_B0 = :(ifs(x^2 + y^2 < 0.1, 1.0, 0.0))
# draw a happy face
# circle = x^2 + y^2 < 0.1 && x^2 + y^2 > 0.05
# mouth = x^2 + y^2 < 0.05 && x^2 + y^2 > 0.03 && y < 0 && x > -0.01 && x < 0.01
# eye1 = (x + 0.05)^2 + (y + 0.05)^2 < 0.01
# eye2 = (x - 0.05)^2 + (y + 0.05)^2 < 0.01
# F_A0 = :(ifs(or(or(or(and((x^2 + y^2) < 0.1, (x^2 + y^2) > 0.09), and(and((x^2 + y^2) < 0.05, (x^2 + y^2) > 0.04), and(and(y > 0, x > -0.1), x < 0.1))),and((x + 0.05)^2 + (y + 0.05)^2 < 0.002 ,(x + 0.05)^2 + (y + 0.05)^2 > 0.001)),and((x - 0.05)^2 + (y + 0.05)^2 < 0.002, (x - 0.05)^2 + (y + 0.05)^2 > 0.001)), 0.0, 1.0))
F_B0 = :(ifs(and((x^2 + y^2) < 0.1, (x^2 + y^2) > 0.09), 0.0, 1.0))
F_A0 = :(ifs(rand_scalar(18.2345 * (1.0 + y * x)) > 0.99, 0.0, 1))
# color_expr = :((a, b) -> RGB(b, b, b))

F_A0 = :(ifs(rand_scalar() > 0.97, 0.0, 1))
# F_B0 = :(+0.032)
# F_dA = :(neighbor_min(A; Δx=2, Δy=2))
# F_dB = :(4.99* laplacian(A))

# F_B0 = :(Complex(0.4,1))

# F_C0 = :(Complex(0.4,1))
# F_dC = :(atan(laplacian(B*0.25; Δx=1, Δy=0), laplacian(B*0.25; Δx=0, Δy=1)))

# F_dB = :(laplacian(C, Δx=1, Δy=0)))
# F_dA = :(laplacian(C)))
A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
# C = VariableDynamics(:C, F_C0, F_dC)
ds = DynamicalSystem([A, B])

color_expr = :((a, b) -> RGB(a, b, a*b % 1.))
complex_expr = :((c) -> abs(c))

animate_system(ds, 128, 128, 2.0, 0.002; color_expr, complex_expr)