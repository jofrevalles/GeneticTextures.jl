
# Example usage:
using GeneticTextures

# F_A0= :(Color(Complex(x, y), Complex(y, x), Complex(abs(x - y), abs(x + y))))
# F_dA= :(min(A, Complex(1.0, 0.2)) * A + exp(laplacian(A * Complex(-1.2, 1.0))))

F_A0 = :(ifs(rand_scalar() > 0.97, 0.0, 1.0))
F_dA = :(x * y + exp(laplacian(A * B * Complex(-1.2, 1.0))))

F_B0 = :(Complex(y, x))
F_dB = :(4.99 * laplacian(A + neighbor_min(A; Δx = 2, Δy = 2)))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A, B])
complex_expr = :((c) -> abs(c))
# color_expr = :((a) -> RGB(a.r, a.g, a.b))
color_expr = :((a, b) -> RGB(a.r * 0.8, a.g, ((a.b) * 0.2) % 1.0))

animate_system_2(ds, 64, 64, 2.0, 0.002; color_expr, complex_expr)