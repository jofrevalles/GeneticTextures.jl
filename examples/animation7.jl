# Example usage:
using GeneticTextures

# F_A0= :(Color(Complex(x, y), Complex(y, x), Complex(abs(x - y), abs(x + y))))
# F_dA= :(min(A, Complex(1.0, 0.2)) * A + exp(laplacian(A * Complex(-1.2, 1.0))))

F_A0 = :(Complex(x, y))
F_dA = :(Complex(0, 1) * (min(A, 1.0) * A) - 0.7 * exp(neighbor_max(A) * Complex(0.2, -0.12)))

A = VariableDynamics(:A, F_A0, F_dA)


ds = DynamicalSystem([A])
complex_expr = :((c) -> abs(c))
# color_expr = :((a) -> RGB(a.r, a.g, a.b))
color_expr = :((a) -> RGB(abs(a.r), abs(a.g), abs(a.b)))

animate_system_2(ds, 64, 64, 2.0, 0.02; color_expr, complex_expr)