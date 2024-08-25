using GeneticTextures

# Let's create a spiral wave
F_A0 = :(Complex(4*x, 4*y) + 0.0)
F_dA = :(min(abs(A), 1.0)/A - 0.7*(max(A, Complex(0.2, -0.12))^3.5))

color_expr = :((a) ->  RGB(abs(a.r), abs(a.g), abs(a.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
# B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A])
width = height = 128
animate_system(ds, width, height, 80.0, 1.0; color_expr, complex_expr, normalize_img=true, renderer=:basic, adjust_brighness=false)
