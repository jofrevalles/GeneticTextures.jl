using GeneticTextures

# Let's create a reaction-diffusion system
F_A0 = :(rand_scalar() + 0.0)
F_dA = :(laplacian(B) + 0.1)

F_B0 = :(rand_scalar() + 0.0)
F_dB = :(laplacian(A) + 0.1)

color_expr = :((a, b) ->  RGB(abs(a.r), abs(b.g), abs(a.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A, B])
width = height = 256
animate_system(ds, width, height, 80.0, 0.001; color_expr, complex_expr, normalize_img=true, renderer=:basic, adjust_brighness=false)
