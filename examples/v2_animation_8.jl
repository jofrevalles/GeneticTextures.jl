using GeneticTextures

# Let's create a spiral wave
F_A0 = :(Complex(rand_scalar(), rand_scalar()) + 0.0)   
F_dA = :(laplacian(A) + grad_direction(abs(A)))

color_expr = :((a) ->  RGB(abs(a.r), abs(a.g), abs(a.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
# B = VariableDynamics(:B, F_B0, F_dB)

ds = DynamicalSystem([A])
width = height = 128
animate_system(ds, width, height, 80.0, 0.1; color_expr, complex_expr, normalize_img=true, renderer=:basic, adjust_brighness=false)
