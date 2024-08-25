using GeneticTextures
begin
dt = 1.0

F_A0 = :(1.0+perlin_2d(123, 10*x, 10*y))
F_dA = :(neighbor_min_radius(A; Δr=2) - A)
F_B0 = :(0.1 * cos(2π * x))
F_dB = :(laplacian(B*3.5 - A*2.0))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
ds = DynamicalSystem([A, B])

color_expr = :((a, b) -> RGB(abs(a.r), abs(a.g), abs(a.b)))
complex_expr = :((c) -> abs(c))

# Set up and run the animation
width = height = 256
animate_system(ds, width, height, 50.0, dt; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end