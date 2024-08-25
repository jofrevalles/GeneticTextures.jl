using GeneticTextures
delta = 2
F_A0 = :((y+0.0)*1+0.0)
F_dA = :(neighbor_max_radius(neighbor_max_radius(C; Δr=$delta); Δr=$delta))

F_B0 = :(0.0+0.0)
F_dB = :(x_grad(C)+0.)

F_C = :((-rand_scalar()*1.68-0.12) + y)
F_dC = :(neighbor_ave_radius(grad_direction(B*0.25); Δr=$delta))


color_expr = :((a, b, c) ->  RGB(abs(a.r), abs(a.g), abs(a.b)))
#color_expr = :((a, b, c) -> RGB(clamp(c.r, 0., 1.0), clamp(c.g, 0.0, 1.0), clamp(c.b,0.0,1.0)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C, F_dC)

ds = DynamicalSystem([A, B, C])
width = height = 128
animate_system(ds, width, height, 200.0, 4.; color_expr, complex_expr, normalize_img=false, renderer=:basic, adjust_brighness=false)
