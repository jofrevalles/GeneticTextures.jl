begin
    using GeneticTextures
    delta = 2
    F_A0 = :(real(x)+imag(y))
    F_dA = :((min(A, 1.0) / A) -0.7 * 3.5^(max(A, (0.2 -0.12*im))))
    #color_expr  = :((a, b) -> RGB(abs(a.r), abs(a.g), abs(a.b)))
    #color_expr  = :((a, b) -> RGB((x -> (x > 0.) ? x*0.001 : -x*0.001).([b.r, b.g,b.b])...))
    color_expr  = :((b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))
    #color_expr  = :((a, b, c) -> RGB((x -> (x > 0.) ? log(x) : -log(-x)).([c.r, c.g,c.b])...))
    #color_expr  = :((a, b) -> RGB(clamp(0.002*b.r, 0, 1), clamp(0.002*b.g, 0, 1), clamp(0.002*b.b, 0, 1)))
    complex_expr = :((c) -> imag(c))

    A = VariableDynamics(:A, F_A0, F_dA)

    ds = DynamicalSystem([A])
    width = height = 32
    animate_system(ds, width, height, 50.0, 0.12; color_expr, complex_expr, normalize_img=true, renderer=:basic)
    end

begin
    using GeneticTextures
    delta = 2
    F_A0 = :(Complex(4*x, 4*y))
    F_dA = :(A^2 + Complex(0.355, 0.355))

    F_B0 = :(0.0+0.0im)
    F_dB = :(ifs(abs(A) > 2, B + 0.1, B))

    color_expr = :((a, b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))
    complex_expr = :((c) -> real(c))

    A = VariableDynamics(:A, F_A0, F_dA)
    B = VariableDynamics(:B, F_B0, F_dB)
    ds = DynamicalSystem([A,B])

    #color_expr  = :((a, b) -> RGB(abs(a.r), abs(a.g), abs(a.b)))
    #color_expr  = :((a, b) -> RGB((x -> (x > 0.) ? x*0.001 : -x*0.001).([b.r, b.g,b.b])...))
    color_expr  = :((a, b) -> RGB(abs(b.r), abs(b.g), abs(b.b)))
    #color_expr  = :((a, b, c) -> RGB((x -> (x > 0.) ? log(x) : -log(-x)).([c.r, c.g,c.b])...))
    #color_expr  = :((a, b) -> RGB(clamp(0.002*b.r, 0, 1), clamp(0.002*b.g, 0, 1), clamp(0.002*b.b, 0, 1)))
    complex_expr = :((c) -> imag(c))

    A = VariableDynamics(:A, F_A0, F_dA)
    B = VariableDynamics(:B, F_B0, F_dB)
    ds = DynamicalSystem([A, B])
    width = height = 32
    animate_system(ds, width, height, 50.0, 0.12; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end

begin
    F_A0 = :(Complex(4*x, y))
    F_dA = :(A^2 + Complex(0.74543, 0.11301))

    F_B0 = :(0.0+0.0)
    F_dB = :(ifs(abs(A) > 2, B + 1, B))

    F_C0 = :(0.0+0.0) # C will be used as a counter
    F_dC = :(C + 1)


    # Define the value expression
    color_expr = :(begin
        smooth_modifier(k, z) = abs(z) > 2 ? k - log2(log2(abs(z))) : k

        angle_modifier(k, z) =  abs(z) > 2 ? angle(z)/2 * pi : k

        (a, b, c) -> begin
            # Calculate the modified iteration count
            modified_k = angle_modifier(c.r, a.r)
        end
    end)

    complex_expr = :((c) -> real(c))

    A = VariableDynamics(:A, F_A0, F_dA)
    B = VariableDynamics(:B, F_B0, F_dB)
    C = VariableDynamics(:C, F_C0, F_dC)
    ds = DynamicalSystem([A,B,C])

    width = height = 32
    animate_system(ds, width, height, 50.0, 0.01; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end

begin
    using GeneticTextures
F_A0 = :(y+0.0)
F_dA = :(neighbor_max(neighbor_max(C; Δx=2, Δy=2); Δx=2, Δy=2))

F_B0 = :(1.0+0.0)
F_dB = :(x_grad(C))

F_C = :((1 - rand_scalar()*1.68+0.12) + y)
F_dC = :(neighbor_ave(grad_direction(B * 0.25; Δx=2, Δy=2)))


color_expr = :((a, b, c) -> RGB(abs(c.r), abs(c.g), abs(c.b)))
complex_expr = :((c) -> real(c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C, F_dC)

ds = DynamicalSystem([A, B, C])
width = height = 64
animate_system(ds, width, height, 10.0, 0.1; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end

begin
    dt = 0.1
    A0 = :(Complex(x, y)+0.)
    dA = :(ifs(abs(A) > 2, A, A^2 + Complex(0.2, 0.4) - A/$dt))

    # B will count the iterations
    B0 = :(0.0+0.0)
    dB = :(ifs(abs(A) > 2, B - 1, B + 1) - B/$dt)

    A = VariableDynamics(:A, A0, dA)
    B = VariableDynamics(:B, B0, dB)
    ds = DynamicalSystem([A, B])

    # color_expr = :((a) -> RGB(a.r, a.g, a.b))

    color_expr = :(begin

    smooth_modifier(k, z) = abs(z) > 2 ? k - log2(log2(abs(z))) : k

    angle_modifier(k, z) =  abs(z) > 2 ? angle(z)/2 * pi : k

    f_ = smooth_modifier
    complex_f = abs
    (a, b) -> begin
        # Calculate the modified iteration count
        modified_k = RGB(complex_f(f_(b.r, a.r)), complex_f(f_(b.g, a.g)), complex_f(f_(b.b, a.b)))
    end
end)
    complex_expr = :((c) -> abs(c))

    width = height = 256
    animate_system(ds, width, height, 50.0, dt; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end

using GeneticTextures
begin
dt = 1.0
maxiter = 100
# Julia set constant parameter
c = Complex(0.8, -0.156)

# Initial conditions and evolution rules
A0 = :(Complex((x) * 4, (y) * 4))
dA = :(ifs(abs(A) > 2.0, A, A^2 + $c)/$dt - A / $dt)

B0 = :(0.0+0.0)
dB = :(ifs(abs(A) > 2.0, B - 1, B + 1)/$dt - B / $dt)

A = VariableDynamics(:A, A0, dA)
B = VariableDynamics(:B, B0, dB)
ds = DynamicalSystem([A, B])

# Color expression to visualize the fractal
color_expr = :(begin
    function smooth_modifier(k, z)
        return abs(z) > 2 ? k - log2(log2(abs(z)))  : k
    end

    function angle_modifier(k, z)
        if abs(z) > 2
            return angle(z) / (2 * pi)
        else
            return k
        end
    end

    function continuous_potential_modifier(k, z)
        if abs(z) > 2
            return 1 + k - log(log(abs(z))) / log(2)
        else
            return k
        end
    end

    (a, b) -> begin
        k = continuous_potential_modifier(a.r, b.r)
        return k
    end
end)

complex_expr = :((c) -> abs(c))

# Set up and run the animation
width = height = 256
animate_system(ds, width, height, 50.0, dt; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end
