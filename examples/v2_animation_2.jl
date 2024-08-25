using GeneticTextures
begin
dt = 1.0
maxiter = 100
# Julia set constant parameter
c = Complex(0.3, -0.01)

# Initial conditions and evolution rules
A0 = :(Complex((x) * 4, (y) * 4))
dA = :(ifs(abs(A) > 2.0, A, A^2 + $c)/$dt - A / $dt)

B0 = :(0.0+0.0)
dB = :(ifs(abs(A) > 2.0, - 1, $maxiter)/$dt - B / $dt)

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
        k = smooth_modifier(b.r, a.r)
        return k
    end
end)

complex_expr = :((c) -> abs(c))

# Set up and run the animation
width = height = 256
animate_system(ds, width, height, 50.0, dt; color_expr, complex_expr, normalize_img=true, renderer=:basic)
end