using GeneticTextures

dt = 1.

F_A0 = :(Complex(4*x, 4*y))
# F_dA = :(A^2 + Complex(0.355, 0.355))
# F_dA = :(A^2 + Complex(0.74543, 0.11301))
# F_dA = :(neighbor_min(A; Δx=4, Δy=4)^2 + A^2 + Complex(0.74543,0.11301))
F_dA = :(A^2 + Complex(0.3,-0.01) - A/$dt)

F_dA = :(ifs(abs(A) > 2, 0, A^2 + Complex(0.3,-0.01) - A/$dt))
# F_dA = :(A*x-1*A/$dt)

F_B0 = :(0) # C will be used as a counter
F_dB = :(ifs(abs(A) > 2, -1/$dt, 1/$dt)) # stop iterating when A escapes

F_C0 = :(0)
F_dC = :(C + 1)


# Define the value expression
value_expr = :(begin
    smooth_modifier(k, z) = abs(z) > 2 ? k - log2(log2(abs(z))) : k

    angle_modifier(k, z) =  abs(z) > 2 ? angle(z)/2 * pi : k

    continuous_potential(k, z) = abs(z) > 2 ? 1 + k - log(log(abs(z))) / log(2) : k

    tester(k, z) = real(z)

    orbit_trap_modifier(k, z, trap_radius, max_iter, c) = begin
        trapped = false
        z_trap = z
        for i in 1:k
            z_trap = z_trap^2 + c
            if abs(z_trap) <= trap_radius
                trapped = true
                break
            end
        end

        if trapped
            return abs(z_trap) / trap_radius
        else
            return k / max_iter
        end
    end

    # (a, b, c) -> begin
    #     # Include orbit trap modifier
    #     trap_radius = 1  # Set this to whatever trap radius you want
    #     max_iter = 1  # Set this to your maximum number of iterations
    #     orbit_value = orbit_trap_modifier(c.r, a.r, trap_radius, max_iter, Complex(0.74543, 0.11301))  # c is your Julia constant
    # end

    (a, b) -> begin
        # Calculate the modified iteration count
        # modified_k = orbit_trap_modifier(b.r, a.r, 10, 1000, Complex(0.74543, 0.11301))
        modified_k = continuous_potential(b.r, a.r)
    end
end)

complex_expr = :((c) -> (c))

A = VariableDynamics(:A, F_A0, F_dA)
B = VariableDynamics(:B, F_B0, F_dB)
C = VariableDynamics(:C, F_C0, F_dC)
ds = DynamicalSystem([A,B])


w = h = 64
GeneticTextures.animate_system_3(ds, w, h, 200.0, dt;
normalize_img = true,
cmap = :curl,
value_expr,
complex_expr)
