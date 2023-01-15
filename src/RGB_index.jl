using CoherentNoise
using Colors
using Base

mutable struct RGB_index
    val::RGB
    pos::CartesianIndex
    w::Int
    l::Int
end

Base.:(+)(p::RGB, val::AbstractFloat) = RGB(p.r + val, p.g + val, p.b + val)
Base.:(-)(p::RGB, val::AbstractFloat) = RGB(p.r - val, p.g - val, p.b - val)
Base.:(*)(p::RGB, val::AbstractFloat) = RGB(p.r * val, p.g * val, p.b * val)
Base.:(/)(p::RGB, val::AbstractFloat) = RGB(p.r / val, p.g / val, p.b / val)
Base.:(^)(p::RGB, val::AbstractFloat) = RGB(p.r ^ val, p.g ^ val, p.b ^ val)

Base.:(+)(p::RGB, vec::Vector) = RGB(p.r + vec[1], p.g + vec[2], p.b + vec[3])
Base.:(-)(p::RGB, vec::Vector) = RGB(p.r - vec[1], p.g - vec[2], p.b - vec[3])
Base.:(*)(p::RGB, vec::Vector) = RGB(p.r * vec[1], p.g * vec[2], p.b * vec[3])
Base.:(/)(p::RGB, vec::Vector) = RGB(p.r / vec[1], p.g / vec[2], p.b / vec[3])
Base.:(^)(p::RGB, vec::Vector) = RGB(p.r ^ vec[1], p.g ^ vec[2], p.b ^ vec[3])

Base.:(+)(p::RGB, p2::RGB) = RGB(p.r + p2.r, p.g + p2.g, p.b + p2.b)
Base.:(-)(p::RGB, p2::RGB) = RGB(p.r - p2.r, p.g - p2.g, p.b - p2.b)
Base.:(*)(p::RGB, p2::RGB) = RGB(p.r * p2.r, p.g * p2.g, p.b * p2.b)
Base.:(/)(p::RGB, p2::RGB) = RGB(p.r / p2.r, p.g / p2.g, p.b / p2.b)
Base.:(^)(p::RGB, p2::RGB) = RGB(p.r ^ p2.r, p.g ^ p2.g, p.b ^ p2.b)

Base.:(+)(p::RGB_index, val::AbstractFloat) = RGB_index(p.val + val, p.pos, p.w, p.l)
Base.:(-)(p::RGB_index, val::AbstractFloat) = RGB_index(p.val - val, p.pos, p.w, p.l)
Base.:(*)(p::RGB_index, val::AbstractFloat) = RGB_index(p.val * val, p.pos, p.w, p.l)
Base.:(/)(p::RGB_index, val::AbstractFloat) = RGB_index(p.val / val, p.pos, p.w, p.l)
Base.:(^)(p::RGB_index, val::AbstractFloat) = RGB_index(p.val ^ val, p.pos, p.w, p.l)

Base.:(+)(p::RGB_index, vec::Vector) = RGB_index(p.val + vec, p.pos, p.w, p.l)
Base.:(-)(p::RGB_index, vec::Vector) = RGB_index(p.val - vec, p.pos, p.w, p.l)
Base.:(*)(p::RGB_index, vec::Vector) = RGB_index(p.val * vec, p.pos, p.w, p.l)
Base.:(/)(p::RGB_index, vec::Vector) = RGB_index(p.val / vec, p.pos, p.w, p.l)
Base.:(^)(p::RGB_index, vec::Vector) = RGB_index(p.val ^ vec, p.pos, p.w, p.l)

Base.:(+)(p::RGB_index, p2::RGB_index) = RGB_index(p.val + p2.val, p.pos, p.w, p.l)
Base.:(-)(p::RGB_index, p2::RGB_index) = RGB_index(p.val - p2.val, p.pos, p.w, p.l)
Base.:(*)(p::RGB_index, p2::RGB_index) = RGB_index(p.val * p2.val, p.pos, p.w, p.l)
Base.:(/)(p::RGB_index, p2::RGB_index) = RGB_index(p.val / p2.val, p.pos, p.w, p.l)
Base.:(^)(p::RGB_index, p2::RGB_index) = RGB_index(p.val ^ p2.val, p.pos, p.w, p.l)


function Base.:rand(T::Type{<:RGB_index}, l::Int, w::Int)
    im = Matrix{RGB_index}(undef, l, w)
    for i in 1:l
        for j in 1:w
            im[i,j] = RGB_index(rand(RGB{Float16}), CartesianIndex(i,j), w, l)
        end
    end
    return im
end

function Base.:mod(p::RGB_index, p2::RGB_index)
    values = RGB(mod(p.val.r, p2.val.r), mod(p.val.g, p2.val.g), mod(p.val.b, p2.val.b))
    return RGB_index(values, p.pos, p.w, p.l)
end

function Base.:abs(p::RGB_index)
    p.val = RGB(abs(p.val.r), abs(p.val.g), abs(p.val.b))
    return p
end

function X(p::RGB_index)
    p.val = RGB(p.pos[2]/p.w - 0.5, p.pos[2]/p.w - 0.5, p.pos[2]/p.w - 0.5)
    return p
end

function Y(p::RGB_index)
    p.val = RGB(p.pos[1]/p.l - 0.5, p.pos[1]/p.l - 0.5, p.pos[1]/p.l - 0.5)
    return p
end

function perlin(p::RGB_index, seed::Int, factor::AbstractFloat, sampler)
    # sampler_perlin = perlin_3d()
    r = abs(sample(sampler, seed+p.pos[1]*factor, seed+p.pos[2]*factor, seed+1/factor))
    g = abs(sample(sampler, seed+p.pos[1]*factor, seed+p.pos[2]*factor, seed+2/factor))
    b = abs(sample(sampler, seed+p.pos[1]*factor, seed+p.pos[2]*factor, seed+3/factor))

    p.val = RGB(r, g, b)
    return p
end

function bw(p::RGB_index)
    vec = [0.299, 0.587, 0.114]
    val = p.val.r * vec[1] + p.val.g * vec[2] + p.val.b * vec[3]
    return RGB_index(RGB(val, val, val), p.pos, p.w, p.l)
end

# create a function that normalizes?