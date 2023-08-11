using Base
using Base.Broadcast: BroadcastStyle, DefaultArrayStyle, broadcasted

# TODO: Consider changing the name of Color, since it can conflict with Images.jl and Colors.jl
mutable struct Color{T<:Real} <: AbstractArray{T, 1}
    r::T
    g::T
    b::T
end

function Color(v::AbstractVector{T}) where {T<:Real}
    length(v) == 3 || throw(ArgumentError("Vector must have exactly 3 elements"))
    return Color(v[1], v[2], v[3])
end

red(c::Color) = c.r
green(c::Color) = c.g
blue(c::Color) = c.b

struct ColorStyle <: Broadcast.BroadcastStyle end

Base.Broadcast.BroadcastStyle(::Type{<:Color}) = ColorStyle()
Base.Broadcast.BroadcastStyle(::ColorStyle, ::DefaultArrayStyle) = ColorStyle()
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle, ::ColorStyle) = ColorStyle()
Base.Broadcast.BroadcastStyle(::ColorStyle, ::ColorStyle) = ColorStyle()

Base.Broadcast.BroadcastStyle(::ColorStyle, ::Broadcast.BroadcastStyle) = ColorStyle()
Base.Broadcast.BroadcastStyle(::Broadcast.BroadcastStyle, ::ColorStyle) = ColorStyle()

function Base.Broadcast.broadcasted(::ColorStyle, f, cs::Color...)
    vecs = (Vector(c) for c in cs)
    result_vec = f.(vecs...)
    return Color(result_vec)
end

Base.Broadcast.broadcastable(x::Real) = Ref(x)

function Base.Broadcast.broadcasted(::ColorStyle, op, c1::Color, c2)
    return Color(op.(Vector(c1), c2))
end

function Base.Broadcast.broadcasted(::ColorStyle, op, c1, c2::Color)
    return Color(op.(c1, Vector(c2)))
end

function Base.similar(bc::Base.Broadcast.Broadcasted{ColorStyle}, ::Type{ElType}) where {ElType}
    return Color(ones(ElType, 3))
end

Base.length(c::Color) = 3

Base.size(c::Color) = (3,)

function Base.setindex!(c::Color, val, i::Int)
    if i == 1
        c.r = val
    elseif i == 2
        c.g = val
    elseif i == 3
        c.b = val
    else
        throw(BoundsError(c, i))
    end
    return val
end

function Base.getindex(c::Color, i::Int)
    if i == 1
        return c.r
    elseif i == 2
        return c.g
    elseif i == 3
        return c.b
    else
        throw(BoundsError(c, i))
    end
end

function Base.iterate(c::Color, state::Int=1)
    if state > 3
        return nothing
    else
        return (c[state], state + 1)
    end
end

function Base.show(io::IO, c::Color)
    print(io, "Color(", round(c.r, digits=2), ", ", round(c.g, digits=2), ", ", round(c.b, digits=2), ")")
end

using Colors
Colors.RGB(c::GeneticTextures.Color) = RGB(c.r, c.g, c.b)