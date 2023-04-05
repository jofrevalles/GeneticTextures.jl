using Base

struct Color{T<:Real} <: AbstractArray{T, 1}
    r::T
    g::T
    b::T
end

Base.length(c::Color) = 3

Base.size(c::Color) = (3,)

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

function Base.iterate(c::Color, state::Int=1)
    if state > 3
        return nothing
    else
        return (c[state], state + 1)
    end
end