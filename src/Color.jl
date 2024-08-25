import Base: sin, cos, tan, +, -, *, /, ^, sqrt, exp, log, abs, atan, asin, acos, sinh, cosh, tanh, sech, csch, coth, asec, acsc, acot, sec, csc, cot, mod, rem, fld, cld, ceil, floor, round, max, min
using Base
using Base.Broadcast: BroadcastStyle, DefaultArrayStyle, broadcasted
using Colors

# TODO: Consider changing the name of Color, since it can conflict with Images.jl and Colors.jl
mutable struct Color{T<:Number} <: AbstractArray{T, 1}
    r::T
    g::T
    b::T
end

function Color(v::AbstractVector{T}) where {T<:Number}
    length(v) == 3 || throw(ArgumentError("Vector must have exactly 3 elements"))
    return Color(v[1], v[2], v[3])
end

Color(n::Number) = Color(n, n, n)

red(c::Color) = c.r
green(c::Color) = c.g
blue(c::Color) = c.b

Base.isreal(c::Color) = isreal(c.r) && isreal(c.g) && isreal(c.b)
Base.real(c::Color) = Color(real(c.r), real(c.g), real(c.b))
Base.imag(c::Color) = Color(imag(c.r), imag(c.g), imag(c.b))
Base.abs(c::Color) = Color(abs(c.r), abs(c.g), abs(c.b))

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

Colors.RGB(c::Color) = RGB(c.r, c.g, c.b)

Color(val::Colors.RGB) = Color(val.r, val.g, val.b)
convert(::Type{Color}, val::Float64) = Color(val, val, val)
# Color(val::Colors.RGB{N0f8}) = Color(Float64(val.r), Float64(val.g), Float64(val.b))

unary_functions = [sin, cos, tan, sqrt, exp, log, asin, acos, atan, sinh, cosh, tanh, sech, csch, coth, asec, acsc, acot, sec, csc, cot, mod, rem, fld, cld, ceil, floor, round]
binary_functions = [+, -, *, /, ^, atan, mod, rem, fld, cld, ceil, floor, round, max, min]

# Automatically define methods
for func in unary_functions
    func = Symbol(func)
    @eval begin
        ($func)(c::Color) = Base.broadcast($func, c)
    end
end

for func in binary_functions
    func = Symbol(func)  # Get the function name symbol

    @eval begin
        ($func)(c1::Color, c2::Color) =  Base.broadcast($func, c1, c2)
        ($func)(c::Color, x::Number) = Base.broadcast($func, c, x)
        ($func)(x::Number, c::Color) = Base.broadcast($func, x, c)
    end
end

Base.isless(x::Number, y::Color) = isless(x, sum([y.r, y.g, y.b])/3.)
Base.isless(x::Color, y::Number) = isless(sum([x.r, x.g, x.b])/3., y)
Base.isless(x::Color, y::Color) = isless(sum([x.r, x.g, x.b])/3., sum([y.r, y.g, y.b])/3.)

Base.isequal(x::Color, y::Color) = isequal(x.r, y.r) && isequal(x.g, y.g) && isequal(x.b, y.b)
Base.isequal(x::Color, y::Number) = isequal(x.r, y) && isequal(x.g, y) && isequal(x.b, y)
Base.isequal(x::Number, y::Color) = isequal(x, y.r) && isequal(x, y.g) && isequal(x, y.b)