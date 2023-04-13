using CoherentNoise
using Base
using Base.Broadcast

struct PerlinStyle <: Broadcast.BroadcastStyle end

Base.Broadcast.BroadcastStyle(::Type{<:CoherentNoise.Perlin{2}}) = PerlinStyle()
Base.Broadcast.BroadcastStyle(::PerlinStyle, ::DefaultArrayStyle) = PerlinStyle()
Base.Broadcast.BroadcastStyle(::DefaultArrayStyle, ::PerlinStyle) = PerlinStyle()
Base.Broadcast.BroadcastStyle(::PerlinStyle, ::PerlinStyle) = PerlinStyle()

Base.Broadcast.BroadcastStyle(::PerlinStyle, ::Broadcast.BroadcastStyle) = PerlinStyle()
Base.Broadcast.BroadcastStyle(::Broadcast.BroadcastStyle, ::PerlinStyle) = PerlinStyle()

function Base.Broadcast.broadcasted(::PerlinStyle, f, sampler::CoherentNoise.Perlin{2}, cs::Color...)
    vecs = (Vector(c) for c in cs)
    result_vec = f.(sampler, vecs...)
    return Color(result_vec)
end

function Base.Broadcast.broadcasted(::PerlinStyle, op, sampler::CoherentNoise.Perlin{2}, c1::Color, c2)
    return Color(op.(sampler, Vector(c1), c2))
end

function Base.Broadcast.broadcasted(::PerlinStyle, op, sampler::CoherentNoise.Perlin{2}, c1, c2::Color)
    return Color(op.(sampler, c1, Vector(c2)))
end

Base.Broadcast.broadcastable(sampler::CoherentNoise.Perlin{2}) = Ref(sampler)