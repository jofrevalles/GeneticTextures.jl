module ArtificialEvolution

include("RGB_index.jl")

using Images, ImageView, ImageFiltering
using CoherentNoise
using Colors

export mod,abs,X,Y,perlin,RGB_index,bw

w = 1080
l = 1800

# examples:
sampler_perlin = perlin_3d()
perlin_caller = (x,y,z) -> perlin(x,y,z,sampler_perlin)# do a setting function for this kind of issues
perlin_img = perlin_caller.(rand(ArtificialEvolution.RGB_index, w, l), 123, 0.01) # make sure you define a new for each

img = X.(rand(RGB_index, w, l)) .|> abs |> (x -> mod.(x, Y.(rand(RGB_index, w, l)))) |> (x -> mod.(x, abs.(Y.(rand(RGB_index, w, l))))) .|> (x ->  x * [1.8,2.6,3.5]) .|> (x -> x ^ [0.8, 1.2, 0.8]) |> (x -> (mod.(abs.(X.(rand(RGB_index, w, l))), x))) .|> (x-> x ^ 0.8)

ImageView.imshow(Images.colorview(RGB, (p -> p.val).(img)))
Images.save("15-01-22.png", Images.colorview(RGB, map(Images.clamp01nan, (p -> p.val).(img))))

end # module