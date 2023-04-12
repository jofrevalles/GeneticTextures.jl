function generate_unique_filename(folder::String, prefix::String, extension = "")
    existing_files = readdir(folder) .|> x ->  x[begin:findlast(isequal('.'),x)-1]
    counter = 1
    while true
        extension == "" ? filename = "$prefix$counter" : filename = "$prefix$counter.$extension"
        if filename ∉ existing_files
            return filename
        end
        counter += 1
    end
end