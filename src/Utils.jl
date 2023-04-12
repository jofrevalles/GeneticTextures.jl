function generate_unique_filename(folder::String, prefix::String, extension::String)
    existing_files = readdir(folder)
    counter = 1
    while true
        filename = "$prefix$counter.$extension"
        if filename ∉ existing_files
            return filename
        end
        counter += 1
    end
end