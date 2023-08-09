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

function safe_getfield(field::Symbol)
    if isdefined(Main, field)
        return getfield(Main, field)
    elseif isdefined(Base, field)
        return getfield(Base, field)
    elseif isdefined(GeneticTextures, field)
        return getfield(GeneticTextures, field)
    else
        error("Symbol $field is not defined")
    end
end

function capture_function(func_expr::Expr)
    raw_str = string(func_expr)

    # Remove comments
    cleaned_str = replace(raw_str, r"\#\=.*\=\#" => "")

    # Remove newline characters and extra spaces
    cleaned_str = replace(cleaned_str, '\n' => " ")
    cleaned_str = replace(cleaned_str, r"\s+" => " ")

    return strip(cleaned_str)
end
