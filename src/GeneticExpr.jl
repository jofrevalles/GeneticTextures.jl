using Base: show

struct GeneticExpr
    expr::Expr
end

function GeneticExpr(x::Union{Number, Symbol, Color})
    return x
end

function Base.show(io::IO, c_expr::GeneticExpr)
    function short_expr(expr)
        if expr.head == :call && length(expr.args) > 0
            new_args = Any[]
            for arg in expr.args
                if arg isa Expr
                    push!(new_args, short_expr(arg))
                elseif arg isa Number || arg isa Color
                    push!(new_args, round.(arg, digits=4))
                else
                    push!(new_args, arg)
                end
            end
            return Expr(expr.head, new_args...)
        end
        return expr
    end

    show(io, short_expr(c_expr.expr))
end