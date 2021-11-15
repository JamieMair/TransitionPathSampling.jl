function get_iterator(iter, args...; kwargs...)
    if typeof(iter)==Int
        return 1:iter
    else
        return iter
    end
end