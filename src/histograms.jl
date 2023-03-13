module Histograms
using Base

abstract type AbstractHistogram end
Base.push!(::AbstractHistogram) = error("Unimplemented.")
reset!(::AbstractHistogram) = error("Unimplemented")
Base.count(::AbstractHistogram) = error("Unimplemented")

mutable struct FixedWidthHistogram{E, T, A<:AbstractArray{Int}} <: AbstractHistogram
    const bin_edges::LinRange{T, Int}
    const values::A
    const ignore_extrema::Val{E}
    total_entries::UInt
end

function FixedWidthHistogram(min_value, max_value, num_bins::Int; ignore_extrema=true)
    bin_edges = LinRange(min_value, max_value, num_bins+1)

    values = zeros(Int, num_bins + (ignore_extrema ? 0 : 2))
    return FixedWidthHistogram(bin_edges, values, Val(ignore_extrema), UInt(0))
end


function Base.push!(histogram::FixedWidthHistogram{IgnoreExtrema}, val) where {IgnoreExtrema}
    min_bin = minimum(histogram.bin_edges)
    max_bin = maximum(histogram.bin_edges)
    histogram.total_entries += 1
    if val < min_bin
        (IgnoreExtrema) || (histogram.values[begin] += 1)
        return nothing
    elseif val > max_bin
        (IgnoreExtrema) || (histogram.values[end] += 1)
        return nothing
    end
    # Find the bin index in bounds (left < val <= right). First bin has (left<=val<=right)
    bin_index = max(ceil(Int, (val - min_bin) / (max_bin-min_bin) * (length(histogram.bin_edges)-1)), 1)
    # Increment the count in the bin
    histogram.values[bin_index + (IgnoreExtrema ? 0 : 1)] += 1
    return nothing
end
function reset!(histogram::FixedWidthHistogram)
    fill!(histogram.values, 0)
    histogram.count = 0
    nothing
end
count(histogram::FixedWidthHistogram) = histogram.total_entries


export AbstractHistogram, FixedWidthHistogram, reset!
end