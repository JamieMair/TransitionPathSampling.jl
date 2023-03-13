module Corrections
using Logging
using Distributions
import BSON
using Random
import LinearAlgebra: I

export get_cached_correction_cdf, precompute_correction_cdf, sample, CorrectionDistribution

sigmoid(x) = inv(1+exp(-x))

struct CorrectionDistribution
    x_values::LinRange{Float64, Int64}
    y_values::Vector{Float64}
    cumulative_y_values::Vector{Float64}
    sigma::Float64
    N::Int
    V::Float64
    lambda::Float64
end

function _sample(r::AbstractFloat, corr::CorrectionDistribution)
    # Binary search for corresponding points
    top = length(corr.x_values)
    bottom = 1
    while (top - bottom > 1)
        mid = (top+bottom) ÷ 2
        if (r > corr.cumulative_y_values[mid])
            bottom = mid
        else
            top = mid
        end
    end

    y0 = corr.cumulative_y_values[bottom]
    y1 = corr.cumulative_y_values[top]
    alpha = y1 != y0 ? ((r-y0)/(y1-y0)) : zero(y0)
    x0 = corr.x_values[bottom]
    x1 = corr.x_values[top]
    return (alpha * x1 + (1-alpha) * x0)
end

sample(rng, corr::CorrectionDistribution) = _sample(rand(rng), corr)
sample(corr::CorrectionDistribution) = _sample(rand(), corr)

function _parse_bid_corrections(path=joinpath(pwd(), ".cache", "corrections.txt"))
    xs = Float64[]
    ys = Float64[]
    open(path, "r") do file
        while !eof(file)
            (x, y) = split(readline(file), '\t') .|> strip .|> (s->parse(Float64, s))
            push!(xs, x)
            push!(ys, y)
        end
    end
    x_range = LinRange(minimum(xs), maximum(xs), length(xs))
    @assert all(isapprox.(x_range, xs))
    cumsum_ys = cumsum(ys)
    cumsum_ys ./= cumsum_ys[end]
    return CorrectionDistribution(x_range, ys, cumsum_ys, 1.0, (length(ys)-1)÷2, maximum(x_range)/2, 10.0)
end

function precompute_correction_cdf(V, N, lambda, sigma)
    X = LinRange(-V, V, 4N+1)
    Y = LinRange(-2V, 2V, 2N+1)

    M = zeros(length(X), length(Y))
    normal_dist = Normal(0, 1)
    @inbounds for j in axes(M, 2), i in axes(M, 1)
        M[i, j] = cdf(normal_dist, (X[i]-Y[j])/sigma)
    end

    v = reshape(sigmoid.(X), :, 1)

    u = reshape(inv(M'*M + lambda * I) * M' * v, :)

    u .= max.(u, 0)

    return Y, reshape(u, :)
end

_default_cache_path(V, N, lambda, sigma) = joinpath(pwd(), ".cache", "correction_cdf_$(V)_$(N)_$(lambda)_$(sigma).bson")

function cache_correction_cdf(cache_path=nothing; V=10.0, N=4000, lambda=25.0, sigma=1.0)
    if isnothing(cache_path)
        cache_path = _default_cache_path(V, N, lambda, sigma)
    end
    Y, CY = precompute_correction_cdf(V, N, lambda, sigma)
    if !isdir(dirname(cache_path))
        mkpath(dirname(cache_path))
        @info "Created $(dirname(cache_path))"
    end
    cumsum_CY = cumsum(CY)
    cumsum_CY ./= cumsum_CY[end]
    correction_data = CorrectionDistribution(Y, CY, cumsum_CY, sigma, N, V, lambda)
    BSON.@save cache_path correction_data
    nothing
end

function get_cached_correction_cdf(cache_path=nothing; V=10.0, N=4000, lambda=25.0, sigma=1.0)
    if isnothing(cache_path)
        cache_path = _default_cache_path(V, N, lambda, sigma)
    end

    if !isfile(cache_path)
        cache_correction_cdf(cache_path; V, N, lambda, sigma)
    end
    correction_data = nothing
    BSON.@load cache_path correction_data
    return correction_data
end

end