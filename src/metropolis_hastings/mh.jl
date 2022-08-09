module MetropolisHastings
using ..TPS


abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end
function TPS.step!(solution::T, alg::AbstractMetropolisHastingsAlg, args...; kwargs...) where {T<:TPSSolution}
    throw("Unimplemented method")
end

include("mh_sa_alg.jl")
include("mh_trajectory_alg.jl")

export MetropolisHastingsAlgorithm, gaussian_trajectory_algorithm

end