module MetropolisHastings
using ..TPS


abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end
function TPS.step!(solution::T, alg::AbstractMetropolisHastingsAlg, args...; kwargs...) where {T<:TPSSolution}
    throw("Unimplemented method")
end

include("mh_sa_alg.jl")
include("mh_trajectory_alg.jl")

export MetropolisHastingsAlgorithm, get_guassian_mh_alg, get_shooting_mh_alg, get_shooting_and_bridging_mh_alg

end