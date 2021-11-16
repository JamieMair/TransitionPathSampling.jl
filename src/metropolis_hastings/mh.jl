module MetropolisHastings
using ..TPS


abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end
include("mh_sa_alg.jl")
include("mh_trajectory_alg.jl")

export TPS.step!, MetropolisHastingsAlgorithm

end