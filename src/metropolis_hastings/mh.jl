module MetropolisHastings
using ..TPS


abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end

macro _check_fraction_domain(val, parameter_name)
    quote
        if !isnothing($(esc(val))) && ($(esc(val)) < 0.0 && $(esc(val)) > 1.0)
            throw(DomainError($(esc(val)), string($(esc(parameter_name)), " must be between 0 and 1 inclusive.")))
        end
    end
end

include("mh_sa_alg.jl")
include("mh_trajectory_alg.jl")

export gaussian_sa_algorithm, gaussian_trajectory_algorithm

end