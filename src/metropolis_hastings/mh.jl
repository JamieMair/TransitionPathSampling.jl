module MetropolisHastings
using ..TPS


abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end
abstract type AbstractMetropolisHastingsCache end

function last_accepted(cache::AbstractMetropolisHastingsCache)
    if hasfield(typeof(cache), :last_accepted)
        return getfield(cache, :last_accepted)
    else
        error("Cache $(typeof(cache)) does not record the status of the last update.")
    end
end

function get_parameters(alg::AbstractMetropolisHastingsAlg)
    if hasfield(typeof(alg), :parameters)
        return getfield(alg, :parameters)
    end
    nothing
end

macro _check_fraction_domain(val, parameter_name)
    quote
        if !isnothing($(esc(val))) && ($(esc(val)) < 0.0 && $(esc(val)) > 1.0)
            throw(DomainError($(esc(val)), string($(esc(parameter_name)), " must be between 0 and 1 inclusive.")))
        end
    end
end

include("mh_sa_alg.jl")
include("mh_trajectory_alg.jl")

export gaussian_sa_algorithm, gaussian_trajectory_algorithm, last_accepted 

end