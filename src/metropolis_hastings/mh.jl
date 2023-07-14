module MetropolisHastings
using ..TransitionPathSampling


abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end
abstract type AbstractMetropolisHastingsCache end


perturb!(cache::AbstractMetropolisHastingsCache, alg::AbstractMetropolisHastingsAlg, state) = error("unimplemented")
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

function get_observable(cache::AbstractMetropolisHastingsCache)
    if hasfield(typeof(cache), :observable)
        return getfield(cache, :observable)
    else
        error("Cache $(typeof(cache)) does not have the observable field.")
    end
end

function set_last_acceptance!(cache::AbstractMetropolisHastingsCache, acceptance)
    if hasfield(typeof(cache), :last_accepted)
        setfield!(cache, :last_accepted, acceptance)
        nothing
    else
        error("Cache $(typeof(cache)) does not have the last_accepted field.")
    end
end


get_last_observation(cache::AbstractMetropolisHastingsCache) = error("unimplemented")

macro _check_fraction_domain(val, parameter_name)
    quote
        if !isnothing($(esc(val))) && ($(esc(val)) < 0.0 && $(esc(val)) > 1.0)
            throw(DomainError($(esc(val)), string($(esc(parameter_name)), " must be between 0 and 1 inclusive.")))
        end
    end
end

include("mh_sa_alg.jl")
include("mh_trajectory_alg.jl")


function TransitionPathSampling.step!(cache::AbstractMetropolisHastingsCache, solution::TPSSolution, alg::AbstractMetropolisHastingsAlg, iter, args...; kwargs...) 
    state = TransitionPathSampling.get_current_state(solution)
    perturb!(cache, alg, state)
    accept = acceptance!(cache, state, alg)
    set_last_acceptance!(cache, accept)
    if accept
        TransitionPathSampling.set_current_state!(solution, state)
    end
    TransitionPathSampling.set_observation!(solution, iter, get_last_observation(cache))
    # ToDo specialise on the type of solution to record more details
    nothing
end

export gaussian_sa_algorithm, gaussian_trajectory_algorithm, last_accepted 

end