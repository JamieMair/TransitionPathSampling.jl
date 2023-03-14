module Minibatch
import Lazy: @forward
using ..TransitionPathSampling
import ..TransitionPathSampling.MetropolisHastings: AbstractMetropolisHastingsAlg

include("corrections.jl")
include("acceptance.jl")

using .Acceptance
using .Corrections

struct MinibatchMHAlg{A<:AbstractMetropolisHastingsAlg, T} <: AbstractMetropolisHastingsAlg
    inner_algorithm::A
    bias::T
end

export MinibatchMHAlg

# TODO:
# 1. Add a "batch" observable, containing the dataset, model etc
# 2. "batch" observable should produce an array of "deltas"
# 3. Create a new cache which includes the cache of the inner algorithm, along with the acceptance and correction distribution caches
# 4. Write some tests, which will run the algorithm, and check for statistical differences with a mock observable
# 5. Allow parameters to be annealed over time

abstract type AbstractBatchObservable <: TransitionPathSampling.AbstractObservable end
abstract type AbstractLossSummaryMethod end
struct MeanLossSummary <: AbstractLossSummaryMethod end
summarise(::AbstractLossSummaryMethod, losses) = error("unimplemented.")
summarise(::MeanLossSummary, losses) = sum(losses)/length(losses)
Base.@kwdef struct BatchObservable{F<:Acceptance.AbstractBatchLossFn, S<:AbstractLossSummaryMethod} <: AbstractBatchObservable
    batch_deltas_fn::F
    batch_size::Int
    total_samples::Int
    summarise_method::S = MeanLossSummary()
end
function TransitionPathSampling.observe(observable::BatchObservable, state)
    losses = Acceptance.calculate_losses!(observable.batch_deltas_fn, state)
    summary = summarise(observable.summarise_method, losses)
    return summary
end
function TransitionPathSampling.observe(observable::BatchObservable, states::AbstractArray{<:AbstractArray})
    return [TransitionPathSampling.observe(observable, state) for state in states]
end
function TransitionPathSampling.observe!(cache, observable, states, state_indices)
    for i in state_indices
        cache[i] = TransitionPathSampling.observe(observable, states[i])
    end
    nothing
end
@forward BatchObservable.batch_deltas_fn Acceptance.select_samples!, Acceptance.calculate_losses!, Acceptance.calculate_delta_losses!

abstract type AbstractBatchMHCache <: TransitionPathSampling.MetropolisHastings.AbstractMetropolisHastingsCache end
get_acceptance_cache(alg::AbstractBatchMHCache) = error("unimplemented")
mutable struct BatchMHCache{A<:TransitionPathSampling.MetropolisHastings.AbstractMetropolisHastingsCache, B<:BatchObservable, C<:Acceptance.AbstractMinibatchAcceptanceCache, D<:Union{<:Number, AbstractArray{<:Number}}} <: AbstractBatchMHCache
    const inner_cache::A
    const observable::B
    const acceptance_cache::C
    last_observations::D
    cached_observations::D
    last_accepted::Bool
end
get_acceptance_cache(cache::BatchMHCache) = cache.acceptance_cache
TransitionPathSampling.MetropolisHastings.get_observable(cache::BatchMHCache) = cache.observable
@forward BatchMHCache.inner_cache TransitionPathSampling.MetropolisHastings.proposed_changed_state, TransitionPathSampling.MetropolisHastings.original_changed_state

function build_acceptance_cache(alg::MinibatchMHAlg, observable::BatchObservable, ::Nothing)
    correction = Corrections.get_cached_correction_cdf()
    
    return Acceptance.build_cache(observable.total_samples, observable.batch_size, correction)
end
function build_acceptance_cache(alg::MinibatchMHAlg, observable::BatchObservable, config::Dict{Symbol, Any})
    correction = Corrections.get_cached_correction_cdf()
    
    return Acceptance.build_cache(observable.total_samples, observable.batch_size, correction; config...)
end

function TransitionPathSampling.generate_cache(alg::MinibatchMHAlg, problem::TPSProblem; minibatch_config::Union{Dict{Symbol, Any}, Nothing}=nothing)
    inner_cache = TransitionPathSampling.generate_cache(alg.inner_algorithm, problem)
    observable = TransitionPathSampling.get_observable(problem)
    current_observations = TransitionPathSampling.MetropolisHastings.get_cached_observations(inner_cache)
    observation_cache = similar(current_observations) # TODO FIX ME PLEASE!!!!!!
    # NOTES: 
    # ABSTRACT AWAY THE OBSERVATION CHECK
    acceptance_cache = build_acceptance_cache(alg, observable, minibatch_config)
    return BatchMHCache(inner_cache, observable, acceptance_cache, current_observations, observation_cache, false)
end


function TransitionPathSampling.MetropolisHastings.acceptance!(cache::BatchMHCache, states, alg::MinibatchMHAlg)
    acceptance_cache = get_acceptance_cache(cache)
    inner_cache = cache.inner_cache
    new_state = TransitionPathSampling.MetropolisHastings.proposed_changed_state(inner_cache)
    old_state = TransitionPathSampling.MetropolisHastings.original_changed_state(inner_cache, states)

    a, quants = Acceptance.minibatch_acceptance!(acceptance_cache, old_state, new_state, cache.observable, alg.bias);

    if a
        TransitionPathSampling.MetropolisHastings.apply!(states, inner_cache)
    end
    return a
end

function TransitionPathSampling.MetropolisHastings.perturb!(cache::BatchMHCache, alg::MinibatchMHAlg, state)
    TransitionPathSampling.MetropolisHastings.perturb!(cache.inner_cache, alg.inner_algorithm, state)
end

function TransitionPathSampling.MetropolisHastings.get_last_observation!(cache::BatchMHCache)
    NaN # TODO IMPLEMENT THIS PROPERLY
end

export MinibatchMHAlg, AbstractBatchObservable, BatchObservable
export summarise, MeanLossSummary, AbstractLossSummaryMethod
import .Acceptance: AbstractBatchLossFn, select_samples!, calculate_losses!, calculate_delta_losses!
export AbstractBatchLossFn, select_samples!, calculate_losses!, calculate_delta_losses!


end