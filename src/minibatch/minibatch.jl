module Minibatch
import Lazy: @forward
using ..TransitionPathSampling
import ..TransitionPathSampling.MetropolisHastings: AbstractMetropolisHastingsAlg

mutable struct SAProposedChange{T, Q}
    const old_state::T
    const new_state::T
    old_observation::Q
    new_observation::Q
end
struct TrajectoryProposedChange{T, I, L}
    old_state::T
    new_state::T
    changed_indices::I
    old_losses::L
    new_losses::L
end
export SAProposedChange, TrajectoryProposedChange

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
struct BatchMHCache{A<:TransitionPathSampling.MetropolisHastings.AbstractMetropolisHastingsCache, B<:BatchObservable, C<:Acceptance.AbstractMinibatchAcceptanceCache} <: AbstractBatchMHCache
    inner_cache::A
    observable::B
    acceptance_cache::C
end
get_acceptance_cache(cache::BatchMHCache) = cache.acceptance_cache
function TransitionPathSampling.MetropolisHastings.last_accepted(cache::BatchMHCache)
    return TransitionPathSampling.MetropolisHastings.last_accepted(cache.inner_cache)
end
function TransitionPathSampling.MetropolisHastings.set_last_acceptance!(cache::BatchMHCache, accepted)
    return TransitionPathSampling.MetropolisHastings.set_last_acceptance!(cache.inner_cache, accepted)
end
TransitionPathSampling.MetropolisHastings.get_observable(cache::BatchMHCache) = cache.observable

function build_acceptance_cache(alg::MinibatchMHAlg, observable::BatchObservable, ::Nothing)
    correction = Corrections.get_cached_correction_cdf()
    
    return Acceptance.build_cache(observable.total_samples, observable.batch_size, correction)
end
function build_acceptance_cache(alg::MinibatchMHAlg, observable::BatchObservable, config::Dict{Symbol, Any})
    correction = Corrections.get_cached_correction_cdf()
    
    return Acceptance.build_cache(observable.total_samples, observable.batch_size, correction; config...)
end

function TransitionPathSampling.generate_cache(alg::MinibatchMHAlg, problem::TPSProblem, args...; minibatch_config::Union{Dict{Symbol, Any}, Nothing}=nothing, kwargs...)
    inner_cache = TransitionPathSampling.generate_cache(alg.inner_algorithm, problem)
    observable = TransitionPathSampling.get_observable(problem)
    acceptance_cache = build_acceptance_cache(alg, observable, minibatch_config)
    return BatchMHCache(inner_cache, observable, acceptance_cache)
end



function get_proposed_change(cache::BatchMHCache{A}, state) where A<:TransitionPathSampling.MetropolisHastings.GaussianSACache
    T = typeof(cache.inner_cache.last_observation)
    return SAProposedChange(state, cache.inner_cache.state, T(0), T(0))
end
function get_proposed_change(cache::BatchMHCache{A}, state) where A<:TransitionPathSampling.MetropolisHastings.GaussianMHTrajectoryCache
    return TrajectoryProposedChange(state, cache.inner_cache.state_cache, cache.inner_cache.indices_changed, cache.inner_cache.last_observation, cache.inner_cache.cached_observation)
end
function update!(cache::BatchMHCache{A}, proposed_change::SAProposedChange, state, accept) where A<:TransitionPathSampling.MetropolisHastings.GaussianSACache
    if accept
        TransitionPathSampling.MetropolisHastings.apply!(state, cache.inner_cache)
        cache.inner_cache.last_observation = proposed_change.new_observation
    else
        cache.inner_cache.last_observation = proposed_change.old_observation
    end
    nothing
end
function update!(cache::BatchMHCache{A}, proposed_change::TrajectoryProposedChange, states, accept) where A<:TransitionPathSampling.MetropolisHastings.GaussianMHTrajectoryCache
    if accept
        TransitionPathSampling.MetropolisHastings.apply!(states, cache.inner_cache)
        # Update the losses
        for i in eachindex(proposed_change.new_losses, proposed_change.old_losses)
            proposed_change.old_losses[i] = proposed_change.new_losses[i]
        end
    end

    cache.inner_cache.total_observation = sum(proposed_change.old_losses)
    nothing
end

function TransitionPathSampling.MetropolisHastings.acceptance!(cache::BatchMHCache, states, alg::MinibatchMHAlg)
    acceptance_cache = get_acceptance_cache(cache)
    proposed_change = get_proposed_change(cache, states)

    a = Acceptance.minibatch_acceptance!(acceptance_cache, proposed_change, cache.observable, alg.bias);

    update!(cache, proposed_change, states, a)

    return a
end

function TransitionPathSampling.MetropolisHastings.perturb!(cache::BatchMHCache, alg::MinibatchMHAlg, state)
    TransitionPathSampling.MetropolisHastings.perturb!(cache.inner_cache, alg.inner_algorithm, state)
end

function TransitionPathSampling.MetropolisHastings.get_last_observation!(cache::BatchMHCache)
    TransitionPathSampling.MetropolisHastings.get_last_observation!(cache.inner_cache)
end

export MinibatchMHAlg, AbstractBatchObservable, BatchObservable
export summarise, MeanLossSummary, AbstractLossSummaryMethod
import .Acceptance: AbstractBatchLossFn, select_samples!, calculate_losses!, calculate_delta_losses!
export AbstractBatchLossFn, select_samples!, calculate_losses!, calculate_delta_losses!

end