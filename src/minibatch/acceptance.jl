module Acceptance
import ..Minibatch.Corrections: CorrectionDistribution, sample, get_cached_correction_cdf
import ..Minibatch.TransitionPathSampling.Histograms: AbstractHistogram, FixedWidthHistogram
import ..Minibatch: SAProposedChange, TrajectoryProposedChange
import Lazy
using Random

export minibatch_acceptance!, build_cache

abstract type AbstractBatchLossFn end
"""
Sets the samples to calculate the losses with.
"""
select_samples!(::AbstractBatchLossFn, indices) = error("Unimplemented")
"""
Calculates the sample-wise losses for a set of parameters. Returns a vector of sample losses.
"""
calculate_losses!(::AbstractBatchLossFn, params) = error("Unimplemented")
"""
Calculates the sample-wise differences of loss using a proposed change (Either SAProposedChange or TrajectoryProposedChange). Returns the delta losses as an array.
"""
calculate_delta_losses!(::AbstractBatchLossFn, proposed_change) = error("Unimplemented")


Base.@kwdef struct MinibatchQuantities{T<:AbstractFloat}
    mean_delta::T = 0.0
    var_delta::T = 0.0
    mean_abs_delta::T = 0.0
    mean_abs_delta_cubed::T = 0.0
    error_estimate::T = 0.0
    num_samples::Int = 0
end


error_estimate(mean_abs_delta_cubed, mean_abs_delta, minibatch_size) = (6.4 * mean_abs_delta_cubed + 2 * mean_abs_delta) / sqrt(minibatch_size)

function calculate_totals(deltas::AbstractArray, s, old_mean_delta, old_n)
    total_delta = sum(deltas)
    total_delta_sq = sum(x->x^2, deltas)
    total_abs_delta = zero(eltype(deltas))
    total_abs_delta_cubed = zero(eltype(deltas))

    mean_delta_no_s = (old_mean_delta*old_n/(-s) + total_delta) / (length(deltas)+old_n)

    total_delta *= -s
    s_sq = s*s
    total_delta_sq *= s_sq
    
    @simd for delta in deltas
        abs_diff = abs(delta - mean_delta_no_s)
        total_abs_delta += abs_diff
        total_abs_delta_cubed += abs_diff ^ 3
    end

    return total_delta, total_delta_sq, total_abs_delta, total_abs_delta_cubed
end
@inline function update_mean(old_mean, old_n, delta_total, total_samples)
    return (old_mean * old_n + delta_total) / total_samples
end
function combine_quants(quants::MinibatchQuantities, total_delta, total_delta_sq, total_abs_delta, total_abs_delta_cubed, n)
    total_samples = quants.num_samples + n
    mean_delta = update_mean(quants.mean_delta, quants.num_samples, total_delta, total_samples)

    # Calculate new variance
    old_total_delta_sq = (quants.var_delta + quants.mean_delta*quants.mean_delta) * quants.num_samples
    var_delta = (old_total_delta_sq + total_delta_sq) / total_samples - mean_delta * mean_delta
    

    # TODO: This is only an estimate of the quantity using a running mean, and not the true mean
    mean_abs_delta = update_mean(quants.mean_abs_delta, quants.num_samples, total_abs_delta, total_samples)
    mean_abs_delta_cubed = update_mean(quants.mean_abs_delta_cubed, quants.num_samples, total_abs_delta_cubed, total_samples)

    estimated_error = error_estimate(mean_abs_delta_cubed, mean_abs_delta, total_samples)
    return MinibatchQuantities(mean_delta, var_delta, mean_abs_delta, mean_abs_delta_cubed, estimated_error, total_samples)
end
function calculate_batch_quantities!(quants::MinibatchQuantities, delta_loss_func, proposed_change, s, indices)::MinibatchQuantities
    select_samples!(delta_loss_func, indices)
    deltas = calculate_delta_losses!(delta_loss_func, proposed_change)

    
    total_delta, total_delta_sq, total_abs_delta, total_abs_delta_cubed = calculate_totals(deltas, s, quants.mean_delta, quants.num_samples)

    n = length(indices)
    new_quants = combine_quants(quants, total_delta, total_delta_sq, total_abs_delta, total_abs_delta_cubed, n)
    return new_quants
end

abstract type AbstractMinibatchAcceptanceCache end
Base.@kwdef mutable struct MinibatchAcceptanceCache{I<:AbstractArray{<:Integer}, T<:AbstractFloat} <: AbstractMinibatchAcceptanceCache
    const all_indices::I
    const total_samples::Int
    const batch_size::Int
    const error_tol::T
    const cutoff::T
    const offset::T
    const correction::CorrectionDistribution
    offset_index::Int = 0
end
get_all_indices(c::MinibatchAcceptanceCache) = c.all_indices
get_indices(c::MinibatchAcceptanceCache) = view(get_all_indices(c), c.offset_index+1:c.offset_index+c.batch_size)
get_total_samples(c::MinibatchAcceptanceCache) = c.total_samples
get_error_tol(c::MinibatchAcceptanceCache) = c.error_tol
get_batch_size(c::MinibatchAcceptanceCache) = c.batch_size
get_correction_dist(c::MinibatchAcceptanceCache) = c.correction
get_cutoff(c::MinibatchAcceptanceCache) = c.cutoff
get_offset(c::MinibatchAcceptanceCache) = c.offset

function should_cutoff(mean, variance, batch_size, cutoff, offset)
    z = zero(typeof(variance))
    return variance > z && batch_size * (max(abs(mean) - offset, z)/cutoff)^2 > variance
end

struct MinibatchAcceptanceCacheWithHistogram{C<:MinibatchAcceptanceCache, H<:AbstractHistogram} <: AbstractMinibatchAcceptanceCache
    cache::C
    histogram::H
end
has_histogram(::AbstractMinibatchAcceptanceCache) = false
has_histogram(::MinibatchAcceptanceCacheWithHistogram) = true
get_histogram(c::MinibatchAcceptanceCacheWithHistogram) = c.histogram
Lazy.@forward MinibatchAcceptanceCacheWithHistogram.cache get_all_indices, get_indices, increment_indices!, get_total_samples, get_error_tol, get_batch_size, get_correction_dist, get_cutoff, get_offset

function reset_indices!(cache::MinibatchAcceptanceCache)
    cache.offset_index = 0
    nothing
end
function shuffle_next_indices!(cache::MinibatchAcceptanceCache)
    num_samples = length(cache.all_indices)
    
    @assert cache.offset_index>=0
    @assert cache.offset_index+cache.batch_size <= num_samples

    @inbounds for i in cache.offset_index+1:cache.offset_index+cache.batch_size
        next_index = rand(i:num_samples)
        # Swap current index with sample from remainder of the list
        cache.all_indices[i], cache.all_indices[next_index] = (cache.all_indices[next_index], cache.all_indices[i]) 
    end
    nothing
end
function increment_indices!(cache::MinibatchAcceptanceCache)
    cache.offset_index = (cache.offset_index + cache.batch_size + 1) % cache.total_samples - 1
    # TODO: fix batch size wrapping
    nothing
end

function init!(proposed_change::SAProposedChange)
    T = typeof(proposed_change.old_observation)
    proposed_change.old_observation = zero(T)
    proposed_change.new_observation = zero(T)
end
function init!(proposed_change::TrajectoryProposedChange)
    z_t = zero(eltype(proposed_change.old_losses))
    for i in proposed_change.changed_indices
        proposed_change.old_losses[i] = z_t
        proposed_change.new_losses[i] = z_t
    end
end

function finalise!(proposed_change::SAProposedChange, quants::MinibatchQuantities)
    proposed_change.old_observation /= quants.num_samples
    proposed_change.new_observation /= quants.num_samples
    nothing
end
function finalise!(proposed_change::TrajectoryProposedChange, quants::MinibatchQuantities)
    for i in proposed_change.changed_indices
        proposed_change.old_losses[i] /= quants.num_samples
        proposed_change.new_losses[i] /= quants.num_samples
    end
end

struct MinibatchAcceptanceInfo
    quantities::MinibatchQuantities
    has_full_batch::Bool
    has_been_cutoff::Bool
    has_default_acceptance::Bool
    has_accepted::Bool
end

function minibatch_acceptance!(cache::AbstractMinibatchAcceptanceCache, proposed_change, loss_fn, bias)
    total_samples = get_total_samples(cache)
    s = bias
    error_tol = get_error_tol(cache)
    cutoff = get_cutoff(cache)
    offset = get_offset(cache)

    init!(proposed_change)
    reset_indices!(cache)
    shuffle_next_indices!(cache)

    quants = calculate_batch_quantities!(MinibatchQuantities(), loss_fn, proposed_change, s, get_indices(cache))
    has_been_cutoff = should_cutoff(quants.mean_delta, quants.var_delta, quants.num_samples, cutoff, offset)

    while ((quants.var_delta >= quants.num_samples && !has_been_cutoff) || quants.error_estimate > error_tol) && quants.num_samples < total_samples
        shuffle_next_indices!(cache)
        increment_indices!(cache)
        quants = calculate_batch_quantities!(quants, loss_fn, proposed_change, s, get_indices(cache))
        has_been_cutoff = should_cutoff(quants.mean_delta, quants.var_delta, quants.num_samples, cutoff, offset)
    end

    finalise!(proposed_change, quants)


    # Keep track of the minibatch sizes
    if has_histogram(cache)
        push!(get_histogram(cache), quants.num_samples)
    end

    # Normal acceptance test if entire dataset is used
    if quants.num_samples>=total_samples
        acceptance_rate = inv(1+exp(-quants.mean_delta))
        accept = rand(typeof(quants.mean_delta)) < acceptance_rate
        return MinibatchAcceptanceInfo(quants, true, false, false, accept)
    end

    
    # Short-circuit the cutoff if the acceptance will obviously be 1 or 0
    if has_been_cutoff
        return MinibatchAcceptanceInfo(quants, false, true, false, (quants.mean_delta > 0))
    end

    # Normal minibatch acceptance
    x_nc = randn() * sqrt(1-quants.var_delta/quants.num_samples)
    correction_dist = get_correction_dist(cache)
    x_corr = sample(correction_dist)

    return MinibatchAcceptanceInfo(quants, false, false, true, (quants.mean_delta + x_nc + x_corr > 0))
end
# TODO: Come up with a more sensible tolerance, perhaps based on s
function build_cache(total_samples, batch_size, correction::CorrectionDistribution; use_histogram=false, error_tol=0.5e-2, to_device=identity, cutoff=5.0, offset=10.0, index_type=Int)
    # todo: support device shuffling!
    all_indices = collect(convert(index_type, 1):convert(index_type, total_samples))
    # all_indices = collect(1:total_samples) |> to_device
    cache = MinibatchAcceptanceCache(all_indices, total_samples, batch_size, error_tol, cutoff, offset, correction, 0)

    if use_histogram
        histogram = FixedWidthHistogram(batch_size, total_samples, total_samples ÷ batch_size)
        return MinibatchAcceptanceCacheWithHistogram(cache, histogram)
    else
        return cache
    end
end

end