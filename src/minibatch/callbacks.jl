using ..TransitionPathSampling.Callbacks


# TODO: ADD OPTIONS TO DISABLE THE STATS COLLECTION

struct MinibatchStatisticsCallback <: TransitionPathSampling.Callbacks.AbstractCallback
    acceptances::BitVector
    is_full_batch::BitVector
    is_cutoff_batch::BitVector
    is_default_batch::BitVector
    batch_sizes::Vector{Int}
    mean_deltas::Vector{Float64}
    var_deltas::Vector{Float64}
end

function MinibatchStatisticsCallback(num_epochs::Int)
    acceptances = BitVector(undef, num_epochs)
    is_full_batch = BitVector(undef, num_epochs)
    is_cutoff_batch = BitVector(undef, num_epochs)
    is_default_batch = BitVector(undef, num_epochs)
    batch_sizes = Vector{Int}(undef, num_epochs)
    mean_deltas = Vector{Float64}(undef, num_epochs)
    var_deltas = Vector{Float64}(undef, num_epochs)
    return MinibatchStatisticsCallback(acceptances, is_full_batch, is_cutoff_batch, is_default_batch, batch_sizes, mean_deltas, var_deltas)
end

function _run!(::MinibatchStatisticsCallback, ::Any, ::Any)
    @error "MinibatchStatisticsCallback requires a BatchMHCache to be used and an integer iteration state."
end
function _run!(cb::MinibatchStatisticsCallback, cache::BatchMHCache, epoch::Int)
    cb.acceptances[epoch] = cache.last_acceptance_info.has_accepted
    cb.is_full_batch[epoch] = cache.last_acceptance_info.has_full_batch
    cb.is_cutoff_batch[epoch] = cache.last_acceptance_info.has_been_cutoff
    cb.is_default_batch[epoch] = cache.last_acceptance_info.has_default_acceptance
    cb.batch_sizes[epoch] = cache.last_acceptance_info.quantities.num_samples
    cb.mean_deltas[epoch] = cache.last_acceptance_info.quantities.mean_delta
    cb.var_deltas[epoch] = cache.last_acceptance_info.quantities.var_delta
    nothing
end

function TransitionPathSampling.Callbacks.run(cb::MinibatchStatisticsCallback, deps::TransitionPathSampling.Callbacks.SolveDependencies)
    _run!(cb, deps.cache, deps.iterator_state)
end
