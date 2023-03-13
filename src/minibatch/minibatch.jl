module Minibatch
import Lazy: @forward
using ..TransitionPathSampling
import ..TransitionPathSampling.MetropolisHastings: AbstractMetropolisHastingsAlg

include("corrections.jl")
include("acceptance.jl")

using Acceptance
using Corrections

struct MinibatchMHAlg{A<:AbstractMetropolisHastingsAlg, B<:Acceptance.AbstractMinibatchAcceptanceCache}
    inner_algorithm::A
    acceptance_cache::B
    correction_dist::CorrectionDistribution
end

# TODO:
# 1. Add a "batch" observable, containing the dataset, model etc
# 2. "batch" observable should produce an array of "deltas"
# 3. Create a new cache which includes the cache of the inner algorithm, along with the acceptance and correction distribution caches
# 4. Write some tests, which will run the algorithm, and check for statistical differences with a mock observable


end