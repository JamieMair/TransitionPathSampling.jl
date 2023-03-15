using TransitionPathSampling
using TransitionPathSampling.MetropolisHastings
using TransitionPathSampling.SimulatedAnnealing
using TransitionPathSampling.DiscreteTrajectory
using TransitionPathSampling.Minibatch
using Test
using SafeTestsets
using Random
using LinearAlgebra
using Statistics

Random.seed!(1234)

struct TestBatchLossFn{TX, TY, I, D} <: AbstractBatchLossFn
    X::TX
    Y::TY
    current_indices::I
    cache::D
    deltas::D
end
function TransitionPathSampling.Minibatch.select_samples!(fn::TestBatchLossFn, indices)
    fn.current_indices .= indices
end
function TransitionPathSampling.Minibatch.calculate_losses!(fn::TestBatchLossFn, params)
    x_view = view(fn.X, :, fn.current_indices)
    y_view = view(fn.Y, fn.current_indices)
    mul!(reshape(fn.deltas, 1, :), reshape(params, 1, :), x_view)
    fn.deltas .= (fn.deltas .- y_view) .^ 2
    return fn.deltas
end
function TransitionPathSampling.Minibatch.calculate_delta_losses!(fn::TestBatchLossFn, proposed_change::SAProposedChange)
    params_old = proposed_change.old_state
    params_new = proposed_change.new_state
    x_view = view(fn.X, :, fn.current_indices)
    y_view = view(fn.Y, fn.current_indices)
    c = fn.cache
    deltas = fn.deltas

    mul!(reshape(c, 1, :), reshape(params_new, 1, :), x_view)
    deltas .= (c .- y_view) .^ 2
    proposed_change.new_observation += sum(deltas)
    mul!(reshape(c, 1, :), reshape(params_old, 1, :), x_view)
    c .= (c .- y_view) .^ 2
    proposed_change.old_observation += sum(c)
    deltas .-= c
    return deltas
end
function TransitionPathSampling.Minibatch.calculate_delta_losses!(fn::TestBatchLossFn, proposed_change::TrajectoryProposedChange)
    x_view = view(fn.X, :, fn.current_indices)
    y_view = view(fn.Y, fn.current_indices)
    c = fn.cache
    deltas = fn.deltas
    is_first = true

    for i in proposed_change.changed_indices
        pn = proposed_change.new_state[i]
        po = proposed_change.old_state[i]

        mul!(reshape(c, 1, :), reshape(pn, 1, :), x_view)
        c .= (c .- y_view) .^ 2
        proposed_change.new_losses[i] += sum(c)
        if is_first
            deltas .= c
            is_first = false

        else
            deltas .+= c
        end
        
        mul!(reshape(c, 1, :), reshape(po, 1, :), x_view)
        c .= (c .- y_view) .^ 2
        proposed_change.old_losses[i] += sum(c)
        deltas .-= c
    end

    return deltas
end

const N = 1024
const d = 3

const x_data = rand(d, N);
const y_data = reshape((rand(1, d) .* 2 .- 1) * x_data, :)
const batch_size = 16

function create_obs(X, Y, batch_size)
    c1 = similar(Y, eltype(Y), batch_size)
    c2 = similar(c1)
    loss_fn = TestBatchLossFn(X, Y, collect(1:batch_size), c1, c2)
    obs = BatchObservable(loss_fn, batch_size, length(Y), MeanLossSummary())
    return obs
end

function create_problem(X, Y, batch_size, tau)
    state = zeros(size(X, 1))
    obs = create_obs(X, Y, batch_size)
    if tau == 1
        return SAProblem(obs, state)
    else
        return DTProblem(obs, [deepcopy(state) for _ in 1:tau])
    end
end

function test_acceptance(X, Y, batch_size, tau, s, params_old, params_new, sigma, repeats)
    problem = create_problem(X, Y, batch_size, tau)
    inner_alg = if tau == 1
        TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, sigma);
    else
        TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, sigma);
    end
    alg = MinibatchMHAlg(inner_alg, s)
    config = Dict{Symbol, Any}(
        :use_histogram => true,
        :error_tol => 0.2
    )

    cache = TransitionPathSampling.generate_cache(alg, problem; minibatch_config=config)


    original_state = deepcopy(params_old)
    acceptances = Vector{Bool}(undef, repeats)
    for i in 1:repeats
        # Reset the cache
        if typeof(params_new)<:AbstractArray{<:AbstractArray}
            for (state, new_state) in zip(cache.inner_cache.state_cache, params_new)
                state .= new_state
            end
            for (state, old_state) in zip(original_state, params_old)
                state .= old_state
            end
        else
            cache.inner_cache.state .= params_new
            original_state .= params_old
        end
        accept = TransitionPathSampling.MetropolisHastings.acceptance!(cache, original_state, alg)

        acceptances[i] = accept
    end
    histogram = TransitionPathSampling.Minibatch.Acceptance.get_histogram(TransitionPathSampling.Minibatch.get_acceptance_cache(cache))
    return acceptances, histogram
end

function measure_losses(X, Y, tau, params...)
    problem = create_problem(X, Y, length(Y), tau)
    obs = TransitionPathSampling.get_observable(problem)
    return map(params) do p
        TransitionPathSampling.observe(obs, p)
    end
end

function get_params(d, tau, sigma)
    if tau > 1
        params_a = [randn(d) for _ in 1:tau]
        params_b = deepcopy(params_a)
        for p in params_b
            p .+= randn(size(p)).*sigma
        end
        return params_a, params_b
    else
        params_a = randn(d)
        params_b = randn(d) .* sigma .+ params_a
        return params_a, params_b
    end
end