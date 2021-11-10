using Random

abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end

struct MetropolisHastingsAlgorithm <: AbstractMetropolisHastingsAlg
    perturb_gen_fn
    apply_perturbation_fn!
    undo_perturbation_fn!
    acceptance_fn!
end

function get_perturbation(algorithm::MetropolisHastingsAlgorithm, solution)
    return algorithm.perturb_gen_fn(solution)
end
function apply_perturbation(algorithm::MetropolisHastingsAlgorithm, solution, perturbation)
    return algorithm.apply_perturbation_fn(solution, perturbation)
end
function undo_perturbation(algorithm::MetropolisHastingsAlgorithm, solution, perturbation)
    return algorithm.undo_perturbation_fn(solution, perturbation)
end
function get_acceptance(algorithm::MetropolisHastingsAlgorithm, solution, perturbation)
    return algorithm.acceptance_fn(solution, perturbation)
end

function get_guassian_perturbation_fn(σ; rng=Random.GLOBAL_RNG)
    return state -> begin
        delta = Base.similar(state)
        randn!(rng, delta)
        delta .*= σ
        return delta
    end
end
function get_apply_perturbation_fn()
    return (solution, perturbation) -> begin
        state = get_current_state(solution)
        state .+= perturbation
        set_current_state!(solution, state)
    end
end
function get_undo_perturbation_fn()
    return (solution, perturbation) -> begin
        state = get_current_state(solution)
        state .-= perturbation
        set_current_state!(solution, state)
    end
end
function get_observable_acceptance_fn(s, apply_fn, undo_fn; rng=Random.GLOBAL_RNG)
    return (solution, perturbation) -> begin
        obs = get_observable(get_problem(solution))
        previous_observation = last(solution)
        apply_fn(solution, perturbation)
        new_observation = observe(obs, get_current_state(solution))
        if rand(rng) <= exp(-s*(new_observation-previous_observation))
            push!(solution, new_observation)
            return true
        else
            push!(solution, previous_observation)
            undo_fn(solution, perturbation)
            return false
        end
    end
end

function MetropolisHastingsAlgorithm(s, σ; rng=Random.GLOBAL_RNG)
    perturb_fn = get_guassian_perturbation_fn(σ; rng=rng)
    apply_fn = get_apply_perturbation_fn()
    undo_fn = get_undo_perturbation_fn()
    acpt_fn = get_observable_acceptance_fn(s, apply_fn, undo_fn; rng=rng)
    return MetropolisHastingsAlgorithm(perturb_fn, apply_fn, undo_fn, acpt_fn)
end


function step!(solution::T, alg::MetropolisHastingsAlgorithm, args...; kwargs...) where {T<:TPSSolution}
    state = get_current_state(solution)
    delta = alg.perturb_gen_fn(state)
    alg.acceptance_fn!(solution, delta)
    nothing
end

export step!, MetropolisHastingsAlgorithm