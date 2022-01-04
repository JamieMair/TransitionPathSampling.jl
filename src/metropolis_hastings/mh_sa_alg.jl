using Random

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

function get_guassian_perturbation_fn(σ; rng=Random.GLOBAL_RNG, fraction_to_exclude=0.0)
    bit_array_memoized = Ref{Tuple{BitArray, Int, Int}}()
    return state -> begin
        delta = Base.similar(state)
        randn!(rng, delta)

        total_params = length(state)
        num_parameters = Int(floor(fraction_to_exclude*total_params))
        if (num_parameters > 0)
            if (!isdefined(bit_array_memoized, 1) || bit_array_memoized[][2] != num_parameters || bit_array_memoized[][3] != total_params)
                parameters_to_exclude = BitArray(i <= num_parameters for i = 1:total_params)
                bit_array_memoized[] = (parameters_to_exclude, num_parameters, total_params)
            end
            parameters_to_exclude = bit_array_memoized[][1]
            Random.shuffle!(rng, parameters_to_exclude)
            delta[parameters_to_exclude] .= zero(eltype(delta))
        end
        
        delta .*= σ
        return delta
    end
end
function get_apply_perturbation_fn()
    return (solution, perturbation) -> begin
        state = TPS.get_current_state(solution)
        state .+= perturbation
        TPS.set_current_state!(solution, state)
    end
end
function get_undo_perturbation_fn()
    return (solution, perturbation) -> begin
        state = TPS.get_current_state(solution)
        state .-= perturbation
        TPS.set_current_state!(solution, state)
    end
end
function get_observable_acceptance_fn(s, apply_fn, undo_fn; rng=Random.GLOBAL_RNG)
    return (solution, perturbation) -> begin
        obs = TPS.get_observable(TPS.get_problem(solution))
        previous_observation = last(solution)
        apply_fn(solution, perturbation)
        new_observation = TPS.observe(obs, TPS.get_current_state(solution))
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

function get_guassian_mh_alg(s, σ; rng=Random.GLOBAL_RNG, fraction_to_include=1.0)
    @assert (fraction_to_include >= 0.0 && fraction_to_include <= 1.0) "The fraction of parameters to include should be between 0 and 1 inclusive."
    fraction_to_exclude = 1.0-fraction_to_include
    perturb_fn = get_guassian_perturbation_fn(σ; rng=rng, fraction_to_exclude=fraction_to_exclude)
    apply_fn = get_apply_perturbation_fn()
    undo_fn = get_undo_perturbation_fn()
    acpt_fn = get_observable_acceptance_fn(s, apply_fn, undo_fn; rng=rng)
    return MetropolisHastingsAlgorithm(perturb_fn, apply_fn, undo_fn, acpt_fn)
end


function TPS.step!(solution::T, alg::MetropolisHastingsAlgorithm, args...; kwargs...) where {T<:TPSSolution}
    state = TPS.get_current_state(solution)
    delta = alg.perturb_gen_fn(state)
    alg.acceptance_fn!(solution, delta)
    nothing
end