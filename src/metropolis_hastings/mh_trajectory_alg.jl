using Random

function shoot_perturbation(states, static_index, σ, forwards::Bool; rng=Random.GLOBAL_RNG)
    T = length(states)
    indices = forwards ? (static_index+1:T) : (static_index-1:-1:1)
    n = length(indices)
    perturbations = [similar(states[i]) for i in 1:n]
    current_state = deepcopy(states[static_index])
    for i=2:n
        randn!(rng, perturbations[i])
        current_state .+= perturbations[i].*σ
        perturbations[i] .= current_state .- states[indices[i]]
    end

    return perturbations, indices
end

function get_guassian_shooting_perturbation_fn(σ; rng=Random.GLOBAL_RNG)
    return states -> begin
        T = length(states)
        @assert T >= 2 "You must have at least 2 states in your trajectory to perform shooting."

        # Choose where to shoot from
        static_index = rand(1:T)
        forwards = rand([true false])
        if static_index==1
            forwards = true
        elseif static_index==T
            forwards = false
        end

        return shoot_perturbation(states, static_index, σ, forwards; rng=rng)
    end
end

function get_apply_shooting_perturbation_fn()
    return (solution, perturbation) -> begin
        # Unpack the combined state of the perturbation
        perturbations, indices = perturbation
        state = get_current_state(solution)
        for (p, state_idx) in zip(perturbations, indices)
            state[state_idx] .+= p
        end
        set_current_state!(solution, state)
        nothing
    end
end

function get_undo_shooting_perturbation_fn()
    return (solution, perturbation) -> begin
        # Unpack the combined state of the perturbation
        perturbations, indices = perturbation
        state = get_current_state(solution)
        for (p, state_idx) in zip(perturbations, indices)
            state[state_idx] .-= p
        end
        set_current_state!(solution, state)
        nothing
    end
end

function get_shooting_observable_acceptance_fn(s, apply_fn, undo_fn; rng=Random.GLOBAL_RNG)
    return (solution, perturbation) -> begin
        obs = get_observable(get_problem(solution))
        previous_observation = last(solution)
        apply_fn(solution, perturbation)
        # TODO: Implement an observable which can calculate changes efficiently
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

function get_shooting_mh_alg(s, σ; rng=Random.GLOBAL_RNG)
    perturb_fn = get_guassian_shooting_perturbation_fn(σ; rng=rng)
    apply_fn = get_apply_shooting_perturbation_fn()
    undo_fn = get_undo_shooting_perturbation_fn()
    acpt_fn = get_shooting_observable_acceptance_fn(s, apply_fn, undo_fn; rng=rng)
    return MetropolisHastingsAlgorithm(perturb_fn, apply_fn, undo_fn, acpt_fn)
end