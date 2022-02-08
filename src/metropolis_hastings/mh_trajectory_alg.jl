using Random

function shoot_perturbation(states, static_index, σ, forwards::Bool; rng=Random.GLOBAL_RNG, parameter_exclude_mask=Nothing)
    T = length(states)
    indices = forwards ? (static_index+1:T) : (static_index-1:-1:1)
    n = length(indices)
    perturbations = [similar(states[i]) for i in 1:n]
    current_state = deepcopy(states[static_index])
    for i=1:n
        randn!(rng, perturbations[i])
        current_state .+= perturbations[i].*σ
        perturbations[i] .= current_state .- states[indices[i]]

        if (parameter_exclude_mask != Nothing)
            perturbations[i][parameter_exclude_mask] .= zero(eltype(perturbations[i]))
        end
    end

    return perturbations, indices
end

function bridge_perturbation(states, start_index, end_index, σ; rng=Random.GLOBAL_RNG, parameter_exclude_mask=Nothing)
    indices = start_index+1:end_index-1
    n = length(indices)
    @assert n>=1
    perturbations = [similar(states[i]) for i in 1:n]
    current_state = deepcopy(states[start_index])
    final_state = states[end_index]
    for i=1:n
        sigma = σ*sqrt((n-i)/(n-i+1)) # Variance is time dependent
        randn!(rng, perturbations[i]) # Normal distribution
        # Choose current state according to μ(x_(t-1), t) and v(x,t)
        current_state .= perturbations[i].*sigma .+ final_state ./ (n-i+1) .+ current_state .* ((n-i)/(n-i+1))

        # Calculate the perturbation needed so the change can be reversed
        perturbations[i] .= current_state .- states[indices[i]]

        if (parameter_exclude_mask != Nothing)
            perturbations[i][parameter_exclude_mask] .= zero(eltype(perturbations[i]))
        end
    end

    return perturbations, indices
end

function get_guassian_shooting_perturbation_fn(σ; rng=Random.GLOBAL_RNG, fraction_to_exclude=0.0, max_width=nothing)
    # This function is NOT thread-safe! - might be worth removing the cache here.

    function generate_shoot_index_and_direction(T, ::Nothing)
        # Choose where to shoot from
        static_index = rand(rng, 1:T)
        forwards = rand(rng, [true false])
        if static_index==1
            forwards = true
        elseif static_index==T
            forwards = false
        end
        return static_index, forwards
    end
    function generate_shoot_index_and_direction(T, max_width)
        forwards = rand(rng, [true false])
        w = min(T-1, max_width)
        static_index = forwards ? T-rand(rng, 1:w) : rand(rng, 1:w)+1
        return static_index, forwards
    end

    bit_array_memoized = Ref{Tuple{BitArray, Int, Int}}()
    return states -> begin
        T = length(states)
        @assert T >= 2 "You must have at least 2 states in your trajectory to perform shooting."

        # Choose where to shoot from
        static_index, forwards = generate_shoot_index_and_direction(T, max_width)

        if (fraction_to_exclude==0.0)
            parameters_to_exclude = Nothing
        else
            total_params = length(states[begin])
            num_parameters = Int(floor(fraction_to_exclude*total_params))
            if (!isdefined(bit_array_memoized, 1) || bit_array_memoized[][2] != num_parameters || bit_array_memoized[][3] != total_params)
                parameters_to_exclude = BitArray(i <= num_parameters for i = 1:total_params)
                bit_array_memoized[] = (parameters_to_exclude, num_parameters, total_params)
            end

            parameters_to_exclude = bit_array_memoized[][1]
            Random.shuffle!(rng, parameters_to_exclude)
        end

        return shoot_perturbation(states, static_index, σ, forwards; rng=rng, parameter_exclude_mask=parameters_to_exclude)
    end
end

function get_bridging_indices(rng, T, ::Nothing)
    T==3 && return 1,3
    
    a,b = sort(randperm(rng, T)[1:2])
    while abs(a-b) <= 1
        a,b = sort(randperm(rng, T)[1:2])
    end
    return a,b
end
function get_bridging_indices(rng, T, max_size)
    T==3 && return 1,3 # Special case with no choice

    fixed_index = rand(rng, 1:T)    
    
    if fixed_index < 3
        forwards = true
    elseif fixed_index > T-2
        forwards = false
    else
        forwards = rand(rng, (true, false))
    end
    direction = forwards ? 1 : -1
    second_index = max(1, min(T, fixed_index + direction*(max_size+1)))
    return min(fixed_index, second_index), max(fixed_index, second_index)
end


function get_guassian_bridging_perturbation_fn(σ; rng=Random.GLOBAL_RNG, fraction_to_exclude=0.0, max_width=nothing)
    # This function is NOT thread-safe! - might be worth removing the cache here.
    bit_array_memoized = Ref{Tuple{BitArray, Int, Int}}()
    return states -> begin
        T = length(states)
        @assert T >= 3 "You must have at least 3 states in your trajectory to perform bridging."

        # Choose where to shoot from
        start_index, end_index = get_bridging_indices(rng, T, max_width)

        if (fraction_to_exclude==0.0)
            parameters_to_exclude = Nothing
        else
            total_params = length(states[begin])
            num_parameters = Int(floor(fraction_to_exclude*total_params))
            if (!isdefined(bit_array_memoized, 1) || bit_array_memoized[][2] != num_parameters || bit_array_memoized[][3] != total_params)
                parameters_to_exclude = BitArray(i <= num_parameters for i = 1:total_params)
                bit_array_memoized[] = (parameters_to_exclude, num_parameters, total_params)
            end

            parameters_to_exclude = bit_array_memoized[][1]
            Random.shuffle!(rng, parameters_to_exclude)
        end
        
        return bridge_perturbation(states, start_index, end_index, σ; rng=rng, parameter_exclude_mask=parameters_to_exclude)
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
        obs = TPS.get_observable(TPS.get_problem(solution))
        previous_observation = last(solution)
        apply_fn(solution, perturbation)
        # TODO: Implement an observable which can calculate changes efficiently
        new_observation = TPS.observe(obs, TPS.get_current_state(solution))
        # TODO: Abstract this method to calculate a chance of acceptance - the responsibilities should be separated
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

function get_combined_shooting_and_bridging_perturbation_fn(args...; rng=Random.GLOBAL_RNG, kwargs...)
    shoot_fn = get_guassian_shooting_perturbation_fn(args...; rng, kwargs...)
    bridging_fn = get_guassian_bridging_perturbation_fn(args...; rng, kwargs...)
    return states -> begin
        fn = rand(rng, (shoot_fn, bridging_fn))
        return fn(states)
    end
end


function get_shooting_mh_alg(s, σ; rng=Random.GLOBAL_RNG, fraction_to_include=1.0)
    @assert (fraction_to_include >= 0.0 && fraction_to_include <= 1.0) "The fraction of parameters to include should be between 0 and 1 inclusive."
    fraction_to_exclude = 1.0-fraction_to_include

    perturb_fn = get_guassian_shooting_perturbation_fn(σ; rng=rng, fraction_to_exclude = fraction_to_exclude)
    apply_fn = get_apply_shooting_perturbation_fn()
    undo_fn = get_undo_shooting_perturbation_fn()
    acpt_fn = get_shooting_observable_acceptance_fn(s, apply_fn, undo_fn; rng=rng)
    return MetropolisHastingsAlgorithm(perturb_fn, apply_fn, undo_fn, acpt_fn)
end

function get_shooting_and_bridging_mh_alg(s, σ; rng=Random.GLOBAL_RNG, fraction_to_include=1.0, max_width=nothing)
    @assert (fraction_to_include >= 0.0 && fraction_to_include <= 1.0) "The fraction of parameters to include should be between 0 and 1 inclusive."
    fraction_to_exclude = 1.0-fraction_to_include

    perturb_fn = get_combined_shooting_and_bridging_perturbation_fn(σ; rng=rng, fraction_to_exclude=fraction_to_exclude, max_width=max_width)
    apply_fn = get_apply_shooting_perturbation_fn()
    undo_fn = get_undo_shooting_perturbation_fn()
    acpt_fn = get_shooting_observable_acceptance_fn(s, apply_fn, undo_fn; rng=rng)
    return MetropolisHastingsAlgorithm(perturb_fn, apply_fn, undo_fn, acpt_fn)
end