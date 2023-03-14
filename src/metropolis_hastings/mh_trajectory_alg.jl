using Random
using Base

Base.@kwdef mutable struct GaussianMHTrajectoryParameters{T<:Real, K<:Union{Nothing, Real}, X<:Union{Nothing, Real}, Q<:Union{Nothing,Int}}
    s::T
    σ::T
    fraction_to_include::K
    chance_shoot::X
    max_width::Q
end

Base.@kwdef mutable struct GaussianMHTrajectoryCache{T,Q<:AbstractArray{T},X,K<:AbstractArray{X},V,W<:AbstractObservable} <: AbstractMetropolisHastingsCache
    state_cache::Q
    exclude_parameter_mask::V
    total_observation::X
    last_observation::K
    cached_observation::K
    use_mask::Bool
    indices_changed::Union{StepRange, UnitRange}
    num_models::Int
    num_parameters::Int
    observable::W
    last_accepted::Bool = false
end

struct GaussianTrajectoryAlgorithm <: AbstractMetropolisHastingsAlg
    parameters::GaussianMHTrajectoryParameters
end

function TransitionPathSampling.generate_cache(alg::GaussianTrajectoryAlgorithm, problem::TPSProblem)
    initial_state = TransitionPathSampling.get_initial_state(problem)
    state_cache = [similar(s) for s in initial_state]
    num_models = length(state_cache)
    num_parameters = length(first(state_cache))
    num_params_to_change = !isnothing(alg.parameters.fraction_to_include) ? max(1, min(num_parameters, Int(round(alg.parameters.fraction_to_include * num_parameters)))) : num_parameters
    use_mask = (num_parameters != num_params_to_change)
    exclude_parameter_mask = BitArray(i > num_params_to_change for i in 1:num_parameters)
    observable = TransitionPathSampling.get_observable(problem)
    observations = TransitionPathSampling.observe(observable, initial_state)
    @assert typeof(observations)<:AbstractArray "The observation function does not return a vector of observations for each state in the trajectory."
    total_observation = sum(observations)
    indices_changed = 1:num_models

    return GaussianMHTrajectoryCache(
        state_cache=state_cache,
        exclude_parameter_mask=exclude_parameter_mask,
        total_observation=total_observation,
        last_observation=observations,
        cached_observation=deepcopy(observations),
        use_mask=use_mask,
        indices_changed=indices_changed,
        num_models = num_models,
        num_parameters=num_parameters,
        observable=observable
    )
end

function perturb!(cache::GaussianMHTrajectoryCache, alg::GaussianTrajectoryAlgorithm, states)
    fn! = isnothing(alg.parameters.chance_shoot) ? shoot! : shoot_or_bridge!
    fn!(cache, alg.parameters, states)
    nothing
end

function mask_cache!(cache::GaussianMHTrajectoryCache{T, Q}, states::Q) where {T, Q}
    if cache.use_mask
        for i in cache.indices_changed
            cache.state_cache[i][cache.exclude_parameter_mask] .= states[i][cache.exclude_parameter_mask]
        end
    end
    nothing
end

function shoot!(cache::GaussianMHTrajectoryCache{T, Q}, states::Q, static_index, σ, forwards) where {T, Q}
    indices = forwards ? (static_index+1:length(states)) : (static_index-1:-1:1)
    state_cache = cache.state_cache
    current_cache = states[static_index]
    for i in indices
        randn!(state_cache[i])
        state_cache[i] .= current_cache .+  state_cache[i] .* σ
        
        current_cache = state_cache[i]
    end
    cache.indices_changed = indices
    mask_cache!(cache, states)
    nothing
end

function bridge!(cache::GaussianMHTrajectoryCache{T, Q}, states::Q, start_index, end_index, σ) where {T, Q}
    indices = start_index+1:end_index-1
    n = length(indices) + 1
    state_cache = cache.state_cache
    current_state = states[start_index]
    final_state = states[end_index]
    for t in 1:(n-1)
        i = indices[t]
        sigma = σ*sqrt((n-t)/(n-t+1)) # Variance is time dependent
        randn!(state_cache[i])
        # Choose current state according to μ(x_(t-1), t) and v(x,t)
        state_cache[i] .= state_cache[i].*sigma .+ final_state ./ (n-t+1) .+ current_state .* ((n-t)/(n-t+1))
        
        current_state = state_cache[i]
    end
    cache.indices_changed = indices
    mask_cache!(cache, states)
    nothing
end
function _shoot_index_and_direction(T, ::Nothing)
    idx = rand(1:T)
    if idx == 1
        return (idx, true)
    elseif idx == T
        return (idx, false)
    else
        return (idx, rand() < 0.5)
    end
end
function _shoot_index_and_direction(T, max_width)
    forwards = rand() < 0.5
    w = min(T-1, max_width)
    idx = forwards ? T-rand(1:w) : rand(1:w)+1
    return idx, forwards
end

function shoot!(cache::GaussianMHTrajectoryCache{T, Q}, parameters::GaussianMHTrajectoryParameters, states::Q) where {T, Q}
    idx, forwards = _shoot_index_and_direction(cache.num_models, parameters.max_width)
    if cache.use_mask
        Random.shuffle!(cache.exclude_parameter_mask)
    end
    shoot!(cache, states, idx, parameters.σ, forwards)
end

function _sorted_rand_two_items(max_value)
    first_item = rand(1:max_value)
    second_item = rand(2:max_value)
    if second_item == first_item
        second_item = 1
    end
    return min(first_item, second_item), max(first_item, second_item)
end

function _bridge_indices(T, ::Nothing)
    T==3 && return 1,3 # Special case with no choice
    a,b = _sorted_rand_two_items(T)
    while b-a <= 1
        a,b = _sorted_rand_two_items(T)
    end
    return a,b
end
function _bridge_indices(T, max_size)
    T==3 && return 1,3 # Special case with no choice

    fixed_index = rand(2:T-1)    
    forwards = rand() < 0.5
    direction = forwards ? 1 : -1
    first_index = fixed_index - direction
    second_index = max(1, min(T, first_index + direction * (max_size + 1)))
    return min(first_index, second_index), max(first_index, second_index)
end

function bridge!(cache::GaussianMHTrajectoryCache{T, Q}, parameters::GaussianMHTrajectoryParameters, states::Q) where {T, Q}
    start_index, end_index = _bridge_indices(cache.num_models, parameters.max_width)
    if cache.use_mask
        Random.shuffle!(cache.exclude_parameter_mask)
    end
    bridge!(cache, states, start_index, end_index, parameters.σ)
end

function apply!(states::Q, cache::GaussianMHTrajectoryCache{T, Q}) where {T, Q}
    for i in cache.indices_changed
        copyto!(states[i], cache.state_cache[i])
    end
    nothing
end

function proposed_changed_state(cache::GaussianMHTrajectoryCache) 
    return view(cache.state_cache, cache.indices_changed)
end
function original_changed_state(cache::GaussianMHTrajectoryCache, states) 
    return view(states, cache.indices_changed)
end
get_last_observation!(cache::GaussianMHTrajectoryCache) = cache.total_observation
get_cached_observations(cache::GaussianMHTrajectoryCache) = cache.last_observation
function acceptance!(cache::GaussianMHTrajectoryCache{T, Q}, states::Q, alg::GaussianTrajectoryAlgorithm) where {T, Q}
    parameters = alg.parameters
    TransitionPathSampling.observe!(cache.cached_observation, cache.observable, cache.state_cache, cache.indices_changed)
    delta_obs = zero(eltype(cache.cached_observation))
    for i in cache.indices_changed
        delta_obs += cache.cached_observation[i] - cache.last_observation[i]
    end
    if rand() <= exp(-parameters.s * delta_obs)
        cache.last_observation[cache.indices_changed] .= cache.cached_observation[cache.indices_changed]
        # Apply the changes in the cache
        apply!(states, cache)
        cache.total_observation += delta_obs
        return true
    end

    return false
end

function shoot_or_bridge!(cache::GaussianMHTrajectoryCache{T, Q}, parameters::GaussianMHTrajectoryParameters, states::Q) where {T, Q}
    fn = rand() < parameters.chance_shoot ? shoot! : bridge!
    fn(cache, parameters, states)
end


function gaussian_trajectory_algorithm(s, σ; chance_to_shoot = nothing, params_changed_frac = nothing, max_width::Union{Nothing, Int} = nothing)
    @_check_fraction_domain chance_to_shoot "Chance to shoot"
    @_check_fraction_domain params_changed_frac "Fraction of changed parameters"

    parameters = GaussianMHTrajectoryParameters(s, σ, params_changed_frac, chance_to_shoot, max_width)
    return GaussianTrajectoryAlgorithm(parameters)
end