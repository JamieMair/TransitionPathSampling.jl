using Random

Base.@kwdef mutable struct GaussianSAParameters{T<:Real, K<:Union{Nothing, Real}}
    s::T
    σ::T
    fraction_to_include::K
end

Base.@kwdef mutable struct GaussianSACache{Q,K,V,W<:AbstractObservable} <: AbstractMetropolisHastingsCache
    state::Q
    last_observation::K
    exclude_parameter_mask::V
    use_mask::Bool
    num_parameters::Int
    observable::W
    last_accepted::Bool = false
end


struct GaussianSAAlgorithm <: AbstractMetropolisHastingsAlg
    parameters::GaussianSAParameters
end

function TPS.generate_cache(alg::GaussianSAAlgorithm, problem::TPSProblem)
    initial_state = TPS.get_initial_state(problem)
    state_cache = similar(initial_state)
    num_parameters = length(state_cache)
    num_params_to_change = !isnothing(alg.parameters.fraction_to_include) ? max(1, min(num_parameters, Int(round(alg.parameters.fraction_to_include * num_parameters)))) : num_parameters
    use_mask = (num_parameters != num_params_to_change)
    exclude_parameter_mask = BitArray(i > num_params_to_change for i in 1:num_parameters)
    observable = TPS.get_observable(problem)
    observation = TPS.observe(observable, initial_state)
    @assert !(typeof(observation)<:AbstractArray) "The observation function should return a scalar"
    
    return GaussianSACache(
        state=state_cache,
        last_observation=observation,
        exclude_parameter_mask=exclude_parameter_mask,
        use_mask=use_mask,
        num_parameters=num_parameters,
        observable=observable
    )
end

function TPS.step!(cache::GaussianSACache, solution::TPSSolution, alg::GaussianSAAlgorithm, iter, args...; kwargs...) 
    state = TPS.get_current_state(solution)
    perturb!(cache, state, alg.parameters.σ)
    accept = acceptance!(cache, state, alg.parameters)
    cache.last_accepted = accept
    if accept
        TPS.set_current_state!(solution, state)
    end
    push!(solution, cache.last_observation)
    # ToDo specialise on the type of solution to record more details
    nothing
end

function mask_cache!(cache::GaussianSACache{Q}, state::Q) where {Q}
    if cache.use_mask
        cache.state[cache.exclude_parameter_mask] .= state[cache.exclude_parameter_mask]
    end
    nothing
end

function perturb!(cache::GaussianSACache{Q}, state::Q, σ) where {Q}
    if cache.use_mask
        Random.shuffle!(cache.exclude_parameter_mask)
    end
    randn!(cache.state)
    cache.state .= state .+ cache.state .* σ
    mask_cache!(cache, state)
end

function apply!(state::Q, cache::GaussianSACache{Q}) where {Q}
    state .= cache.state
end

function acceptance!(cache::GaussianSACache{Q}, state::Q, parameters::GaussianSAParameters) where {Q}
    new_observation = TPS.observe(cache.observable, cache.state)
    delta_obs = new_observation - cache.last_observation
    if rand() <= exp(-parameters.s * delta_obs)
        cache.last_observation = new_observation
        # Apply the changes in the cache
        apply!(state, cache)
        return true
    end

    return false
end


function gaussian_sa_algorithm(s, σ; params_changed_frac=nothing)
    @_check_fraction_domain params_changed_frac "Fraction of changed parameters"
    
    parameters = GaussianSAParameters(s, σ, params_changed_frac)
    return GaussianSAAlgorithm(parameters)
end