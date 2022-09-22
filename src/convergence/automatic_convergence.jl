using Statistics
import Lazy: @forward

mutable struct AutomaticConvergenceBuffer{T}
    measurements::Array{T}
    max_values::Int
    current_timespan::Int
    current_time::Int
    sum::T
    squared_sum::T
end

function create_automatic_buffer(observation_type, max_values::Int)
    return AutomaticConvergenceBuffer(zeros(observation_type, max_values), max_values, 0, 0, zero(observation_type), zero(observation_type))
end

function Base.push!(ensemble::AutomaticConvergenceBuffer{T}, measurement::T) where {T}
    time = ensemble.current_time
    index = time % ensemble.max_values + 1
    old_value = ensemble.measurements[index]

    # Set the number of current samples in the array
    ensemble.current_timespan = min(time + 1, ensemble.max_values)

    # Update the running sum and squared sum
    ensemble.sum += measurement - old_value
    ensemble.squared_sum += measurement*measurement - old_value*old_value

    # Replace the cache with the new max_values
    ensemble.measurements[index] = measurement

    # Update the state to the next time
    ensemble.current_time += 1
    return nothing
end

function Statistics.mean(ensemble::AutomaticConvergenceBuffer)
    return ensemble.sum / ensemble.current_timespan
end

function Statistics.var(ensemble::AutomaticConvergenceBuffer)
    return ensemble.squared_sum / (ensemble.current_timespan-1)  - ensemble.current_timespan/(ensemble.current_timespan-1) * mean(ensemble)^2
end

function Statistics.std(ensemble::AutomaticConvergenceBuffer)
    return sqrt(max(Statistics.var(ensemble), 0))
end

struct ConvergenceSolutionWrapper{T<:TPSSolution}
    solution::T
    buffer::AutomaticConvergenceBuffer
end

mutable struct AutomaticConvergenceOptions
    warmup_steps::Int
    polling_frequency::Int
    max_buffer_size::Int
    max_epochs::Int
    relative_gradient_size::Float64
    relative_error_size::Float64
end
get_buffer_size(options::AutomaticConvergenceOptions) = options.max_buffer_size
get_warmup_buffer_size(options::AutomaticConvergenceOptions) = options.warmup_steps
get_polling_frequency(options::AutomaticConvergenceOptions) = options.polling_frequency
get_max_iterations(options::AutomaticConvergenceOptions) = options.max_epochs
mutable struct AutomaticConvergenceIterator
    solution::ConvergenceSolutionWrapper
    options::AutomaticConvergenceOptions
end

struct AutomaticConvergenceIteratorState{T}
    epoch::Integer
    mean_observable::T
end

function Base.iterate(iter::AutomaticConvergenceIterator)
    push!(iter.solution.buffer, last(iter.solution.solution))
    return (1, AutomaticConvergenceIteratorState(1, Statistics.mean(iter.solution.buffer)))
end

function Base.iterate(iter::AutomaticConvergenceIterator, state::AutomaticConvergenceIteratorState)
    # Update the buffer
    i = state.epoch
    current_observable = state.mean_observable
    push!(iter.solution.buffer, last(iter.solution.solution))
    # For warm ups, just iterate to the next state
    if i <= get_warmup_buffer_size(iter.options)
        return (i+1, AutomaticConvergenceIteratorState(i + 1, current_observable))
    elseif i > get_max_iterations(iter.options)
        return nothing
    end
    polling_freq = get_polling_frequency(iter.options)
    if i % polling_freq == 0
        next_observable = Statistics.mean(iter.solution.buffer)


        gradient_estimate = (next_observable - current_observable) / polling_freq
        normalised_gradient_estimate = gradient_estimate / next_observable

        mean_error = Statistics.std(iter.solution.buffer) / sqrt(iter.solution.buffer.current_timespan)
        relative_mean_error = mean_error / next_observable
        
        current_observable = next_observable
        if abs(normalised_gradient_estimate) < iter.options.relative_gradient_size && abs(relative_mean_error) < iter.options.relative_error_size
            return nothing # Signals the end of the iteration
        end
    end

    return (i+1, AutomaticConvergenceIteratorState(i + 1, current_observable))
end

function TransitionPathSampling.get_iterator(options::AutomaticConvergenceOptions, args...; solution, kwargs...)
    observable_type = get_observable_type(solution)
    buffer = create_automatic_buffer(observable_type, get_buffer_size(options))
    sol_wrapper = ConvergenceSolutionWrapper(solution, buffer)
    iterator = AutomaticConvergenceIterator(sol_wrapper, options)
    return iterator
end

TransitionPathSampling.get_epoch_from_state(state::AutomaticConvergenceIteratorState) = state.epoch

export AutomaticConvergenceOptions