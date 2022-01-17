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

function create_automatic_buffer(::T, max_values::Int) where {T}
    return AutomaticConvergenceBuffer(zeros(T, max_values), max_values, 0, 0, zero(T), zero(T))
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
    return ensemble.squared_sum / ensemble.current_timespan  - mean(ensemble)^2
end

function Statistics.std(ensemble::AutomaticConvergenceBuffer)
    return sqrt(var(ensemble))
end

struct ConvergenceSolutionWrapper{T<:TPSSolution}
    solution::T
    buffer::AutomaticConvergenceBuffer
end

mutable struct AutomaticConvergenceOptions{T}
    warmup_steps::Int
    polling_frequency::Int
    max_buffer_size::Int
    max_epochs::Int
    relative_gradient_size::T
    relative_error_size::T
end
get_buffer_size(iter::AutomaticConvergenceOptions) = iter.max_buffer_size
get_warmup_buffer_size(iter::AutomaticConvergenceOptions) = iter.warmup_steps
get_polling_frequency(iter::AutomaticConvergenceOptions) = iter.polling_frequency

mutable struct AutomaticConvergenceIterator{T<:Number}
    solution::ConvergenceSolutionWrapper
    options::AutomaticConvergenceOptions
    current_observable::T
end

function Base.iterate(iter::AutomaticConvergenceIterator)::Int
    push!(iter.solution.buffer, last(iter.solution.solution))
    iter.current_observable = Statistics.mean(iter.solution.buffer)
    return 1
end

function Base.iterate(iter::AutomaticConvergenceIterator, state::Int)
    # Update the buffer
    push!(iter.solution.buffer, last(iter.solution.solution))
    # For warm ups, just iterate to the next state
    if state <= get_buffer_size(iter.options)
        return state + 1
    end
    polling_freq = get_polling_frequency(iter.options)
    if state % polling_freq == 0
        next_observable = Statisics.mean(iter.solution.buffer)


        gradient_estimate = (next_observable - iter.current_observable) / polling_freq
        normalised_gradient_estimate = gradient_estimate / next_observable

        mean_error = Statistics.std(iter.solution.buffer) / sqrt(iter.solution.buffer.current_timespan)
        relative_mean_error = next_observable / mean_error
        
        iter.current_observable = next_observable
        if abs(normalised_gradient_estimate) < iter.options.relative_gradient_size && abs(relative_mean_error) < iter.options.relative_error_size
            return nothing # Signals the end of the iteration
        else
            return state + 1
        end
    end    
end

function TPS.get_iterator(options::AutomaticConvergenceOptions, args...; solution, kwargs...)
    observable_type = get_observable_type(solution)
    buffer = create_automatic_buffer(observable_type, get_buffer_size(options))
    sol_wrapper = ConvergenceSolutionWrapper(solution, buffer)
    iterator = AutomaticConvergenceIterator(sol_wrapper, options, zero(observable_type))
    return iterator
end