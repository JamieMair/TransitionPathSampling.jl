module Annealing
using ..TPS.MetropolisHastings
using ..TPS
import Lazy:@forward


abstract type AbstractAnnealedAlgorithm <: TPSAlgorithm end
get_main_algorithm(alg::AbstractAnnealedAlgorithm) = throw("Unimplemented exception.")
abstract type SingleParameterAnnealedAlgorithm <: AbstractAnnealedAlgorithm end
get_symbol(alg::SingleParameterAnnealedAlgorithm) = throw("Unimplemented exception.")
calculate_parameter_value(alg::SingleParameterAnnealedAlgorithm, epoch) = throw("Unimplemented exception.")
get_parameters_object(alg) = getfield(alg, :parameters)
get_parameters_object(alg::MetropolisHastingsAlgorithm) = alg.parameters
function set_parameter!(algorithm::SingleParameterAnnealedAlgorithm, parameter_value)
    main_algorithm = get_main_algorithm(algorithm)
    parameters = get_parameters_object(main_algorithm)
    parameter_symbol = get_symbol(algorithm)
    setfield!(parameters, parameter_symbol, parameter_value)
    nothing
end

struct ExponentialDecayAnnealedAlgorithm{T} <: SingleParameterAnnealedAlgorithm where {T<:TPSAlgorithm}
    algorithm::T
    parameter_starting_value::Float64
    decay_constant::Float64
    epoch_width::Integer
    parameter_symbol::Symbol
end
struct ClippedExponentialDecayAnnealedAlgorithm{T} <: SingleParameterAnnealedAlgorithm where {T<:TPSAlgorithm}
    algorithm::ExponentialDecayAnnealedAlgorithm{T}
    min_parameter_value
    max_parameter_value
end
get_main_algorithm(alg::ExponentialDecayAnnealedAlgorithm) = alg.algorithm
get_symbol(alg::ExponentialDecayAnnealedAlgorithm) = alg.parameter_symbol
@forward ClippedExponentialDecayAnnealedAlgorithm.algorithm get_main_algorithm, get_symbol

function calculate_parameter_value(alg::ExponentialDecayAnnealedAlgorithm, epoch)
    param = alg.parameter_starting_value * alg.decay_constant ^ (epoch/alg.epoch_width)
    return param
end
function calculate_parameter_value(alg::ClippedExponentialDecayAnnealedAlgorithm, epoch)
    exp_alg = alg.algorithm
    param = calculate_parameter_value(exp_alg, epoch)
    if !isnothing(alg.min_parameter_value)
        param = max(alg.min_parameter_value, param)
    end
    if !isnothing(alg.max_parameter_value)
        param = min(alg.max_parameter_value, param)
    end
    return param
end

function TPS.step!(solution::T, alg::SingleParameterAnnealedAlgorithm, iter_state, args...; kwargs...) where {T<:TPSSolution}
    epoch = TPS.get_epoch_from_state(iter_state)
    next_parameter_value = calculate_parameter_value(alg, epoch)
    set_parameter!(alg, next_parameter_value)

    # Perform the original algorithm
    TPS.step!(solution, get_main_algorithm(alg), iter_state, args...; kwargs...)
    nothing
end


function create_exponential_decay_algorithm(algorithm, starting_parameter_value, decay_constant, epoch_width, parameter_symbol; max_parameter_value=nothing, min_parameter_value=nothing)
    alg = ExponentialDecayAnnealedAlgorithm(algorithm, starting_parameter_value, decay_constant, epoch_width, parameter_symbol)
    if !isnothing(max_parameter_value) || !isnothing(min_parameter_value)
        return ClippedExponentialDecayAnnealedAlgorithm(alg, min_parameter_value, max_parameter_value)
    else
        return alg
    end
end


export ExponentialDecayAnnealedAlgorithm, ClippedExponentialDecayAnnealedAlgorithm, create_exponential_decay_algorithm
end