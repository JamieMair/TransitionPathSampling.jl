module Callbacks
using ..TPS
using Base
using SimpleTraits

"""
An AbstractCallback is a type which is used to define some custom functional behaviour.
"""
abstract type AbstractCallback end

# Assign traits to each dependency
@traitdef RequiresProblem{X}
@traitdef RequiresAlgorithm{X}
@traitdef RequiresSolution{X}
@traitdef RequiresCache{X}
@traitdef RequiresIterator{X}
@traitdef RequiresIteratorState{X}

function requires_quote(::T, D::Type{Q}) where {T, Q}
    if Q <: TPSProblem
        return quote
            @traitimpl RequiresProblem{$(esc(T))}
        end
    elseif Q <: TPSAlgorithm
        return quote
            @traitimpl RequiresAlgorithm{$(esc(T))}
        end
    elseif Q <: TPSSolution
        return quote
            @traitimpl RequiresAlgorithm{$(esc(T))}
        end
    end
end
function requires_quote(::T, D::Symbol) where {T}
    if D == :cache
        return quote
            @traitimpl RequiresCache{$(esc(T))}
        end
    elseif D == :iterator
        return quote
            @traitimpl RequiresIterator{$(esc(T))}
        end
    elseif D == :iterator_state
        return quote
            @traitimpl RequiresIteratorState{$(esc(T))}
        end
    end
end

# Assign traits to the location in which these are run
@traitdef RunsAtInitialisation{X}
@traitdef RunsAtFinalisation{X}
@traitdef RunsPreInnerLoop{X}
@traitdef RunsPrePostLoop{X}

# Create structs for the different timings
struct InitialisationCallback{T<:Function} <: AbstractCallback
    fn::T
end
struct FinalisationCallback{T<:Function} <: AbstractCallback
    fn::T
end
struct PreInnerLoopCallback{T<:Function} <: AbstractCallback
    fn::T
end
struct PostInnerLoopCallback{T<:Function} <: AbstractCallback
    fn::T
end
@traitimpl RunsAtInitialisation{InitialisationCallback}
@traitimpl RunsAtFinalisation{FinalisationCallback}
@traitimpl RunsPreInnerLoop{PreInnerLoopCallback}
@traitimpl RunsPrePostLoop{PostInnerLoopCallback}


macro register(cb, dependencies...)
    return quote
        begin
            for d in $dependencies
                Meta.eval(requires_quote($cb, d))
            end 
        end
    end
end

end