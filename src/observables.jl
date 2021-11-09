module Observables
# Interface for Observable
abstract type AbstractObservable end

struct SimpleObservable{T<:Function} <: AbstractObservable
    observe::T
end

function observe(observable::AbstractObservable, state) error("Not implemented") end
function observe(observable::SimpleObservable, state::T) where {T}
    return observable.observe(state)
end

export observe, AbstractObservable, SimpleObservable
end