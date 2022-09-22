# Interface for Observable
abstract type AbstractObservable end

struct SimpleObservable{T<:Function} <: AbstractObservable
    observe::T
end

function observe(observable::AbstractObservable, state) error("Not implemented") end
function observe(observable::SimpleObservable, state)
    return observable.observe(state)
end
function observe!(cache, observable::SimpleObservable, state::AbstractArray, indices)
    for i in indices
        cache[i] = observable.observe(state[i])
    end
    nothing
end


export observe, observe!, AbstractObservable