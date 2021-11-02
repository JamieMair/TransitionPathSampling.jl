abstract type AbstractSAProblem <: TPSProblem end
import Lazy: @forward
using Base

struct SAProblem{T, S<:AbstractObservable}
    observable::S
    state::T
end

function get_initial_state(problem::SAProblem)
    return problem.state
end
function get_observable(problem::SAProblem)
    return problem.observable
end

struct SimpleObservable{T<:Function} <: AbstractObservable
    observe::T
end

function observe(observable::AbstractObservable, state) error("Not implemented") end
function observe(observable::SimpleObservable, state::T) where {T}
    return observable.observe(state)
end

abstract type AbstractSolution <: TPSSolution end

mutable struct SimpleSolution{T, S}
    observations::Vector{T}
    problem::TPSProblem
    algorithm::TPSAlgorithm
    state::S
end
# Forward these methods so that the solution can be plotted
@forward SimpleSolution.observations (size, getindex, setindex!, IndexStyle, iterate, similar, push!, pop!)

abstract type AbstractMetropolisHastingsAlg <: TPSAlgorithm end

function init_solution(alg::AbstractMetropolisHastingsAlg, problem::TPSProblem, args...; kwargs...)
    observable = get_observable(problem)
    initial_state = get_initial_state(problem)
    initial_observable = observe(observable, initial_state)

    # By default, have a simple solution
    # TODO: Add some configuration options here
    # sol = SimpleSolution([initial_observable])

    return sol
end

export SAProblem, SimpleObservable, observe, SimpleSolution, init_solution
