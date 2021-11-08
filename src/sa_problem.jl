abstract type AbstractSAProblem <: TPSProblem end
import Lazy: @forward
using Base

struct SAProblem{T, S<:AbstractObservable} <: TPSProblem
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

mutable struct SimpleSolution{T, S} <: TPSSolution
    observations::Vector{T}
    problem::TPSProblem
    algorithm::TPSAlgorithm
    state::S
end
function SimpleSolution(problem::TPSProblem, algorithm::TPSAlgorithm)
    initial_observable = observe(observable, initial_state)
    return SimpleSolution([initial_observable], problem, algorithm, deepcopy(get_initial_state(problem)))
end
# Forward these methods so that the solution can be plotted
@forward SimpleSolution.observations (size, getindex, setindex!, IndexStyle, iterate, similar, push!, pop!)
function get_current_state(solution::SimpleSolution)
    return solution.state
end
get_problem(solution::SimpleSolution) = solution.problem
function set_current_state!(solution::SimpleSolution{T, S}, state::S) where {T, S}
    solution.state = state
    return nothing
end

function init_solution(alg::TPSAlgorithm, problem::TPSProblem, args...; kwargs...)
    # By default, have a simple solution
    # TODO: Add some configuration options here
    sol = SimpleSolution(problem, alg)
    return sol
end

export SAProblem, SimpleObservable, observe, SimpleSolution, init_solution
