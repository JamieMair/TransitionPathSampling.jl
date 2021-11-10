import Lazy: @forward
using Base
using ..TPS

abstract type AbstractSolution <: TPSSolution end
mutable struct SimpleSolution{T, S} <: TPSSolution
    observations::Vector{T}
    problem::TPSProblem
    algorithm::TPSAlgorithm
    state::S
end
function SimpleSolution(problem::T, algorithm::S) where {T<:TPSProblem, S<:TPSAlgorithm}
    observable = get_observable(problem)
    state = get_initial_state(problem)
    initial_observable = observe(observable, state)
    return SimpleSolution([initial_observable], problem, algorithm, deepcopy(state))
end
# Forward these methods so that the solution can be plotted
@forward SimpleSolution.observations (Base.size, Base.getindex, Base.setindex!, Base.IndexStyle, Base.iterate, Base.similar, Base.push!, Base.pop!, Base.first, Base.firstindex, Base.last, Base.lastindex, Base.length)
function get_current_state(solution::SimpleSolution)
    return solution.state
end
get_problem(solution::SimpleSolution) = solution.problem
function set_current_state!(solution::SimpleSolution{T, S}, state::S) where {T, S}
    solution.state = state
    return nothing
end

export AbstractSolution, get_current_state, set_current_state!, SimpleSolution