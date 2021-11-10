module SimulatedAnnealing
using ..TPS


abstract type AbstractSAProblem <: TPSProblem end
struct SAProblem{T, S<:AbstractObservable} <: AbstractSAProblem
    observable::S
    state::T
end
function get_initial_state(problem::SAProblem)
    return problem.state
end
function get_observable(problem::SAProblem)
    return problem.observable
end
function init_solution(alg::TPSAlgorithm, problem::AbstractSAProblem, args...; kwargs...)
    # By default, have a simple solution
    # TODO: Add some configuration options here
    sol = SimpleSolution(problem, alg)
    return sol
end

export SAProblem, init_solution, get_observable, get_initial_state
end