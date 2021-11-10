module SimulatedAnnealing
using ..TPS


abstract type AbstractSAProblem <: TPSProblem end
struct SAProblem{T, S<:AbstractObservable} <: AbstractSAProblem
    observable::S
    state::T
end
function TPS.get_initial_state(problem::SAProblem)
    return problem.state
end
function TPS.get_observable(problem::SAProblem)
    return problem.observable
end
function init_solution(alg, problem::T, args...; kwargs...) where {T<:AbstractSAProblem}
    # By default, have a simple solution
    # TODO: Add some configuration options here
    sol = SimpleSolution(problem, alg)
    return sol
end

export SAProblem, init_solution
end