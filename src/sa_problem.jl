module SimulatedAnnealing
using ..TransitionPathSampling


abstract type AbstractSAProblem <: TPSProblem end
struct SAProblem{T, S<:AbstractObservable} <: AbstractSAProblem
    observable::S
    state::T
end
function TransitionPathSampling.get_initial_state(problem::SAProblem)
    return problem.state
end
function TransitionPathSampling.get_observable(problem::SAProblem)
    return problem.observable
end
function TransitionPathSampling.init_solution(alg, problem::T, iterator, args...; kwargs...) where {T<:AbstractSAProblem}
    # By default, have a simple solution
    # TODO: Add some configuration options here
    sol = SimpleSolution(problem, alg, length(iterator))
    return sol
end

export SAProblem, AbstractSAProblem
end