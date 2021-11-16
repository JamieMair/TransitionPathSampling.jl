module DiscreteTrajectory
using ..TPS
using Lazy: @forward

abstract type AbstractDiscreteTrajectoryProblem <: TPSProblem end
struct DTProblem{T, S<:AbstractObservable} <: AbstractDiscreteTrajectoryProblem
    observable::S
    states::Vector{T}
end

TPS.get_initial_state(problem::DTProblem) = problem.states
TPS.get_observable(problem::DTProblem) = problem.observable

get_trajectory_length(problem::DTProblem) = length(problem.states)
function TPS.init_solution(alg, problem::T, args...; kwargs...) where {T<:AbstractDiscreteTrajectoryProblem}
    sol = SimpleSolution(problem, alg)
    return sol
end

export DTProblem, get_trajectory_length, AbstractDiscreteTrajectoryProblem


end