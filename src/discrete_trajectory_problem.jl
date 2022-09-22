module DiscreteTrajectory
using ..TransitionPathSampling
using Lazy: @forward

abstract type AbstractDiscreteTrajectoryProblem <: TPSProblem end
struct DTProblem{T, S<:AbstractObservable} <: AbstractDiscreteTrajectoryProblem
    observable::S
    states::Vector{T}
end

function TransitionPathSampling.get_initial_state(problem::DTProblem) 
    problem.states
end
function TransitionPathSampling.get_observable(problem::DTProblem)
    problem.observable
end
function get_trajectory_length(problem::DTProblem)
    length(problem.states)
end
function TransitionPathSampling.init_solution(alg, problem::T, args...; kwargs...) where {T<:AbstractDiscreteTrajectoryProblem}
    observable = TransitionPathSampling.get_observable(problem)
    state = TransitionPathSampling.get_initial_state(problem)
    initial_observable = sum(TransitionPathSampling.observe(observable, state))
    return SimpleSolution([initial_observable], problem, alg, deepcopy(state))
end


export DTProblem, get_trajectory_length, AbstractDiscreteTrajectoryProblem


end