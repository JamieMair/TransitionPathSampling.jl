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
function TransitionPathSampling.init_solution(algorithm, problem::T, iterator, args...; kwargs...) where {T<:AbstractDiscreteTrajectoryProblem}
    num_epochs = length(iterator)
    observable = TransitionPathSampling.get_observable(problem)
    state = TransitionPathSampling.get_initial_state(problem)
    initial_observable = sum([TransitionPathSampling.observe(observable, s) for s in state])
    observations = Vector{typeof(initial_observable)}(undef, num_epochs+1)
    observations[begin] = initial_observable
    return SimpleSolution(observations, problem, algorithm, deepcopy(state))
end


export DTProblem, get_trajectory_length, AbstractDiscreteTrajectoryProblem


end