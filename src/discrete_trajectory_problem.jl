module DiscreteTrajectory
using ..TPS
using Lazy: @forward

abstract type AbstractDiscreteTrajectoryProblem <: TPSProblem end
struct DTProblem{T, S<:AbstractObservable} <: AbstractDiscreteTrajectoryProblem
    observable::S
    states::Vector{T}
end

function TPS.get_initial_state(problem::DTProblem) 
    problem.states
end
function TPS.get_observable(problem::DTProblem)
    problem.observable
end
function get_trajectory_length(problem::DTProblem)
    length(problem.states)
end
function TPS.init_solution(alg, problem::T, args...; kwargs...) where {T<:AbstractDiscreteTrajectoryProblem}
    observable = TPS.get_observable(problem)
    state = TPS.get_initial_state(problem)
    initial_observable = sum(TPS.observe(observable, state))
    return SimpleSolution([initial_observable], problem, alg, deepcopy(state))
end


export DTProblem, get_trajectory_length, AbstractDiscreteTrajectoryProblem


end