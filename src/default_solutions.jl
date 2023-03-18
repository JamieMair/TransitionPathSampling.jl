import Lazy: @forward
using Base

mutable struct SimpleSolution{T, S} <: TPSSolution
    observations::Vector{T}
    problem::TPSProblem
    algorithm::TPSAlgorithm
    state::S
end
function SimpleSolution(problem::T, algorithm::S, num_epochs::Int) where {T<:TPSProblem, S<:TPSAlgorithm}
    observable = get_observable(problem)
    state = get_initial_state(problem)
    initial_observable = observe(observable, state)
    observations = Vector{typeof(initial_observable)}(undef, num_epochs+1)
    observations[begin] = initial_observable
    return SimpleSolution(observations, problem, algorithm, deepcopy(state))
end

get_current_state(solution::SimpleSolution) = solution.state
get_problem(solution::SimpleSolution) = solution.problem
function set_current_state!(solution::SimpleSolution{T, S}, state::S) where {T, S}
    solution.state = state
    return nothing
end
get_observable_type(::SimpleSolution{T, S}) where {T, S} = T
get_observations(solution::SimpleSolution) = solution.observations
function set_observation!(solution::SimpleSolution, iteration::Int, value)
    solution.observations[iteration] = value
end
export get_current_state, set_current_state!, SimpleSolution, get_problem, get_observable_type, get_observations