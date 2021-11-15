module TPS



# Interface for TPSProblem
abstract type TPSProblem end
"""
    get_initial_state(problem)

Retrieves an initial state for the problem specified.

# Examples
```julia-repl
julia> initial_state = get_initial_state(problem)
```
"""
function get_initial_state(problem::TPSProblem) end
function get_observable(problem::TPSProblem) end

# Interface for TPSAlgorithm
abstract type TPSAlgorithm end
function init_solution(alg, problem, args...; kwargs...) error("Default solution is not specified.") end
function step!(solution, iter, alg::TPSAlgorithm) end

# Interface for TPSSolution
abstract type TPSSolution end

function finalise_solution!(solution::TPSSolution) nothing end

include("iterators.jl")

function solve(problem::TPSProblem, alg::TPSAlgorithm, iterator, args...; kwargs...)
    solution = init_solution(alg, problem, args...; kwargs...)
    iter = get_iterator(iterator; problem = problem, solution=solution, algorithm=alg)
    for iter_state in iter
        step!(solution, alg, iter_state, args...; kwargs...)
    end

    finalise_solution!(solution)
    
    return solution
end

export solve, TPSProblem, TPSAlgorithm, TPSSolution
## Includes ##
include("observables.jl")
include("default_solutions.jl")


include("sa_problem.jl")
include("mh_alg.jl")
    
end