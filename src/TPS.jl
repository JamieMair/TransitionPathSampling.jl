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
function init_solution(alg::TPSAlgorithm, problem::TPSProblem) end
function iterator(alg::TPSAlgorithm) end
function step!(solution, iter, alg::TPSAlgorithm) end

# Interface for TPSSolution
abstract type TPSSolution end

function finalise_solution!(solution::TPSSolution) nothing end
function construct_solution(alg::TPSAlgorithm, problem::TPSProblem) error("Not implemented.") end

function solve(problem::TPSProblem, alg::TPSAlgorithm, args...; kwargs...)
    solution = init_solution(alg, problem, args...; kwargs...)

    for iter in iterator(alg)
        step!(solution, alg, iter, args...; kwargs...)
    end

    finalise_solution!(solution)
    
    return solution
end

export solve, TPSProblem, TPSAlgorithm, TPSSolution
## Includes ##
include("default_solutions.jl")
include("observables.jl")


include("sa_problem.jl")
include("mh_alg.jl")
    
end