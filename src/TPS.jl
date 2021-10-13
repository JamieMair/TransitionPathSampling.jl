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

# Interface for TPSAlgorithm
abstract type TPSAlgorithm end

function init_solution(alg::TPSAlgorithm, problem::TPSProblem) end
function iterator(alg::TPSAlgorithm) end
function step!(solution, iter, alg::TPSAlgorithm) end

# Interface for TPSSolution
abstract type TPSSolution end

function finalise_solution!(solution::TPSSolution) nothing end

function solve(problem::TPSProblem, alg::TPSAlgorithm)
    solution = init_solution(alg, problem)

    for iter in iterator(alg)
        step!(solution, iter, alg)
    end

    finalise_solution!(solution)
    
    return solution
end


export solve, TPSProblem, TPSAlgorithm, TPSSolution

end