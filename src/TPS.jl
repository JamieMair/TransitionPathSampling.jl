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

# Interface for TPSSolution
abstract type TPSSolution end
get_problem(solution::TPSSolution) = error("Unimplemented.")
get_current_state(solution::TPSSolution) = error("Unimplemented.")
set_current_state!(solution::TPSSolution, state) = error("Unimplemented.")
get_observable_type(solution::TPSSolution) = error("Unimplemented.")


function finalise_solution!(solution::TPSSolution) nothing end
function step!(solution::TPSSolution, alg::TPSAlgorithm, iter, args...; kwargs...) end
function step!(::Nothing, solution::TPSSolution, alg::TPSAlgorithm, iter, args...; kwargs...) 
    step!(solution::TPSSolution, alg::TPSAlgorithm, iter, args...; kwargs...)
end
function step!(cache, solution::TPSSolution, alg::TPSAlgorithm, iter, args...; kwargs...) end
function generate_cache(alg::TPSAlgorithm, problem::TPSProblem, args...; kwargs...) nothing end

function get_iterator(iter, args...; kwargs...)
    if typeof(iter)==Int
        return 1:iter
    else
        return iter
    end
end
# Must override this to be able to access the epoch information
get_epoch_from_state(iter_state) = iter_state

function solve(problem::TPSProblem, alg::TPSAlgorithm, iterator, args...; kwargs...)
    solution = init_solution(alg, problem, args...; kwargs...)
    iter = get_iterator(iterator; problem=problem, solution=solution, algorithm=alg)
    cache = generate_cache(alg, problem, args...; kwargs...)
    for iter_state in iter
        step!(cache, solution, alg, iter_state, args...; kwargs...)
    end

    finalise_solution!(solution)    
    return solution
end

export solve, TPSProblem, TPSAlgorithm, TPSSolution, get_epoch_from_state

## Includes ##
include("observables.jl")
include("default_solutions.jl")

# Problems
include("sa_problem.jl")
include("discrete_trajectory_problem.jl")
# Algorithms
include("metropolis_hastings/mh.jl")

# Convergence
include("convergence/convergence.jl")

# Annealing
include("annealing/annealing.jl")

# Callbacks
include("callbacks.jl")

end