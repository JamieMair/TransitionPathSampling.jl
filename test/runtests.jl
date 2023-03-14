using TransitionPathSampling
using SafeTestsets

# @safetestset "Convergence" begin
#     include("convergence/test_convergence.jl")
# end

# @safetestset "Trajectories" begin
#     include("metropolis/test_trajectories.jl")
# end

# @safetestset "Simulated Annealing" begin
#     include("metropolis/test_sa.jl")
# end

# @safetestset "Callbacks" begin
#     include("callbacks/test_callbacks.jl")
# end

# @safetestset "Annealing" begin
#     include("annealing/annealing.jl")
# end

@safetestset "Minibatch" begin
    include("minibatch/minibatch.jl")
end

