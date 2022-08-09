using TPS
using SafeTestsets

@safetestset "Convergence" begin
    include("convergence/test_convergence.jl")
end

@safetestset "Trajectories" begin
    include("metropolis/test_trajectories.jl")
end

@safetestset "Simulated Annealing" begin
    include("metropolis/test_sa.jl")
end
