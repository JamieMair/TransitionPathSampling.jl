using TPS
using SafeTestsets

@safetestset "Convergence" begin
    include("convergence/test_convergence.jl")
end

@safetestset "Trajectories" begin
    include("trajectories/test_trajectories.jl")
end

