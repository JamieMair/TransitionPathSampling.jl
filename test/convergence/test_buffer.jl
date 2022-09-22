using TransitionPathSampling
using TransitionPathSampling.Convergence
using Test
using Statistics
using SafeTestsets

@testset "Buffer Statistics" begin
    
    num_losses = 32
    buffer_size = num_losses ÷ 4
    losses = LinRange(1.0, 10.0, num_losses)
    buffer = TransitionPathSampling.Convergence.create_automatic_buffer(eltype(losses), buffer_size)

    for i=1:length(losses)
        push!(buffer, losses[i])
        window_length = min(i, buffer_size)
        indices = (buffer.current_time-window_length+1:buffer.current_time)
        μ_actual = Statistics.mean(losses[indices])
        μ_buffer = Statistics.mean(buffer)
        @test μ_actual ≈ μ_buffer

        if (i > 2)
            σ_actual = Statistics.std(losses[indices])
            σ_buffer = Statistics.std(buffer)

            @test σ_actual ≈ σ_buffer
        end
    end
end