using Test
using TPS
using TPS.Annealing
using Base.Iterators

struct FakeAlgorithmParams
    param::Float64
end
struct FakeAlgorithm <: TPSAlgorithm
    parameters::FakeAlgorithmParams
end

TPS.Annealing.get_parameters_object(alg::FakeAlgorithm) = alg.parameters

start_values = [1.0, -1.0, 10.0, -10.0]
end_values = [5.0, -5.0, 15.0, -15.0]

@testset "Linear parameter annealing" for (v0, v1) in product(start_values, end_values)
    epochs = 6
    total_epochs = 20
    alg = LinearDecayAnnealedAlgorithm(FakeAlgorithm(FakeAlgorithmParams(v0)), v0, v1, epochs, :param)

    expected_values = LinRange(v0, v1, epochs)

    # Test first values
    for (epoch, expected_val) in enumerate(expected_values)
        @test isapprox(TPS.Annealing.calculate_parameter_value(alg, epoch), expected_val)
    end
    # Test values afterwards
    for epoch in (epochs+1):total_epochs
        @test isapprox(TPS.Annealing.calculate_parameter_value(alg, epoch), v1)
    end
end