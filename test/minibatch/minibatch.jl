include("setup.jl")


@testset "Test cache generation" begin
    s = 1.0
    σ = 1.0
    @testset "SA Cache" begin
        problem = create_problem(x_data, y_data, batch_size, 1)
        inner_alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ);
        alg = MinibatchMHAlg(inner_alg, s)
        @test typeof(TransitionPathSampling.generate_cache(alg, problem))<:TransitionPathSampling.Minibatch.BatchMHCache
    end
    @testset "Trajectory Cache" begin
        problem = create_problem(x_data, y_data, batch_size, 8)
        inner_alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
        alg = MinibatchMHAlg(inner_alg, s)
        @test typeof(TransitionPathSampling.generate_cache(alg, problem))<:TransitionPathSampling.Minibatch.BatchMHCache
    end
end

@testset "50% Acceptance rate for same parameters for single model" begin
    s = 1.0
    tau = 1
    sigma = 0
    params_a, params_b = get_params(d, tau, sigma)

    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 10_000)
    @test isapprox(mean(acceptances), 0.5, atol=0.01)
    @test histogram.values[begin] == histogram.total_entries
end
@testset "50% Acceptance rate for same parameters for trajectory" begin
    s = 1.0
    tau = 4
    sigma = 0
    params_a, params_b = get_params(d, tau, sigma)

    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 10_000)
    @test isapprox(mean(acceptances), 0.5, atol=0.01)
    @test histogram.values[begin] == histogram.total_entries
end

@testset "Small perturbation acceptance rates" begin
    s = 1e-5
    tau = 1
    sigma = 1e-3
    params_a, params_b = get_params(d, tau, sigma)
    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 10_000)
    @test isapprox(mean(acceptances), 0.5, atol=0.05)
    @test histogram.values[begin] == histogram.total_entries
end

@testset "Small perturbation acceptance rates (trajectory)" begin
    s = 1e-5
    tau = 4
    sigma = 1e-3
    params_a = [randn(d) for _ in 1:tau]
    params_b = deepcopy(params_a)
    for p in params_b
        p .+= randn(size(p)).*sigma
    end

    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 10_000)
    @test isapprox(mean(acceptances), 0.5, atol=0.05)
    @test histogram.values[begin] == histogram.total_entries
end

@testset "High s acceptance rates (single)" begin
    tau = 1
    sigma = 1e-1
    params_a, params_b = get_params(d, tau, sigma)

    loss_a, loss_b = measure_losses(x_data, y_data, tau, params_a, params_b)
    delta_l = sum(loss_b) - sum(loss_a)
    s = 1 / abs(delta_l) # set s to have a high rejection/acceptance rate
    true_acceptance = inv(1+exp(s*delta_l))
    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 50_000)
    @test isapprox(mean(acceptances), true_acceptance, atol=0.02)
end

@testset "High s acceptance rates (trajectory)" begin
    tau = 4
    sigma = 1e-1
    params_a, params_b = get_params(d, tau, sigma)

    loss_a, loss_b = measure_losses(x_data, y_data, tau, params_a, params_b)
    delta_l = sum(loss_b) - sum(loss_a)
    s = 1 / abs(delta_l) # set s to have a high rejection/acceptance rate
    true_acceptance = inv(1+exp(s*delta_l))
    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 50_000)
    @test isapprox(mean(acceptances), true_acceptance, atol=0.02)
end

@testset "Negative High s acceptance rates (single)" begin
    tau = 1
    sigma = 1e-1
    params_a, params_b = get_params(d, tau, sigma)

    loss_a, loss_b = measure_losses(x_data, y_data, tau, params_a, params_b)
    delta_l = sum(loss_b) - sum(loss_a)
    s = -1 / abs(delta_l) # set s to have a high rejection/acceptance rate
    true_acceptance = inv(1+exp(s*delta_l))
    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 50_000)
    @test isapprox(mean(acceptances), true_acceptance, atol=0.02)
end

@testset "Negative High s acceptance rates (trajectory)" begin
    tau = 4
    sigma = 1e-1
    params_a, params_b = get_params(d, tau, sigma)

    loss_a, loss_b = measure_losses(x_data, y_data, tau, params_a, params_b)
    delta_l = sum(loss_b) - sum(loss_a)
    s = -1 / abs(delta_l) # set s to have a high rejection/acceptance rate
    true_acceptance = inv(1+exp(s*delta_l))
    acceptances, histogram = test_acceptance(x_data, y_data, batch_size, tau, s, params_a, params_b, 1.0, 50_000)
    @test isapprox(mean(acceptances), true_acceptance, atol=0.02)
end
