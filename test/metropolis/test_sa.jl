using TransitionPathSampling
using TransitionPathSampling.MetropolisHastings
using TransitionPathSampling.SimulatedAnnealing
using Test
using SafeTestsets
using Random

Random.seed!(1234)

square(x) = x*x
function loss_fn(state::AbstractArray)
    return sum(square, state) / length(state)
end

function test_problem(d)
    state = zeros(d)
    obs = TransitionPathSampling.SimpleObservable(loss_fn)
    return SAProblem(obs, state)
end

s = 0.0
σ = 100.0
d = 10
iter = 1:10

@testset "Test cache generation" begin
    problem = test_problem(d)
    @testset "Check algorithm basics" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ);
        cache = TransitionPathSampling.generate_cache(alg, problem)
        @test typeof(cache)<:TransitionPathSampling.MetropolisHastings.GaussianSACache
        @test cache.num_parameters == d
        @test cache.observable == problem.observable
        observation = TransitionPathSampling.observe(problem.observable, TransitionPathSampling.get_initial_state(problem))
        @test observation == cache.last_observation
        @test cache.use_mask == false
    end
    @testset "Check cache bool being set correctly" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ; params_changed_frac=1.0);
        cache = TransitionPathSampling.generate_cache(alg, problem)
        @test cache.use_mask == false
    end
    @testset "Check cache parameter mask being set correctly" begin
        frac = 0.5
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ; params_changed_frac=frac);
        cache = TransitionPathSampling.generate_cache(alg, problem)
        @test cache.use_mask == true
        num_params_changed = d - sum(cache.exclude_parameter_mask)
        expected_num_params_changed = Int(round(frac*d))
        @test expected_num_params_changed == num_params_changed
    end
end

@testset "Test cache masking" begin
    problem = test_problem(d)
    function test_alg_masking(alg)
        state = deepcopy(TransitionPathSampling.get_initial_state(problem))
        cache = TransitionPathSampling.generate_cache(alg, problem)
        randn!(cache.state)
        Random.shuffle!(cache.exclude_parameter_mask)
        TransitionPathSampling.MetropolisHastings.mask_cache!(cache, state)
        @test all(cache.state[j] == state[j] for (j, is_selected) in enumerate(cache.exclude_parameter_mask) if is_selected)
        @test all(cache.state[j] != state[j] for (j, is_selected) in enumerate(cache.exclude_parameter_mask) if !is_selected)
    end
    @testset "Test no masking" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ);
        test_alg_masking(alg)
    end
    @testset "Test no masking" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ; params_changed_frac=1.0);
        test_alg_masking(alg)
    end
    @testset "Test masking" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ; params_changed_frac=0.5);
        test_alg_masking(alg)
    end
end


@testset "Test solve with simulated annealing algorithms" begin
    @testset "All parameters" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ);
        problem = test_problem(d)
        @test typeof(solve(problem, alg, iter)) <: TPSSolution
    end
    @testset "Half parameters" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_sa_algorithm(s, σ, params_changed_frac=0.5);
        problem = test_problem(d)
        @test typeof(solve(problem, alg, iter)) <: TPSSolution
    end
end