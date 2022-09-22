using TransitionPathSampling
using TransitionPathSampling.MetropolisHastings
using TransitionPathSampling.DiscreteTrajectory
using SafeTestsets
using Random

Random.seed!(1234)

square(x) = x*x
function loss_fn(state::AbstractArray{T}) where {T<:AbstractArray}
    return [loss_fn(s) for s in state]
end
function loss_fn(state::AbstractArray)
    return sum(square, state) / length(state)
end

function test_problem(n, d)
    states = [zeros(d) for _ in 1:n]
    obs = TransitionPathSampling.SimpleObservable(loss_fn)
    return DTProblem(obs, states)
end

s = 0.0
σ = 100000.0
n = 8
d = 10
iter = 1:10

@testset "Test cache generation" begin
    problem = test_problem(n, d)
    @testset "Check algorithm basics" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
        cache = TransitionPathSampling.generate_cache(alg, problem)
        @test typeof(cache)<:TransitionPathSampling.MetropolisHastings.GaussianMHTrajectoryCache
        @test cache.num_parameters == d
        @test cache.num_models == n
        @test cache.observable == problem.observable
        observation = TransitionPathSampling.observe(problem.observable, TransitionPathSampling.get_initial_state(problem))
        @test all(observation .== cache.last_observation)
        rand!(cache.cached_observation)
        cache.cached_observation .+= 1000
        # Make sure the cached observation is not pointing to same memory as last observation
        @test all(cache.cached_observation .!= cache.last_observation)
    end
    @testset "Check cache bool being set correctly" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
        cache = TransitionPathSampling.generate_cache(alg, problem)
        @test cache.use_mask == false
    end
    @testset "Check cache parameter mask being set correctly" begin
        frac = 0.5
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ; params_changed_frac=frac);
        cache = TransitionPathSampling.generate_cache(alg, problem)
        @test cache.use_mask == true
        num_params_changed = d - sum(cache.exclude_parameter_mask)
        expected_num_params_changed = Int(round(frac*d))
        @test expected_num_params_changed == num_params_changed
    end
end

@testset "Test bridge method" begin
    problem = test_problem(n, d)
    alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
    states = deepcopy(TransitionPathSampling.get_initial_state(problem))
    cache = TransitionPathSampling.generate_cache(alg, problem)
    function test_bridge(start_index, end_index)
        temp_cache = deepcopy(cache)
        range = (start_index+1):(end_index-1)
        TransitionPathSampling.MetropolisHastings.bridge!(temp_cache, states, start_index, end_index, σ)
        for i in 1:n
            if i in range
                @test all(temp_cache.state_cache[i] .!= cache.state_cache[i])
            else
                @test all(temp_cache.state_cache[i] .== cache.state_cache[i])
            end
        end
        nothing
    end
    @testset "Test bridge from both ends" begin
        test_bridge(1, n)
    end
    @testset "Test bridge in middle" begin
        test_bridge(3, 7)
        test_bridge(2, 8)
        test_bridge(1, 4)
    end
end

@testset "Test shoot method" begin
    problem = test_problem(n, d)
    @testset "Test shoot with all parameters" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
        states = deepcopy(TransitionPathSampling.get_initial_state(problem))
        cache = TransitionPathSampling.generate_cache(alg, problem)

        function test_shoot(start_index, forwards)
            temp_cache = deepcopy(cache)
            TransitionPathSampling.MetropolisHastings.shoot!(temp_cache, states, start_index, σ, forwards)
            shoot_indices = forwards ? ((start_index + 1):n) : ((start_index - 1):-1:1)
            for i = 1:n
                if i in shoot_indices
                    @test all(temp_cache.state_cache[i] .!= cache.state_cache[i])
                else
                    @test all(temp_cache.state_cache[i] .== cache.state_cache[i])
                end
            end
        end
        
        @testset "Test shoot forwards from start" begin
            test_shoot(1, true)
        end
        @testset "Test shoot backwards from end" begin
            test_shoot(n, false)
        end
        @testset "Test shoot forwards from middle" begin
            test_shoot(5, true)
        end
        @testset "Test shoot backwards from middle" begin
            test_shoot(5, false)
        end
    end
end

@testset "Test cache masking" begin
    problem = test_problem(n, d)
    function test_alg_masking(alg)
        states = deepcopy(TransitionPathSampling.get_initial_state(problem))
        cache = TransitionPathSampling.generate_cache(alg, problem)
        changed_indices = 3:6
        # Mutate the variables inside
        for i in changed_indices
            rand!(cache.state_cache[i])
        end
        Random.shuffle!(cache.exclude_parameter_mask)

        TransitionPathSampling.MetropolisHastings.mask_cache!(cache, states)
        for i in changed_indices
            @test all(cache.state_cache[i][j] == states[i][j] for (j, is_selected) in enumerate(cache.exclude_parameter_mask) if is_selected)
            @test all(cache.state_cache[i][j] != states[i][j] for (j, is_selected) in enumerate(cache.exclude_parameter_mask) if !is_selected)
        end
    end
    @testset "Test no masking" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
        test_alg_masking(alg)
    end
    @testset "Test masking" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ; params_changed_frac=0.5);
        test_alg_masking(alg)
    end
end

@testset "Test solve with trajectory algorithms" begin
    @testset "Test shooting algorithm" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ);
        problem = test_problem(n, d)
        @test typeof(solve(problem, alg, iter)) <: TPSSolution
    end
    @testset "Test bridging + shooting algorithm" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ; params_changed_frac=1.0, chance_to_shoot=0.5);
        problem = test_problem(n, d)
        @test typeof(solve(problem, alg, iter)) <: TPSSolution
    end
    @testset "Test bridging + shooting algorithm (single element)" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ; params_changed_frac=1.0, chance_to_shoot=0.5, max_width=1);
        problem = test_problem(n, d)
        @test typeof(solve(problem, alg, iter)) <: TPSSolution
    end
    @testset "Test bridging + shooting algorithm (single element), half parameters" begin
        alg = TransitionPathSampling.MetropolisHastings.gaussian_trajectory_algorithm(s, σ; params_changed_frac=0.5, chance_to_shoot=0.5, max_width=3);
        problem = test_problem(n, d)
        @test typeof(solve(problem, alg, iter)) <: TPSSolution
    end
end