using TPS
using TPS.Callbacks
using TPS.MetropolisHastings
using TPS.SimulatedAnnealing
using Test
using SafeTestsets

square(x) = x * x
function loss_fn(state::AbstractArray)
    return sum(square, state) / length(state)
end

function test_problem(d)
    state = zeros(d)
    obs = TPS.SimpleObservable(loss_fn)
    return SAProblem(obs, state)
end


function setup()
    s = 0.0
    σ = 100.0
    d = 10
    problem = test_problem(d)
    alg = TPS.MetropolisHastings.gaussian_sa_algorithm(s, σ)
    return problem, alg
end

function extract_state_fn(cache::Dict{Symbol,Any}, key, should_push=false)
    if should_push
        cache[key] = []
    end
    function fn(sol::TPSSolution)
        state = deepcopy(get_current_state(sol))
        if should_push
            push!(cache[key], state)
        else
            cache[key] = state
        end
        nothing
    end
    wrapped_fn(deps::SolveDependencies) = fn(deps.solution)
    return wrapped_fn
end

@testset "Initialisation Callback" begin
    problem, alg = setup()

    cb_cache = Dict{Symbol,Any}()
    cb = InitialisationCallback(extract_state_fn(cb_cache, :initial_state))
    solve(problem, alg, 1:10; cb=cb)

    @test all(cb_cache[:initial_state] .== TPS.get_initial_state(problem))
end

@testset "Finalisation Callback" begin
    problem, alg = setup()
    cb_cache = Dict{Symbol,Any}()
    cb = FinalisationCallback(extract_state_fn(cb_cache, :final_state))
    sol = solve(problem, alg, 1:10; cb=cb)

    @test all(cb_cache[:final_state] .== get_current_state(sol))
end

@testset "Pre Inner Loop Callback" begin
    problem, alg = setup()
    cb_cache = Dict{Symbol,Any}()
    cb = PreInnerLoopCallback(extract_state_fn(cb_cache, :states, true))
    n_epochs = 10
    sol = solve(problem, alg, 1:n_epochs; cb=cb)

    @test length(cb_cache[:states]) == n_epochs
    @test all(first(cb_cache[:states]) .== TPS.get_initial_state(problem))
    # Last here is pre initialisation so should be different from end
    @test all(last(cb_cache[:states]) .!= get_current_state(sol))
end

@testset "Post Inner Loop Callback" begin
    problem, alg = setup()
    cb_cache = Dict{Symbol,Any}()
    cb = PostInnerLoopCallback(extract_state_fn(cb_cache, :states, true))
    n_epochs = 10
    sol = solve(problem, alg, 1:n_epochs; cb=cb)

    @test length(cb_cache[:states]) == n_epochs
    @test all(first(cb_cache[:states]) .!= TPS.get_initial_state(problem))
    # Last here is pre initialisation so should be different from end
    @test all(last(cb_cache[:states]) .== get_current_state(sol))
end


@testset "SetCallback" begin
    problem, alg = setup()
    cb_cache = Dict{Symbol,Any}()
    cbs = (
        InitialisationCallback(extract_state_fn(cb_cache, :initial_state)),
        FinalisationCallback(extract_state_fn(cb_cache, :final_state)),
        PreInnerLoopCallback(extract_state_fn(cb_cache, :pre_inner_loop_states, true)),
        PostInnerLoopCallback(extract_state_fn(cb_cache, :post_inner_loop_states, true))
    )
    cb = CallbackSet(cbs...)
    n_epochs = 10
    sol = solve(problem, alg, 1:n_epochs; cb=cb)

    @testset "Initialisation Callbacks" begin
        @test all(cb_cache[:initial_state] .== TPS.get_initial_state(problem))
    end
    @testset "Finalisation Callbacks" begin
        @test all(cb_cache[:final_state] .== get_current_state(sol))
    end
    
    @testset "Inner loop Callbacks" begin
        @test length(cb_cache[:pre_inner_loop_states]) == n_epochs
        @test length(cb_cache[:post_inner_loop_states]) == n_epochs
        for (pre_state, post_state) in zip(cb_cache[:pre_inner_loop_states][2:end], cb_cache[:post_inner_loop_states][1:end-1])
            @test all(pre_state .== post_state)
        end
    end
end