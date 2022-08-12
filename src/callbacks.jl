module Callbacks
using ..TPS
using Base
using SimpleTraits

"""
An AbstractCallback is a type which is used to define some custom functional behaviour.
"""
abstract type AbstractCallback end

Base.@kwdef struct SolveDependencies{A<:TPSProblem,B<:TPSSolution,C<:TPSAlgorithm,D,E}
    problem::A
    solution::B
    algorithm::C
    cache::D = nothing
    iterator_state::E = nothing
end


run(::AbstractCallback, deps) = nothing

# Assign traits to the location in which these are run
@traitdef RunsAtInitialisation{X}
@traitdef RunsAtFinalisation{X}
@traitdef RunsPreInnerLoop{X}
@traitdef RunsPostInnerLoop{X}

# Create structs for the different timings
struct InitialisationCallback{T<:Function} <: AbstractCallback
    fn::T
end
struct FinalisationCallback{T<:Function} <: AbstractCallback
    fn::T
end
struct PreInnerLoopCallback{T<:Function} <: AbstractCallback
    fn::T
end
struct PostInnerLoopCallback{T<:Function} <: AbstractCallback
    fn::T
end
@traitimpl RunsAtInitialisation{InitialisationCallback}
@traitimpl RunsAtFinalisation{FinalisationCallback}
@traitimpl RunsPreInnerLoop{PreInnerLoopCallback}
@traitimpl RunsPostInnerLoop{PostInnerLoopCallback}

run(cb::InitialisationCallback, deps) = cb.fn(deps)
run(cb::FinalisationCallback, deps) = cb.fn(deps)
run(cb::PreInnerLoopCallback, deps) = cb.fn(deps)
run(cb::PostInnerLoopCallback, deps) = cb.fn(deps)

struct CallbackSet{A<:Union{Nothing,Tuple},B<:Union{Nothing,Tuple},C<:Union{Nothing,Tuple},D<:Union{Nothing,Tuple}} <: AbstractCallback
    initialisation_callbacks::A
    finalisation_callbacks::B
    pre_inner_loop_callbacks::C
    post_inner_loop_callbacks::D
end
@traitimpl RunsAtInitialisation{X} <- runs_at_initialisation(X)
runs_at_initialisation(::Any) = false
runs_at_initialisation(::Type{CallbackSet{A,B,C,D}}) where {A,B,C,D} = (A !== Nothing)
@traitimpl RunsAtFinalisation{X} <- runs_at_finalisation(X)
runs_at_finalisation(::Any) = false
runs_at_finalisation(::Type{CallbackSet{A,B,C,D}}) where {A,B,C,D} = (B !== Nothing)
@traitimpl RunsPreInnerLoop{X} <- runs_pre_inner_loop(X)
runs_pre_inner_loop(::Any) = false
runs_pre_inner_loop(::Type{CallbackSet{A,B,C,D}}) where {A,B,C,D} = (C !== Nothing)
@traitimpl RunsPostInnerLoop{X} <- runs_post_inner_loop(X)
runs_post_inner_loop(::Any) = false
runs_post_inner_loop(::Type{CallbackSet{A,B,C,D}}) where {A,B,C,D} = (D !== Nothing)

function tuple_or_nothing_if_empty(t)
    if length(t) == 0
        return nothing
    else
        return t
    end
end

function CallbackSet(callbacks...)
    initialisation_callbacks = Tuple(cb for cb in callbacks if istrait(RunsAtInitialisation{typeof(cb)}))
    finalisation_callbacks = Tuple(cb for cb in callbacks if istrait(RunsAtFinalisation{typeof(cb)}))
    pre_inner_loop_callbacks = Tuple(cb for cb in callbacks if istrait(RunsPreInnerLoop{typeof(cb)}))
    post_inner_loop_callbacks = Tuple(cb for cb in callbacks if istrait(RunsPostInnerLoop{typeof(cb)}))
    return CallbackSet(
        tuple_or_nothing_if_empty(initialisation_callbacks),
        tuple_or_nothing_if_empty(finalisation_callbacks),
        tuple_or_nothing_if_empty(pre_inner_loop_callbacks),
        tuple_or_nothing_if_empty(post_inner_loop_callbacks)
    )
end

run_cb_at_initialisation!(::Nothing, deps) = nothing
run_cb_at_finalisation!(::Nothing, deps) = nothing
run_cb_pre_inner_loop!(::Nothing, deps) = nothing
run_cb_post_inner_loop!(::Nothing, deps) = nothing
function run_cb_at_initialisation!(cb::AbstractCallback, deps)
    istrait(RunsAtInitialisation{typeof(cb)}) && run(cb, deps)
    nothing
end
function run_cb_at_finalisation!(cb::AbstractCallback, deps)
    istrait(RunsAtFinalisation{typeof(cb)}) && run(cb, deps)
    nothing
end
function run_cb_pre_inner_loop!(cb::AbstractCallback, deps)
    istrait(RunsPreInnerLoop{typeof(cb)}) && run(cb, deps)
    nothing
end
function run_cb_post_inner_loop!(cb::AbstractCallback, deps)
    istrait(RunsPostInnerLoop{typeof(cb)}) && run(cb, deps)
    nothing
end

function run_cb_at_initialisation!(cb::CallbackSet, deps)
    if istrait(RunsAtInitialisation{typeof(cb)})
        for init_cb in cb.initialisation_callbacks
            run(init_cb, deps)
        end
    end
    nothing
end
function run_cb_at_finalisation!(cb::CallbackSet, deps)
    if istrait(RunsAtFinalisation{typeof(cb)})
        for init_cb in cb.finalisation_callbacks
            run(init_cb, deps)
        end
    end
    nothing
end
function run_cb_pre_inner_loop!(cb::CallbackSet, deps)
    if istrait(RunsPreInnerLoop{typeof(cb)})
        for init_cb in cb.pre_inner_loop_callbacks
            run(init_cb, deps)
        end
    end
    nothing
end
function run_cb_post_inner_loop!(cb::CallbackSet, deps)
    if istrait(RunsPostInnerLoop{typeof(cb)})
        for init_cb in cb.post_inner_loop_callbacks
            run(init_cb, deps)
        end
    end
    nothing
end

export InitialisationCallback, FinalisationCallback, PreInnerLoopCallback, PostInnerLoopCallback, CallbackSet, SolveDependencies

end