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


run(cb::AbstractCallback, deps::SolveDependencies) = nothing

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

struct CallbackSet{A<:Union{Nothing,NTuple},B<:Union{Nothing,NTuple},C<:Union{Nothing,NTuple},D<:Union{Nothing,NTuple}}
    initialisation_callbacks::A
    finalisation_callbacks::B
    pre_inner_loop_callbacks::C
    post_inner_loop_callbacks::D
end
@traitimpl RunsAtInitialisation{X} <- runs_at_initialisation(X)
runs_at_initialisation(cb) = false
runs_at_initialisation(cb_set::CallbackSet) = !isnothing(cb_set.initialisation_callbacks)
@traitimpl RunsAtFinalisation{X} <- runs_at_finalisation(X)
runs_at_finalisation(cb) = false
runs_at_finalisation(cb_set::CallbackSet) = !isnothing(cb_set.finalisation_callbacks)
@traitimpl RunsPreInnerLoop{X} <- runs_pre_inner_loop(X)
runs_pre_inner_loop(cb) = false
runs_pre_inner_loop(cb_set::CallbackSet) = !isnothing(cb_set.pre_inner_loop_callbacks)
@traitimpl RunsPostInnerLoop{X} <- runs_post_inner_loop(X)
runs_post_inner_loop(cb) = false
runs_post_inner_loop(cb_set::CallbackSet) = !isnothing(cb_set.post_inner_loop_callbacks)

function tuple_or_nothing_if_empty(t)
    if length(t) == 0
        return nothing
    else
        return t
    end
end

function CallbackSet(callbacks...)
    initialisation_callbacks = (cb for cb in callbacks if istrait(RunsAtInitialisation{typeof(cb)}))
    finalisation_callbacks = (cb for cb in callbacks if istrait(RunsAtFinalisation{typeof(cb)}))
    pre_inner_loop_callbacks = (cb for cb in callbacks if istrait(RunsPreInnerLoop{typeof(cb)}))
    post_inner_loop_callbacks = (cb for cb in callbacks if istrait(RunsPrePostLoop{typeof(cb)}))
    return CallbackSet(
        tuple_or_nothing_if_empty(initialisation_callbacks),
        tuple_or_nothing_if_empty(finalisation_callbacks),
        tuple_or_nothing_if_empty(pre_inner_loop_callbacks),
        tuple_or_nothing_if_empty(post_inner_loop_callbacks)
    )
end

function run_cb_at_initialisation!(cb::AbstractCallback, deps::SolveDependencies)
    istrait(RunsAtInitialisation{cb}) && run(cb, deps)
    nothing
end
function run_cb_at_finalisation!(cb::AbstractCallback, deps::SolveDependencies)
    istrait(RunsAtFinalisation{cb}) && run(cb, deps)
    nothing
end
function run_cb_pre_inner_loop!(cb::AbstractCallback, deps::SolveDependencies)
    istrait(RunsPreInnerLoop{cb}) && run(cb, deps)
    nothing
end
function run_cb_post_inner_loop!(cb::AbstractCallback, deps::SolveDependencies)
    istrait(RunsPostInnerLoop{cb}) && run(cb, deps)
    nothing
end

function run_cb_at_initialisation!(cb::CallbackSet, deps::SolveDependencies)
    if istrait(RunsAtInitialisation{cb})
        for init_cb in cb.initialisation_callbacks
            run(cinit_cb, deps)
        end
    end
    nothing
end
function run_cb_at_finalisation!(cb::CallbackSet, deps::SolveDependencies)
    if istrait(RunsAtFinalisation{cb})
        for init_cb in cb.finalisation_callbacks
            run(cinit_cb, deps)
        end
    end
    nothing
end
function run_cb_pre_inner_loop!(cb::CallbackSet, deps::SolveDependencies)
    if istrait(RunsPreInnerLoop{cb})
        for init_cb in cb.pre_inner_loop_callbacks
            run(cinit_cb, deps)
        end
    end
    nothing
end
function run_cb_post_inner_loop!(cb::CallbackSet, deps::SolveDependencies)
    if istrait(RunsPostInnerLoop{cb})
        for init_cb in cb.post_inner_loop_callbacks
            run(cinit_cb, deps)
        end
    end
    nothing
end

export InitialisationCallback, FinalisationCallback, PreInnerLoopCallback, PostInnerLoopCallback, CallbackSet

end