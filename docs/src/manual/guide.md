# Package Guide

This package is designed to help quickly implement Transition Path Sampling (TPS) on user defined problems, providing a set of useful dynamics and utilities to help speed up research.

Briefly, [TPS](https://en.wikipedia.org/wiki/Transition_path_sampling) is a method to sample rare trajectories. If you have some dynamics that produces a trajectory, e.g. a discrete-time random walk, but want to observe some rare behaviour, and want to sample trajectories that are atypical, TPS provides a computationally efficient way of doing this, while still providing the correct statistics. TPS can be thought of as a type of [Metropolis-Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

## Getting started

Let's imagine generating independent discrete-time continuous-space random walk trajectories, which form a trajectory of independent parameters. This dynamics does not produce interesting trajectories, so instead, we can bias towards some observable of the parameters. If we have a trajectory ``\omega``, sampled with some probability given by ``p(\omega)``, then we can define some interesting rare-event distribution ``p'(\omega)`` which is biased towards an observable, ``\mathcal{O}``, via
```math
p'(\omega) = \frac{p(\omega) e^{-s\mathcal{O}(\omega)}}{\mathcal{Z}}, 
```
where ``\mathcal{Z}`` is the partition sum of the new distribution, which acts to normalise ``p'(\omega)``. We can define the observable as the following:
```julia
using TransitionPathSampling
obs_norm(state::AbstractArray) = sum(abs2, state) # Single state in the trajectory
obs_norm(states::AbstractArray{T}) where {T<:AbstractArray} = sum(obs_norm, states) # Trajectory of states
# Wrap the function in a SimpleObservable type
observable = TransitionPathSampling.SimpleObservable(obs_norm)
```
We can define the initial trajectory as all zeros, and create a problem with this initial trajectory and the observable.
```julia
trajectory_length = 8
num_parameters = 2
trajectory = [zeros(Float64, num_parameters) for _ in 1:trajectory_length] # Array of states
problem = TransitionPathSampling.DiscreteTrajectory.DTProblem(observable, trajectory)
```
Now we can choose an algorithm which provides our underlying dynamics:
```julia
using TransitionPathSampling.MetropolisHastings

s = 1.0 # Bias parameter
sigma = 0.1 # s.t.d of dynamics
chance_to_shoot = 2 / trajectory_length # Probability to shoot (vs probability to bridge)
max_width = 1 # Only change one state at a time
algorithm = gaussian_trajectory_algorithm(s, sigma; chance_to_shoot, max_width)
```
This algorithm will perform shooting (i.e. running the dynamics forwards or backwards in time from a fixed point in the trajectory, regenerating part of the trajectory) with a probability of ``\frac{2}{8}``, and only edit a single state on each update. We also provide a way of producing Brownian bridges between two fixed points so that states in the middle of the trajectory can be efficiently updated as well. More information on the exact dynamics can be found in [this](https://arxiv.org/abs/2209.11116) paper.

Now we have everything we need to run the dynamics:
```julia
epochs = 100 # Number of samples to make
solution = solve(problem, algorithm, epochs)
```
We can access the final state of the solution:
```julia
final_state = get_current_state(solution)
```
or, get the observations along the way:
```julia
observable_values = get_observations(solution)
```

This highlights the basic interface to the package, but it can be extended to use your own dynamics, which will be covered in the next section.