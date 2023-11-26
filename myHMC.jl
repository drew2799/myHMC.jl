using Distributions
using Flux
using LinearAlgebra
using Random
using ProgressMeter
using Plots
using StatsPlots
using StatsBase
using LaTeXStrings

@doc raw"""
        compute_energy(q, p, V, mom_dist)

Compute the total energy at (q, p).
...
# Arguments:
- 'q::Vector': position.
- 'p::Vector': momentum.
- 'V::Function': potential energy.
- 'mom_dist::Function': distribution where the momenta are sampled from.

# Returns:
- 'E::Float64': total energy.
"""
function compute_energy(q, p, V, mom_dist)
    return(V(q)-logpdf(mom_dist,p))
end

@doc raw"""
        MH_acc_step(q0, p0, q, p, V, mom_dist)

Metropolis-Hastings acceptance step: it decides whether the new sample (q,p) is accepted or not.
...
# Arguments:
- 'q0::Vector': initial position.
- 'p0::Vector': initial momentum.
- 'q::Vector': new position.
- 'p::Vector': new momentum.
- 'V::Function': potential energy.
- 'mom_dist::Function': distribution where the momenta are sampled from.

# Returns:
- 'MH::Bool': accepted true or false.
"""
function MH_acc_step(q0, p0, q, p, V, mom_dist)
    E0 = compute_energy(q0, p0, V, mom_dist) ##Initial energy
    E = compute_energy(q, p, V, mom_dist) ##final energy
    if (log(rand(1)[1]) <= min(0, E0-E))
        return true
    else
        return false
    end
end

@doc raw"""
        leapfrog_integr(q0, p0, Σ, integr_time, step_size, pot)

Compute the leapfrog integration of hamiltonian equations with given potential energy and gaussian kinetic energy.
...
# Arguments:
- 'q0::Vector': initial position.
- 'p0::Vector': initial momentum.
- 'Σ::Matrix': inverse of mass matrix.
- 'integr_time::Float64': how long the integration path is.
- 'step_size::Float64': how long each integration step is.
- 'pot::Function': gradient of the potential energy (in the hamiltonian formalism).

# Returns:
- 'q::Vector': final position.
- 'p::Vector': final momentum.
...
"""
function leap_frog(q0, p0, Σ, integr_time, step_size, pot)

    ## I make a deep copy of the initial condition to work on without losing the information.
    q, p = copy(q0), copy(p0)
    
    ## I define a loop over the integration steps
    dV = vec(Flux.jacobian(pot, q)[1])
    p .-= step_size .* dV ./ 2.
    for n in 1:(floor(Int,integr_time/step_size))
        ## I let evolve each component of the parameter vector
        q .+= step_size .* (Σ*p)
        dV = vec(Flux.jacobian(pot, q)[1])
        if (n != floor(Int,integr_time/step_size))
            p .-= step_size .* dV
        end
    end
    p .-= step_size .* dV ./ 2.

    return q, -p
end

@doc raw"""
        my_HMC(N_samples, N_burnin, n_dim, neg_log_prob, q0, ham_integrator, integr_length, integr_step)

Structure used to create an Hamiltonian Monte Carlo sampler.
...
# Arguments:
- 'N_samples::Int': number of samples to generate.
- 'N_burnin::Int': number of warming samples of the chain (not to return).
- 'n_dim::Int': dimension of the parameters' space to sample.
- 'neg_log_prob::Function': negative log probability to sample.
- 'q0::Array': initial position in parameter space where sampling starts from.
- 'ham_integrator::Function': algorithm chosen for numerical integration of the hamiltonian equations.
- 'integr_length::Float': how long each integration path is.
- 'integr_step::Float': how long each integration step is.

"""
mutable struct my_HMC
    N_samples::Int
    N_burnin::Int
    n_dim::Int
    neg_log_prob::Function
    q0::Vector
    ham_integrator::Function
    integr_length::Float64
    integr_step::Float64

    mass_mat::Matrix
    samples::Vector
    acceptance_rate::Float64
    energy::Vector

    @doc raw"""
    Constructs all the necessary attributes for the my_HMC object.
    """
    function my_HMC(N_samples, N_burnin, n_dim, neg_log_prob, q0, ham_integrator, integr_length, integr_step)
        mass_mat = Matrix{Float64}(I,n_dim,n_dim)
        samples = [q0]
        acceptance_rate = 0.
        energy = []
        new(N_samples, N_burnin, n_dim, neg_log_prob, q0, ham_integrator, integr_length, integr_step,
            mass_mat, samples, acceptance_rate, energy)
    end
end

@doc raw"""
        best_step_size(sampler::my_HMC)

Warming phase of the algorithm to optimize the integration step.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object, the sampler to warm up.

"""
function best_step_size(sampler::my_HMC)

    println("Adapting the integration step size.")
    sampler.integr_step = 1.
    sampler.acceptance_rate = 0.
    mom_distr = Distributions.MvNormal(zeros(sampler.n_dim), sampler.mass_mat)
    Σ = inv(sampler.mass_mat)
    while sampler.acceptance_rate<0.75

        sampler.integr_step /= 10.
        acc_count = 0.
        sampler.acceptance_rate = 0.

        for i in 1:100
            p0 = vec(rand(mom_distr,1))
            ## Integrating the hamiltonian equations to get a new (q,p) proposal.
            q, p = leap_frog(sampler.samples[end],
                p0, Σ,
                sampler.integr_length,
                sampler.integr_step, 
                sampler.neg_log_prob)
            ## Metropolis-Hastings acceptance step
            if MH_acc_step(sampler.q0, p0, q, p, sampler.neg_log_prob, mom_distr)
                push!(sampler.samples, q)
                acc_count += 1
            else
                push!(sampler.samples, copy(sampler.samples[end]))
            end
        sampler.acceptance_rate = acc_count/100.
        sampler.q0 = sampler.samples[end]
        sampler.samples = [sampler.q0]
        end
    end
    #println("The acceptance rate is: ", sampler.acceptance_rate)
    sampler.acceptance_rate = 0.
    println("The best integration step size is: ", sampler.integr_step)
end

@doc raw"""
        burn_in(sampler::my_HMC)

Burn-in phase of the algorithm to approach the typical set.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object, the sampler to warm up.

"""
function burn_in(sampler::my_HMC)

    println("Approaching the typical set")
    mom_distr = Distributions.MvNormal(zeros(sampler.n_dim), sampler.mass_mat)
    sampler.acceptance_rate = 0.
    acc_count = 0.
    Σ = inv(sampler.mass_mat)
    
    @showprogress for i in 1:sampler.N_burnin
        p0 = vec(rand(mom_distr,1))
        ## Integrating the hamiltonian equations to get a new (q,p) proposal.
        q, p = leap_frog(sampler.samples[end],
            p0, Σ,
            sampler.integr_length,
            sampler.integr_step, 
            sampler.neg_log_prob)
        ## Metropolis-Hastings acceptance step
        if MH_acc_step(sampler.q0, p0, q, p, sampler.neg_log_prob, mom_distr)
            push!(sampler.samples, q)
            acc_count += 1
        else
            push!(sampler.samples, copy(sampler.samples[end]))
        end
    end
    sampler.acceptance_rate = acc_count/(sampler.N_burnin)
    sampler.q0 = sampler.samples[end]
    sampler.samples = [sampler.q0]
    
    #println("The acceptance rate is: ", sampler.acceptance_rate)   
end

@doc raw"""
        best_mass_mat(sampler::my_HMC)

Warming phase of the algorithm to optimize the mass matrix.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object, the sampler to warm up.

"""
function best_mass_mat(sampler::my_HMC)

    println("Adapting the mass matrix.")
    mom_distr = Distributions.MvNormal(zeros(sampler.n_dim), sampler.mass_mat)
    sampler.acceptance_rate = 0.
    acc_count = 0.
    Σ = inv(sampler.mass_mat)
    
    @showprogress for i in 1:100
        p0 = vec(rand(mom_distr,1))
        ## Integrating the hamiltonian equations to get a new (q,p) proposal.
        q, p = leap_frog(sampler.samples[end],
            p0, Σ,
            sampler.integr_length,
            sampler.integr_step, 
            sampler.neg_log_prob)
        ## Metropolis-Hastings acceptance step
        if MH_acc_step(sampler.q0, p0, q, p, sampler.neg_log_prob, mom_distr)
            push!(sampler.samples, q)
            acc_count += 1
        else
            push!(sampler.samples, copy(sampler.samples[end]))
        end
    end
    sampler.acceptance_rate = acc_count/100

    ns = []
    for i in 1:length(sampler.samples)
        push!(ns, sampler.samples[i] .- mean(sampler.samples))
    end
    c = ns .* transpose.(ns)
    cov = mean(c)
    sampler.mass_mat = (inv(cov) + transpose(inv(cov)))/2.
    
    sampler.q0 = sampler.samples[end]
    sampler.samples = [sampler.q0]
    
    #println("The acceptance rate is: ", sampler.acceptance_rate)
end

@doc raw"""
        warm_up(sampler::my_HMC)

The global extended warming phase of the sampler.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object, the sampler to run.

"""
function warm_up(sampler::my_HMC)

    best_step_size(sampler)
    burn_in(sampler)
    best_mass_mat(sampler)
    best_step_size(sampler)
    burn_in(sampler)
    
end

@doc raw"""
        chain_run(sampler::my_HMC)

Running the tuned sampler.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object, the sampler to run.

"""
function chain_run(sampler::my_HMC)

    println("Main chain running.")
    mom_distr = Distributions.MvNormal(zeros(sampler.n_dim), sampler.mass_mat)
    sampler.acceptance_rate = 0.
    acc_count = 0.
    Σ = inv(sampler.mass_mat)
    
    @showprogress for i in 1:sampler.N_samples
        p0 = vec(rand(mom_distr,1))
        ## Integrating the hamiltonian equations to get a new (q,p) proposal.
        q, p = leap_frog(sampler.samples[end],
            p0, Σ,
            sampler.integr_length,
            sampler.integr_step, 
            sampler.neg_log_prob)
        ## Metropolis-Hastings acceptance step
        if MH_acc_step(sampler.samples[end], p0, q, p, sampler.neg_log_prob, mom_distr)
            push!(sampler.samples, q)
            push!(sampler.energy, compute_energy(q, p, sampler.neg_log_prob, mom_distr))
            acc_count += 1
        else
            push!(sampler.samples, copy(sampler.samples[end]))
            push!(sampler.energy, compute_energy(sampler.samples[end], 
                                                    p0, sampler.neg_log_prob, mom_distr))
        end
    end
    sampler.acceptance_rate = acc_count/(sampler.N_samples)
    return sampler.samples
end
