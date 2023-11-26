include("myHMC.jl")
include("statsplots.jl")

c = rand(10,10)
c = (c+c') ./ 2.
cov = c + 0.2*maximum(eigvals(c))*I
inv_cov = inv(cov)
d_cov = det(cov)

m = 10 .* vec(rand(10,1)) .- 5.

function neg_log_prob(q)
    return 0.5*((q-m)'*inv_cov*(q-m)) + 0.5*d_cov
end

samples = 40000
burnin  = 10000
n_dim = 10
q0 = vec(rand(10,1))
T = 1
e = 0.1
sampler = my_HMC(samples, burnin, n_dim, neg_log_prob, q0, leap_frog, T, e)

warm_up(sampler)
chain_run(sampler)

labels = ["q$i" for i=1:length(sampler.samples[1])]
println("Statistics: ", statistics(sampler, labels))
println("The accptance rate of the sampler is: ", sampler.acceptance_rate)

hist_plots(sampler, labels, (1000,500), 2, m)
trace_plots(sampler, labels, (1000,400), 2, m)
mean_plots(sampler, labels, (1000,400), 2)

lags = [i for i=0:500]
t = auto_corr_f(sampler, lags, labels, (1000,400), 2)

print("FMI: ", FMI(sampler))
energy_plots(sampler)
