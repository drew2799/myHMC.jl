include("myHMC.jl")

@doc raw"""
        statistics(sampler::my_HMC, labels::Vector{String})

Print the main statistics for the parameters sampled.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.
- 'labels::Vector{String}': labels identifing each parameter.

"""
function statistics(sampler::my_HMC, labels::Vector{String})
    m = mean(sampler.samples)
    s = std(sampler.samples)
    for i in 1:length(sampler.samples[1])
        println(labels[i], " : ", m[i], " +- ", s[i])
    end
end

@doc raw"""
        function hist_plots(sampler::my_HMC, labels::Vector{String}, figsize::Tuple, n_max::Int, m::Vector{Float64)

Plot histograms of the distribution of each parameter.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.
- 'labels::Vector{String}': labels identifing each parameter.
- 'figsize::Tuple': size of the figure (#pixels x #pixels)
- 'n_max::Int': last parameter to show

"""
function hist_plots(sampler::my_HMC, labels::Vector{String}, figsize::Tuple, n_max::Int, m::Vector{Float64})
    dim = length(sampler.samples[1])
    if n_max<=3
        l = (1, n_max)
    else
        l = (floor(Int,n_max/3)+1, 3)
    end
    p = plot(layout=l, size=figsize, plot_title="Dim $dim")
    for i in 1:n_max
        histogram!(p, getindex.(sampler.samples,i), title=labels[i], legend = false, subplot=i)
        vline!(p, [m[i]], lw=4, color=:red, label="True Value", subplot=i)
    end
    display(p)
    return p
end

@doc raw"""
        function corner_plots(sampler::my_HMC, labels::Vector{String}, lab_font_size::Int)

Corner plots of the distributions of the parameters.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.
- 'labels::Vector{String}': labels identifing each parameter.
- 'lab_font_size::Int': font size of axes labels.

"""
function corner_plots(sampler::my_HMC, labels::Vector{String}, lab_font_size::Int64)
    data = mapreduce(permutedims, vcat, sampler.samples)
    corrplot(data, label = labels, fillcolor=:plasma,
        plot_title="Corner Plots", plot_titlevspan=0.1, tickfontsize=lab_font_size)
end

@doc raw"""
        function trace_plots(sampler::my_HMC, labels::Vector{String}, figsize::Tuple, n_max::Int)

Trace plots of the sampler for each paramaeter.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.
- 'labels::Vector{String}': labels identifing each parameter.
- 'figsize::Tuple{Int}': size of the figure (#pixels x #pixels)
- 'n_max::Int': last parameter to show

"""
function trace_plots(sampler::my_HMC, labels::Vector{String}, figsize::Tuple, n_max::Int, m::Vector{Float64})
    dim = length(sampler.samples[1])
    data = mapreduce(permutedims, vcat, sampler.samples)
    l = (n_max, 1)
    p = plot(layout=l, size=figsize, plot_title="Trace Plots - dim $dim")#(1000,400))
    for i in 1:n_max
        plot!(p, data[:,i], title=labels[i], legend = false, subplot=i)
        hline!(p, [m[i]], lw=2, color=:red, label="True Value", subplot=i)
    end
    display(p)
end 

@doc raw"""
        function mean_plots(sampler::my_HMC, labels::Vector{String}, figsize::Tuple, n_max::Int)

Evolution plots of the mean of each paramaeter.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.
- 'labels::Vector{String}': labels identifing each parameter.
- 'figsize::Tuple{Int}': size of the figure (#pixels x #pixels)
- 'n_max::Int': last parameter to show

"""
function mean_plots(sampler::my_HMC, labels::Vector{String}, figsize::Tuple, n_max::Int)
    data = mapreduce(permutedims, vcat, sampler.samples)
    l = (n_max, 1)
    p = plot(layout=l, size=figsize, plot_title="Mean Plots")#(1000,400))
    for i in 1:n_max
        cum = accumulate(+, data[:,i])
        idx = [j for j=1:length(data[:,1])]
        m = cum ./ idx
        plot!(p, m, title=labels[i], legend = false, subplot=i)
    end
    display(p)
end 

@doc raw"""
        function auto_corr_f(sampler::my_HMC, lags::Vector, labels::Vector{String}, figsize::Tuple, n_max::Int)

Plot the autocorrelation function for each parameter's samples and return the autocorrelation time.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.
- 'lags::Vector': vector of lags.
- 'labels::Vector{String}': labels identifing each parameter.
- 'figsize::Tuple{Int}': size of the figure (#pixels x #pixels)
- 'n_max::Int': last parameter to show

# Returns:
- 'τ::Float64': autocorrelation time.
"""
function auto_corr_f(sampler::my_HMC, lags::Vector, labels::Vector{String}, figsize::Tuple, n_max::Int)
    data = mapreduce(permutedims, vcat, sampler.samples)
    t = []
    l = (n_max, 1)
    p = plot(layout=l, size=figsize, plot_title="Auto-correlation Plots")#(1000,400))
    for i in 1:length(sampler.samples[1])
        acf = autocor(data[:,i], lags, demean=true)
        push!(t, (1+2*sum(acf)))
    end
    for i in 1:n_max
        acf = autocor(data[:,i], lags, demean=true)
        plot!(p, acf, title=labels[i], legend = false, subplot=i)
    end
    display(p)
    println("The auto-correlation times are: ")
    return t
end

@doc raw"""
        function FMI(sampler::my_HMC)

Compute the Bayesian fraction of missing information of tha sampler (Betancourt 2016).
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.

# Returns:
- 'fmi::Float64': fraction of missing information.
"""

function FMI(sampler::my_HMC)
    ΔE =  sampler.energy[2:end] .- sampler.energy[1:(end-1)]
    dmean_E = sampler.energy .- mean(sampler.energy)
    return sum(ΔE .^ 2)/sum(dmean_E .^ 2)
end

@doc raw"""
        function energy_plots(sampler::my_HMC)

Evolution plots of the mean of each paramaeter.
...
# Arguments:
- 'sampler::my_HMC': my_HMC object containing the samples.

"""
function energy_plots(sampler::my_HMC)
    dim = length(sampler.samples[1])
    ΔE =  sampler.energy[2:end] .- sampler.energy[1:(end-1)]
    dmed_E = sampler.energy .- median(sampler.energy)
    histogram(dmed_E, normalize=:pdf, fillalpha=1, fillcolor="blue", label=L"E-\overline{E}", plot_title="Energy plots - dim $dim")
    histogram!(ΔE, normalize=:pdf, fillalpha=0.6, fillcolor="red", label=L"\Delta E")
end
