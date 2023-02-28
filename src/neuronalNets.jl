#implements neuronal networks for sparse coding of audio signal 



module NeuronalNets
include("./TimeFrequencyTools.jl")
using .TimeFrequencyTools


using LinearAlgebra, CUDA
export lca_iir, lca_iir2
#Locally competitive algorithm for highly efficient biologically-plausible sparse coding:
#
# Kaitlin L. Fair, Daniel R. Mendat, Andreas G. Andreou, Christopher J. Rozell,
# Justin Romberg and David V. Anderson. "Sparse Coding Using the Locally Competitive 
# Algorithm on the TrueNorth Neurosynaptic System"
# https://www.frontiersin.org/articles/10.3389/fnins.2019.00754/full

# This implementation uses a gammatone IIR filterbank instead of convolving the signal with an 
# overcomplete dictionary of sinusoids. This is a good first layer sparse code and is similar to 
# our brain's first layer encoding of audio. 

# X: data matrix with rows as samples
# G: correlation/inhibition strength of neurons. For an IIR filterbank, g(i, j) this is the 
# frequency response of filter i at center frequency wj
# τ: time constant, should be about 10ms
# iters: how many times neurons are updated for each time sample. Should be low (as low as 1) if 
# the audio sample rate is high (e.g. fs > 20kHz)
# returns sparse code A, representing a highly sparse time frequency distribution with implicit total 
# variation minimization. 
function hard_threshold!(x::Vector{T}, vals::Vector{T}, threshold::Number) where T <: Real
   
    for i in eachindex(vals)
        if vals[i] < threshold
            x[i] = 0
        else
            x[i] = vals[i]
        end
    end
end

function hard_threshold!(x::Vector{T}, vals::Vector{T}, threshold::Number) where T <: Complex
   
    for i in eachindex(vals)
        if mag(vals[i]) < threshold
            x[i] = 0
        else
            x[i] = vals[i]
        end
    end
end



using Peaks
###### Current best implementation
function lca_iir2(X::Matrix{T}, G::Matrix{T};  τ = .01, threshold = .03, iters = 3, save_every = 1) where T <: Real

    B = permutedims(X)
    G[diagind(G)] .= 0
    G = permutedims(G)
    tr = threshold * maximum(B)
    snapshots = div(size(B, 2), save_every) 
    A = zeros(eltype(B), size(B, 1), snapshots) #activation of each neuron, i.e. the resulting sparse code
    ut = zeros(eltype(B), size(B, 1)) 
    inhibitions = copy(ut)
    activeSet = Int64[]

    for t in axes(B, 2) 
        
        for iter in 1:iters
    
          #equivalent to: ut .+= τ .* (@view(B[:, t]) .- ut .- (G * at))
          mul!(inhibitions, G[:, activeSet], ut[activeSet])
          axpby!(τ, @view(B[:, t]) .- inhibitions, 1 - τ, ut)
          #LinearAlgebra.gemv!(ut, 'N', G[:, activeSet], ut[activeSet], -τ, 1 - τ)
          #axpy!(τ, @view(B[:, t]), ut)
          
          
          peaks = findmaxima(ut)
          activeSet = peaks[1][findall(x -> x > tr, peaks[2])]
          #findall(x -> x > tr, ut)
            
        end
       
        if (t % save_every) == 0
            A[activeSet, div(t, save_every)] .= @view(ut[activeSet])
        end
   
    end

    G[diagind(G)] .= 1

    return permutedims(A)

end




function lca_iir2(X::Matrix{T}, G::Matrix{T}, phaseShift;  τ = .01, threshold = .03, iters = 3, save_every = 1) where T <: Complex

    B = permutedims(X)
    G2 = copy(G)
    G2[diagind(G2)] .= 0
    G2 = permutedims(G2)
    tr = threshold * maximum(mag.(B))
    snapshots = div(size(B, 2), save_every) 
    A = zeros(eltype(B), size(B, 1), snapshots) #activation of each neuron, i.e. the resulting sparse code
    ut = zeros(eltype(B), size(B, 1)) 
    rt = copy(ut)
    inhibitions = copy(ut)
    activeSet = Int64[]

    for t in axes(B, 2) 
        
        for iter in 1:iters
    
            #equivalent to: ut .+= τ .* (@view(B[:, t]) .- ut .- (G * at))
            mul!(inhibitions, G2[:, activeSet], ut[activeSet])
            rt .= @view(B[:, t]) .- inhibitions
            axpby!(τ, rt, 1 - τ, ut)
            #LinearAlgebra.gemv!(ut, 'N', G[:, activeSet], ut[activeSet], -τ, 1 - τ)
            #axpy!(τ, @view(B[:, t]), ut)
            
            peaks = findmaxima(mag.(ut))
            activeSet = peaks[1][findall(x -> x > tr, peaks[2])]
           
        end
       
       
        ut[activeSet] .*= phaseShift[activeSet]
            

        if (t % save_every) == 0
            A[activeSet, div(t, save_every)] .= @view(ut[activeSet])
        end
   
    end

   

    return permutedims(A)

end


function lca_iir(X::Matrix{T}, G::Matrix, phaseShift;  τ = .01, threshold = .03, iters = 3, save_every = 1, delta = .01) where T <: Complex

    
    B = permutedims(X)
    G2 = permutedims(G)
    G2[diagind(G2)] .= 0
    
    tr = threshold * maximum(mag.(B))
    snapshots = div(size(B, 2), save_every) 
    A = zeros(eltype(B), size(B, 1), snapshots) #activation of each neuron, i.e. the resulting sparse code
    #R = copy(A)
    residual_norm = zeros(Float64, snapshots)
    ut = zeros(eltype(B), size(B, 1)) 
    last_ut = copy(ut)
    rt = copy(ut)
    inhibitions = copy(ut)
    activeSet = Int64[]

    residual_norm[1] = 1.1
    residual_norm[2] = 1.0

   
    for t in axes(B, 2) 
        iter = 1
        converged = false
        diff = 1.0
        while (iter <= iters) && !converged 
           
            mul!(inhibitions, @view(G2[:, activeSet]), @view(ut[activeSet]))
            axpby!(τ, @view(B[:, t]) .- inhibitions, 1 - τ, ut)
            
            diff = norm(last_ut .- ut) / norm(ut)
            last_ut .= copy(ut)
            activeSet = findall(x -> x > tr, mag.(ut))
            
            iter +=1 
    
            converged = (diff < delta)
            
        end
        

       
        #at[activeSet] .*= phaseShift[activeSet]
        ut[activeSet] .*= phaseShift[activeSet]
            


        if (t % save_every) == 0
            A[activeSet, div(t, save_every)] .= @view(ut[activeSet])
            rt .= @view(B[:, t]) .- inhibitions
            rt[activeSet] .-= @view(ut[activeSet])
            residual_norm[div(t, save_every)] = norm(rt)
            #R[:, div(t, save_every)] .= (D[:, activeSet]) * @view(ut[activeSet])
        end
   
    end

    #G2[diagind(G)] .= 1

    return permutedims(mag.(A)), residual_norm#permutedims(mag.(R))

end



function lca_iir(X::Matrix{T}, G::Matrix{T}, phaseShift;  τ = .01, threshold = .03, iters = 3, save_every = 1) where T <: Complex

    Xp = permutedims(X)
    B = copy(Xp)
    D = copy(G)
    foreach(normalize!, eachcol(D))
    B = permutedims(D) * B 
    G2 = D * permutedims(D)

   
    G2[diagind(G2)] .= 0
    
    #return G2

    tr = threshold * maximum(mag.(B))
    snapshots = div(size(B, 2), save_every) 
    A = zeros(eltype(B), size(B, 1), snapshots) #activation of each neuron, i.e. the resulting sparse code
    R = copy(A)
    residual_norm = zeros(Float64, iters, size(B, 2))
    ut = zeros(eltype(B), size(B, 1)) 
    rt = copy(ut)
    inhibitions = copy(ut)
    activeSet = Int64[]

    for t in axes(B, 2) 
        
        for iter in 1:iters
    
            #equivalent to: ut .+= τ .* (@view(B[:, t]) .- ut .- (G * at))
            mul!(inhibitions, G2[:, activeSet], ut[activeSet])
            axpby!(τ, @view(B[:, t]) .- inhibitions, 1 - τ, ut)
            
            mul!(rt, D[:, activeSet], ut[activeSet])
           
            rt .= mag.(@view(Xp[:, t])) .- mag.(rt)

            residual_norm[iter, t] = norm(rt) / norm(@view(B[:, t]))
            peaks = findmaxima(mag.(ut))
            activeSet = peaks[1][findall(x -> x > tr, peaks[2])]
           
        end
       
       
        ut[activeSet] .*= phaseShift[activeSet]
            

        if (t % save_every) == 0
            A[activeSet, div(t, save_every)] .= @view(ut[activeSet])
            #R[:, div(t, save_every)] .= (D[:, activeSet]) * @view(ut[activeSet])
        end
   
    end

    #G2[diagind(G)] .= 1

    return permutedims(mag.(A)), residual_norm#permutedims(mag.(R))

end

end

