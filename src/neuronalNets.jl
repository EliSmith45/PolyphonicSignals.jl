#implements neuronal networks for sparse coding of audio signal 



module NeuronalNets
#include("./TimeFrequencyTools.jl")
#using .TimeFrequencyTools


using LinearAlgebra#, Peaks, CUDA
#export hard_threshold!, Dnn, DnnLateral1, DnnInput3, DnnInput, DnnLateral

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


mutable struct LiquidRnn12{T} #block-sparse matrix
    W0::Matrix{T} #input weights
    W1::Matrix{T} #recurrent weights
    bw::Array{T} #weight matrix bias
    b::Array{T} #recurrent layer biases
    τ::T #recurrent layer biases

    U::Array{T}
    A::Array{T}

    delta::T
end

function LiquidRnn12(neurons, n_observations, T = Float32; W0 = rand(T, neurons[2], neurons[1]), W1 = rand(T, neurons[2], neurons[2]), bw = rand(T, neurons[2], n_observations), b = rand(T, neurons[2], n_observations), τ = .010f0, delta = .01)
    U = zeros(T, neurons[2], n_observations)
    A = copy(U)
    LiquidRnn12{T}(W0, W1, bw, b, τ, U, A, delta)
end

using Flux

function (m::LiquidRnn12)(x)
    m.A .=  tanh.((m.W1 * m.U) .+ (m.W0 * x) .+ m.b) 
    m.U .= (m.U .+ (m.delta .* m.A .* m.bw) ) ./ (1 .+ (m.delta .* ((1 / m.τ) .+ m.A)))
    hard_threshold!(m.U, copy(m.U), 0.0f0)
end


a= LiquidRnn12((500, 500), 10000)
x = rand(Float32, 500, 10000)
@time a(x)
a.U



a= Lca1((500, 500), 10000; λ = .01, τ = .01)
x = rand(Float32, 500, 10000)
@time a(x)
a.U

mutable struct Lca1{T} #block-sparse matrix
    W::Matrix{T}
    U::Array{T}
    A::Array{T}

    λ::T
    τ::T


end

function Lca1(neurons, n_observations, T = Float32; W = rand(T, neurons[2], neurons[1]), λ = .01, τ = .01)
    
 
    U = zeros(T, neurons[2], n_observations)
    A = copy(U)
    Lca1{T}(W, U, A, λ, τ)
end
function (m::Lca1)(x)
    m.U .+= m.τ .* ((m.W * m.A) .- m.U .+ x)
    hard_threshold!(m.A, m.U, m.λ)
end



a= Lca1((9, 9), 10; λ = .01, τ = .01)
x = rand(Float32, 9, 10)
a(x)
a.U

function initW(neurons, linked, T)
    W = [[convert.(T, [0 0 ; 0 0]) for i in eachindex(neurons)] for i in eachindex(neurons)]#zeros(T, blockdims..., dims...)
    for j in eachindex(neurons)
        for i in eachindex(neurons)
            if linked[j][i]
                W[j][i] = rand(T, (neurons[j], neurons[i]))
            else
                W[j][i] = 0
            end
        end
    end
    return W
end



function Dnn(neurons, n_observations, T = Float32; W = nothing, linked = nothing, λ = repeat([.01f0], inner = length(neurons)), τ = repeat([.01f0], inner = length(neurons)))
    
  
    if isnothing(linked)
        linked = [zeros(Bool, length(neurons)) for i in eachindex(neurons)]
        
        for j in eachindex(neurons)
            for i in eachindex(neurons)
                if abs(i - j) <= 1
                    linked[j][i] = true
                end
                
            end
        end
        
    end

    

    

    if isnothing(W)
        W = initW(neurons, linked, T)
        
    end
  
   
   
   # dU = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]
    U = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]
    A = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]

  
    Dnn{T}(W, linked,  U, A, λ, τ)
 
end

function (m::Dnn)()
   
    @views for j ∈ eachindex(m.W)
        m.U[j] .+= m.τ[j] .* ((m.W[j]' * m.A).- m.U[j] )
      
    end

    
    @views for j ∈ 2:length(m.A)
        hard_threshold!(m.A[j], m.U[j], m.λ[j]) 
    end
end

function (m::Dnn)(x)
    m.A[1] .= x
end









mutable struct Dnn{T} #block-sparse matrix
    W::Vector{Vector}
    linked::Vector{Vector{Bool}}

   
    #dU::Array{Array{T}}
    U::Array{Array{T}}
    A::Array{Array{T}}

    λ::Array{T}
    τ::Array{T}


end


function initW(neurons, linked, T)
    W = [[convert.(T, [0 0 ; 0 0]) for i in eachindex(neurons)] for i in eachindex(neurons)]#zeros(T, blockdims..., dims...)
    for j in eachindex(neurons)
        for i in eachindex(neurons)
            if linked[j][i]
                W[j][i] = rand(T, (neurons[j], neurons[i]))
            else
                W[j][i] = 0
            end
        end
    end
    return W
end



function Dn(neurons, n_observations, T = Float32; W = nothing, linked = nothing, λ = repeat([.01f0], inner = length(neurons)), τ = repeat([.01f0], inner = length(neurons)))
    
  
    if isnothing(linked)
        linked = [zeros(Bool, length(neurons)) for i in eachindex(neurons)]
        
        for j in eachindex(neurons)
            for i in eachindex(neurons)
                if abs(i - j) <= 1
                    linked[j][i] = true
                end
                
            end
        end
        
    end

    

    

    if isnothing(W)
        W = initW(neurons, linked, T)
        
    end
  
   
   
   # dU = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]
    U = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]
    A = [zeros(T, neurons[i], n_observations) for i in eachindex(neurons)]

  
    Dnn{T}(W, linked,  U, A, λ, τ)
 
end

function hard_threshold!(x::Array{T}, vals::Array{T}, threshold::Number) where T <: Real
   
    for i in eachindex(vals)
        if vals[i] < threshold
            x[i] = 0
        else
            x[i] = vals[i]
        end
    end
end




function (m::Dnn)()
   
    @views for j ∈ eachindex(m.W)
        m.U[j] .+= m.τ[j] .* ((m.W[j]' * m.A).- m.U[j] )
      
    end

    
    @views for j ∈ 2:length(m.A)
        hard_threshold!(m.A[j], m.U[j], m.λ[j]) 
    end
end

function (m::Dnn)(x)
    m.A[1] .= x
end



end

