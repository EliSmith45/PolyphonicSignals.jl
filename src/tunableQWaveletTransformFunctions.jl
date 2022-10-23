

module TQWT



using FourierTools, WAV, DSP, FFTW #fast fourier transform backend
using Peaks, Distributions, Statistics, StatsBase
using DataStructures, Memoize



export analysisFB!, synthesisFB!, tqwt!, itqwt!


 function analysisFB!(V0, V1, X, n1)

    n0 = length(V0)
   # n1 = length(V1)
    ni = length(X)
    P = convert(Int, (ni-n1))
    S = convert(Int, (ni-n0))
    T = convert(Int, (n0+n1-ni))-2


    
    V0[1] = X[1]
    V0[end] = 0
    V1[1] = 0
    V1[n1]=X[end]
    
    

    Threads.@threads for p in 1:(P)
        V0[1+p] = X[1+p]
    end

    Threads.@threads for t in 1:(T)
        θ = daubechies_memoized(t/(T+1)) 
        V0[P+t+1] = X[P+t+1]*θ
        V1[t+1] = X[P+t+1]*sqrt(1-θ^2)
    end

    
    Threads.@threads for s in 1:(S)
        V1[T+1+s] = X[P+T+s+1]
    end

end

function synthesisFB!(Y, V0, V1, n1)

    n0 = length(V0)
   # n1 = length(V1)
    ni = length(Y)
    P = convert(Int, (ni-n1))
    S = convert(Int, (ni-n0))
    T = convert(Int, (n0+n1-ni))-2


    
    Y[1] = V0[1]
    Y[end] = V1[n1]

    Threads.@threads for p in 1:(P)
        Y[1+p] = V0[1+p]
    end

    Threads.@threads for t in 1:(T)
        θ = daubechies_memoized(t/(T+1)) 
        Y[P+t+1] = V0[P+t+1]*θ+ V1[t+1]*sqrt(1-θ^2)
    end


    Threads.@threads for s in 1:(S)
        Y[P+T+s+1] = V1[T+s+1]
    end
    

end


function tqwt!(coeffs; highPass, lowPass, seq, p)

    lowPass[1] = rfft(seq)
    
    for j in 1:p.J
        n1 = FourierTools.center_pos(p.seqLengths[j, 2])
        analysisFB!(lowPass[j+1], highPass[j], lowPass[j], n1)
        coeffs[j] = irfft(highPass[j], nextPower2(p.seqLengths[j, 2]))
       
    end

    sLen =  nextPower2(p.seqLengths[end, 1])
    dLen = FourierTools.center_pos(sLen)
    coeffs[end] = irfft(scale_lowpass(lowPass[end], dLen), sLen)


    #return coeffs
end

function itqwt!(coeffs; highPass, lowPass, p)
    scale_lowpass!(lowPass[end], rfft(coeffs[end]))
   # lowPass[end] = @time scale_lowpass(rfft(coeffs[end]), FourierTools.center_pos(p.seqLengths[end, 1]))
    
    for j in (p.J):-1:1
        n1 = FourierTools.center_pos(p.seqLengths[j, 2])
        scale_lowpass!(highPass[j], rfft(coeffs[j]))
        
        #highPass[j] = scale_lowpass(rfft(coeffs[j]), FourierTools.center_pos(p.seqLengths[j, 1]))
        synthesisFB!(lowPass[j], lowPass[j+1],  highPass[j], n1)
    end
    #return lowPass
    return irfft(lowPass[1], p.seqLengths[1, 3])
end
  
mutable struct tqwtParams
    Q::Float64
    r::Float64
    J::Int64
    β::Float64
    α::Float64
    seqLengths::Matrix{Int64}
    dftLengths::Matrix{Int64}
end

export tqwtParams

#in-place operations
scale_lowpass! = function(Yn, Xn)
    ni = length(Xn)
    n0=length(Yn)
    if (n0 <= ni)
        Yn .= @view(Xn[1:length(Yn)])

    elseif (n0 > ni)
        Yn[1:ni] .= Xn
    end

end
scale_highpass! = function(Yn, Xn)
    ni = length(Xn)
    n1 = length(Yn)
    if (n1 <= ni)
        Yn .= @view(Xn[(ni-n1+1):end])

    elseif (n1 > ni)
        Yn[(n1-ni+1):end] .= Xn
    end
    
end
scale_lowpass = function(Xn, n0)
    ni = length(Xn)
    Yn = zeros(eltype(Xn), n0)
    if (n0 <= ni)
        return @view(Xn[1:length(Yn)])

    elseif (n0 > ni)
        Yn[1:ni] .= Xn
    end
    return Yn
end
scale_highpass = function(Xn, n1)
    ni = length(Xn)
    Yn = zeros(eltype(Xn), n1)
    if (n1 <= ni)
        Yn = @view(Xn[(ni-n1+1):end])

    elseif (n1 > ni)
        Yn[(n1-ni+1):end] .= Xn
    end
    return Yn
end

export scale_lowpass, scale_lowpass!, scale_highpass, scale_highpass!





function init_plan(seq; Q, r, J)
    β=2/(Q+1)
    α = 1-β/r
   # N = FourierTools.center_pos(nextPower2(startLen))
   
    seq = [seq ; repeat([0], nextPower2(length(seq)) - length(seq))]
    N = length(seq)
    jMax = convert(Int, floor(log(β*N/8)/(log(1/α))))
    if J > jMax
        J = jMax
    end

    n0, n1, ni = get_sequence_sizes(α, β, N, J)
    n0d, n1d, nid = get_dft_sizes(n0, n1, ni, J)
    
    return seq, tqwtParams(Q, r, J, β, α, [n0 n1 ni], [n0d n1d nid])
end

function init_coefficients(p::tqwtParams)

    transforms = Vector{Vector{Float32}}(undef, p.J+1)
    for i in eachindex(transforms)
        if i == length(transforms)
            transforms[i] = zeros(Float32, nextPower2(p.seqLengths[i-1, 1]))
        else
            transforms[i] = zeros(Float32,  nextPower2(p.dftLengths[i, 2]))
        end
    end

    return transforms
end
    
function init_highPassVec(p::tqwtParams)
    transforms = Vector{Vector{ComplexF32}}(undef, p.J)
    for i in eachindex(transforms)
       # if i == length(transforms)
        #    transforms[i] = zeros(ComplexF32, p.dftLengths[i-1, 1])
        #else
            transforms[i] = zeros(ComplexF32, p.dftLengths[i, 2])
        #end
    end

    return transforms
end

function init_lowPassVec(p::tqwtParams)
    transforms = Vector{Vector{ComplexF32}}(undef, p.J+1)
    transforms[1] = zeros(ComplexF32, FourierTools.center_pos(p.seqLengths[1, 3]))

    for i in 1:(length(transforms) - 1)
        transforms[i+1] = zeros(ComplexF32,  FourierTools.center_pos(p.seqLengths[i, 1]))
    end

    #transforms[end] = zeros(ComplexF32, p.dftLengths[end, 1])
    return transforms
end

get_sequence_sizes = function(α, β, N, J)

    n0 = zeros(Int64, J)
    n1 = zeros(Int64, J)
    ni = zeros(Int64, J)
    
    for j in 1:J

        n0[j] = convert(Int, 2*div((α^j)*N, 2))+1
        n1[j] = convert(Int, 2*div(β*(α^(j-1))*N, 2))+1
        ni[j] = convert(Int, 2*div((α^(j-1))*N, 2))+1
    end
    ni[1] = N
    return n0, n1, ni
end

get_dft_sizes = function(n0, n1, ni, J)

    n0d = zeros(Int64, J)
    n1d = zeros(Int64, J)
    nid = zeros(Int64, J)
    
    for j in 1:J

        n0d[j] = FourierTools.center_pos(nextPower2(n0[j]))
        n1d[j] = FourierTools.center_pos(nextPower2(n1[j]))
        nid[j] = FourierTools.center_pos(nextPower2(ni[j]))

    end

    # ni[1] = FourierTools.center_pos(nextPower2(ni[j]*2-2))
    return n0d, n1d, nid
end





@memoize function daubechies_memoized(ω)
    return Float32(0.5*(1+cospi(ω))*sqrt(2-cospi(ω)))
end

nextPower2 = function(k)
    return Int(2^ceil(log2(k)))
end

export daubechies_memoized

#get impulse response of a wavelet
function getWavelets(N; p, whichLevel)
    y=zeros(Float32, N)
    
    coeffs = init_coefficients(p)
    highPass = init_highPassVec(p)
    lowPass = init_lowPassVec(p)


    tqwt!(coeffs; highPass=highPass, lowPass=lowPass, seq=y, p=p)
    #d = tqwt!(y; highPass = h, r = plan.r, seq = y, J = plan.J)
    coeffs[whichLevel][div(length(coeffs[whichLevel]), 2)] = 1
    #return coeffs
    wavelets = itqwt!(coeffs; highPass=highPass, lowPass=lowPass, p=p)
    #return wavelets

    return wavelets
end


export get_dft_sizes, get_sequence_sizes, init_lowPassVec, 
    init_highPassVec, init_coefficients, init_plan, getWavelets, 
    daubechies_memoized, nextPower2


end #end of TQWT module


