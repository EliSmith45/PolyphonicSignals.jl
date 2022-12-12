### gammatone filter banks
### There are two versions implemented here, one using real-valued coefficients and the other complex.
### The real-valued version outputs the filtered signal, while the complex version outputs the analytic signal.
### The real component of the complex filter output is the filtered signal while the imaginary component is the
### Hilbert transform. It can therefore be used to estimate the amplitude envelope and instantaneous frequency 
### in each channel, making it far more useful. It is also more accurate than the real-valued version implemented 
### here. Both are implemented as IIR filters. This file also implements the Meddis cochlea inner hair cell model. 


#Source for the real-coefficient filter bank: https://www.scinapse.io/papers/396690109#fullText
#Source for the complex-coefficient filter bank: https://arxiv.org/ftp/arxiv/papers/1503/1503.07015.pdf

module GammatoneFilterBanks

include("./TimeFrequencyTools.jl")
using .TimeFrequencyTools

using DSP, StatsBase, Peaks, DataStructures, LinearAlgebra, RollingFunctions

export create_gt_cascade_filterBank, applyFilterbank, getComplexGammatone, createComplexGTFilterBank, 
    applyComplexFilterbank, gt_complex_filterbank, meddis_IHC, getGTFreqResponse, erbWidth,
    getErbBins, biquadParam #gt_cascaded_filterbank


function erbWidth(f)
    erb = 24.7 * ((4.37 * f / 1000) + 1)
    return erb
end


function getErbBins(fmin, fmax, bins)
    
    erbBins = LinRange(hzToErb(fmin), hzToErb(fmax), bins)
    hzBins = erbToHz.(erbBins)
    freqs = [erbBins hzBins]

    return freqs
end

function getGT_transferFunc(params, bandwidth)
    
    cf = params.cf
    T = params.T
    order = params.order
    erbModel = params.erbModel


    if erbModel == "Glasberg"
        B = 2*pi*bandwidth*24.7*(4.37*cf/1000 + 1)
    end
    
    
    b0 = T
    b2 = 0.0

    
    if order == 1
        b1 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3 + 2^(3/2))*T*sin(2*cf*pi*T)/exp(B*T))/2
    elseif order == 2
        b1 = -(2*T*cos(2*cf*pi*T)/exp(B*T) - 2*sqrt(3 + 2^(3/2))*T*sin(2*cf*pi*T)/exp(B*T))/2
    elseif order == 3
        b1 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3 - 2^(3/2))*T*sin(2*cf*pi*T)/exp(B*T))/2
    elseif order == 4
        b1 = -(2*T*cos(2*cf*pi*T)/exp(B*T) - 2*sqrt(3 - 2^(3/2))*T*sin(2*cf*pi*T)/exp(B*T))/2
    else 
        print("Order greater than 4 not defined!")
    end

    a1 = -2*cos(2*cf*pi*T)/exp(B*T)
    a2 = exp(-2*B*T)

    coefs = (b0, b1, b2, a1, a2)
    return coefs
end

function get_4thOrder_gammatone(cf, fs, bandwidth)
    
    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 1, "Glasberg"), bandwidth)
    filt1 =  Biquad(transFun...)


    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 2, "Glasberg"), bandwidth)
    filt2 =  Biquad(transFun...)

    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 3, "Glasberg"), bandwidth)
    filt3 = Biquad(transFun...)


    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 4, "Glasberg"), bandwidth)
    filt4 = Biquad(transFun...)


    o2 = freqresp(SecondOrderSections([filt1, filt2, filt3, filt4], 1), cf * 2 * pi / fs)
    peakGain = convert.(Float64, sqrt.(o2 .* conj.(o2)))
    
    return SecondOrderSections([filt1, filt2, filt3, filt4], 1/peakGain)

end

function create_gt_cascade_filterBank(fmin, fmax, fs, bins, bandwidth)
    freqBins = getErbBins(fmin, fmax, bins)
    filts = Vector{SecondOrderSections{:z, Float64, Float64}}(undef, bins)
    
    
    for i in axes(freqBins, 1)
        filts[i] = get_4thOrder_gammatone(freqBins[i, 2], fs, bandwidth)
    
    end

    fb = gt_cascaded_filterbank(freqBins, filts)
    return fb
end

function applyFilterbank(s, fb)
    newNumChannels = size(fb.cfs, 1)
    splitAudio = Matrix{Float64}(undef, length(s), newNumChannels)
    Threads.@threads for j in axes(splitAudio, 2)
        x = DSP.filt(fb.filters[j], s)
        splitAudio[:, j] = x
    end

    return splitAudio
end



function getGTFreqResponse!(mag, fb, frq, whichCFs)
  
    Threads.@threads for filt in whichCFs
        H = freqresp(fb.filters[filt], frq)
        mag[filt - whichCFs[1] + 1] = sqrt(real(H)^2 + imag(H)^2)
        #keepFreq = findall(x -> x <= fmax && x >= fmin, W)
        #keep = findall(x -> (x in keepAmp) && ( x in keepFreq), 1:length(hmag))
    end
end
function getGTFreqResponse(fb, frq, whichCFs)
  
    mag = zeros(Float64, length(whichCFs))
    for filt in whichCFs
        H = freqresp(fb.filters[filt], frq)
        mag[filt - whichCFs[1] + 1] = sqrt(real(H)^2 + imag(H)^2)
        #keepFreq = findall(x -> x <= fmax && x >= fmin, W)
        #keep = findall(x -> (x in keepAmp) && ( x in keepFreq), 1:length(hmag))
    end
    return mag
end




function getComplexGammatone(cf, fs, bandwidth, erbStep, Ngd)
    
    erb = bandwidth * erbWidth(cf)
    Ngd = round(Int64, Ngd*fs)
    Npe = round(Int64, 3*fs/(2*π*1.019*erb))
    A = exp(-2*π*1.019*erb/fs)*cispi(2*cf/fs)
    B = sqrt(2)*erbStep*((1-A)^4)/(A + 4*A^2 + A^3) 
    C = cispi(-2*cf*minimum([Ngd, Npe])/fs)
    D = maximum([0, Ngd - Npe])
    zd = zeros(ComplexF64, D+1)
    zd[D+1] = C

    transfer = B * PolynomialRatio([0.0, A, 4*A^2, A^3], [1.0]) * PolynomialRatio([1.0], [1.0, -A])^4 * PolynomialRatio(zd, [1.0])
    
    fr = freqresp(transfer, cf*2*π/fs)
    frm = sqrt(real(fr)^2 + imag(fr)^2)
   # frm=1
    return transfer*(1/frm)
end


function createComplexGTFilterBank(fmin, fmax, fs, bins, bandwidth, Ngd)
    freqBins = getErbBins(fmin, fmax, bins)
    filts = Vector{PolynomialRatio}(undef, bins)
    
    
    for i in axes(freqBins, 1)
        if i != size(freqBins, 1)
            erbStep = (freqBins[i+1, 2] - freqBins[i, 2])/(bandwidth*erbWidth(freqBins[i, 2]))
        else
            erbStep =  (freqBins[i, 2] - freqBins[i - 1, 2])/(bandwidth*erbWidth(freqBins[i - 1, 2]))
        end

        filts[i] = getComplexGammatone(freqBins[i, 2], fs, bandwidth, erbStep, Ngd)
    end

    fb = gt_complex_filterbank(freqBins, filts)
    return fb
end

function applyComplexFilterbank(s, fb, output = "envelopes")
    newNumChannels = size(fb.cfs, 1)
    y = Matrix{ComplexF32}(undef, length(s), newNumChannels)

    Threads.@threads for j in axes(y, 2)
       DSP.filt!(@view(y[:, j]), fb.filters[j], s)
    
    
    end

    return y
end


#meddis inner hair cell model
function meddis_IHC(aud;
    fs, #sample rate
    duration, #duration in seconds
    m, #maximum storage of transmitter pool
    a, #permeability average firing rate
    b, #permeability firing rate compressor
    g, #permeability scale
    y, #factory production rate
    l, #cleft loss rate
    r, #cleft reprocessing rate
    x, #store release rate
    h #firing rate scale
    )

    dt=1/fs #time step
    sTime = 0
    eTime = duration + sTime - dt
    times = sTime:dt:eTime



    Cs = zeros(Float64, length(times),  size(aud, 2)) 
    Qt = similar(Cs)
    Wt = similar(Cs)
    

    #initialize state vars to spontaneous rates
    kt = g*a/(a+b) #cleft permeability
    Cs[1, :] .= m*y*kt/(l*kt + y*(l+r))
    Qt = Cs[1, :] .*(l+r)/kt
    Wt = Cs[1, :].*(r/x)
    #go from rates per second to rates per time step
    g*=dt
    y*=dt
    l*=dt
    r*=dt
    x*=dt
    h*=dt




    replenish=0.0
    reuptake=0.0
    reprocess=0.0
    eject=0.0

    for ch in axes(aud, 2)
        for t in (axes(@view(aud[2:end]), 1) .+ 1)
        
            st = aud[t, ch]
            kt = ((st + a) > 0) ? g*(st + a)/(st + a + b) : 0.0
            
            
            replenish = (m >= Qt[ch] ) ? y*(m-Qt[ch] ) : 0.0
        
            eject=kt*Qt[ch] 
            loss=l* Cs[t-1, ch] 
            reuptake=r*Cs[t-1, ch] 
            reprocess=x*Wt[ch] 

            Cs[t, ch] =  Cs[t-1, ch] + (eject-loss-reuptake)
            Qt[ch] += (replenish-eject+reprocess)
            Wt[ch] += (reuptake-reprocess)

        end
    end
            

    return Cs
end


### instantaneous frequency tracking
export clean_crossTalk!

function clean_crossTalk!(spectra, complexSig, wl, ovlp; neighborhood = 5, sigma_c = 1, sigma_r = 2, cohesionOrder = 3, smoothingWindow = 10, threshold, step)
    
    phaseD = instPhaseDeriv(complexSig, spectra.fs)
    smoothPhase = (windowAverages(phaseD; wl, ovlp, exclude_negative = true, weights = nothing, threshold = threshold))
   

    
    cohesion = zeros(Float64, size(smoothPhase))
    repulsion = zeros(Float64, size(smoothPhase))
    
    Threads.@threads for j in 11:(size(smoothPhase, 2) - 1)
            for i in 2:(size(smoothPhase, 1) - 1)

                neighbors = [(i - neighborhood):(i - 1) ; (i + 1):(i + neighborhood)]
                filter!(x -> (x > 1) && (x < size(spectra.cfs, 1)), neighbors)
                #neighbors = [5, 6]
                for k in neighbors, dj in [-1, 0, 1]
                   
                
                    cohesion[i, j] = sum(exp.(-1 .* 
                        ((smoothPhase[k, j + dj] .- smoothPhase[i, j + dj]) .^ 2) ./ (eps() + 2 * sigma_c ^ 2))) 
                
                    repulsion[i, j] =  sum(exp.(-1 .* 
                        ((smoothPhase[k, j + dj] .- spectra.cfs[k,  2] .- 
                        smoothPhase[i, j + dj] .+ spectra.cfs[i, 2]) .^ 2) ./ (eps() + 2 * sigma_r ^ 2)))

                    if repulsion[i, j] > cohesion[i, j]
                        spectra.channels[i, j] *= (cohesion[i, j] / (eps() + repulsion[i, j]))

                    end

                    smoothPhase[i, j] += (smoothPhase[i, j] .- spectra.cfs[i, 2]) +  step*mean(@view(smoothPhase[i, (j-10):j]) .- spectra.cfs[i, 2]) 
                end
                
            end
        
    end
    #=
    Threads.@threads for i in axes(cohesion, 1)
        cohesion[i, :] .= runmean(@view(cohesion[i, :]), smoothingWindow)
        repulsion[i, :] .= runmean(@view(repulsion[i, :]), smoothingWindow)
    end


    return cohesion', repulsion', smoothPhase' #windowAverages(permutedims(smoothPhase), wl, ovlp, true)
=#
    return smoothPhase

end



mutable struct biquadParams

    cf::Float64
    T::Float64
    order::Int64
    erbModel::String
    
end
#=
mutable struct gt_cascaded_filterbank

    cfs::Matrix{Float64}
    filters::Vector{SecondOrderSections{:z, Float64, Float64}}
    
end
=#
mutable struct gt_complex_filterbank

    cfs::Matrix{Float64}
    filters::Vector{PolynomialRatio}
    
end






end