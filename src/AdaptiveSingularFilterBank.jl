#Adaptive singular filter bank. This is a completely novel method that I've implemented
#for superresolution time-frequency analysis. Typical time-frequency methods struggle for 
#music, as it is filled with nearly-overlapping partials. These are sinusoidal components
#that have similar but distinct frequencies such that when superimposed, they produce an
#amplitude modulation that can be heard as a "beating" sound. The resulting signal has an 
#amplitude that modulates according to some beat frequency proportional to the frequency 
#difference of the two components, making it very difficult to track the peaks in a time-
#frequency distribution.

#This method aims to solve this issue by finding the true amplitude, phase, and frequency 
#of all of the original components in a complex mixture of sinusoids, even when multiple 
#sinusoids are in the frequency range of each other to cause beating. This makes it useful 
#for music where we expect overlapping partials, but not periodic amplitude modulation. 
#It therefore assumes periodic amplitude modulation is an artifact of the combination of 
#the original components and not an actual feature of the underlying signal. This shouldn't
#be a problem for most signals as this type of time-frequency decomposition probably isn't 
#relevant to traditional amplitude demodulation tasks. Also, it handles aperiodic amplitude
#modulation like exponential damping just fine. 

#The method works by cleaning up the output of a complex valued filterbank such as a 
#short-time Fourier transform or gammatone filterbank. It takes a windowed segment of 
#each frequency channel whose energy is above some threshold, then performs singular 
#spectrum analysis. That is, it performs an eigen decomposition on the covariance matrix
#of the windowed signal. The leading eigenvectors will be the exact frequency components, 
#so they act like the columns in a Fourier basis, except with the exact frequency found in
#the signal. The leading eigenvectors are found for each channel and are then clustered, 
#as neighboring channels will detect some of the same frequency components.

#The centroids of each cluster represent the true frequency components found in the signal
#and act like a new Fourier basis, except with exact instantaneous frequencies extracted from
#the signal. One then needs to express the signal as a linear combination of these basis vectors
#to decompose the signal. The Fourier transform does this with a dot product, which is prone to 
#spectral leakage whenever the basis isn't truly orthogonal - which requires an infinitely long
#signal. To circumvent this issue, the new decomposition is found via optimization. Rather than
#solving the optimization problem for the original windowed signal, we continue to work with the 
#windowed channels produced by the original filtering. 

#For this, we assume that the signal is stationary throughout the window, and that each basis 
#component will leak through to several channels. We can use the frequency response of the
#filters and the (optionally averaged) instantaneous frequency of each basis vector (calculated
#as the derivative of the phase, which is implemented in this package in a manner that doesn't 
#require unwrapping) to map the true amplitude to that which is observed in each frequency channel.
#The optimization problem is then min ||Ax - y||₂ where y is the windowed filterbank output written
#as a tall vector y = [y1, y2, ..., yK]ᵀ for K frequency channels, where yₖ = [yₖ₁, yₖ₂, ..., yₖₘ] 
#where m is the signal length. Each column of A is a basis vector found as before, but they are 
#repeated K times and stacked vertically to match the dimension of y. Each duplicate corresponds
#to a filterbank channel and must be multiplied by the frequency response of that channel at the 
#frequency represented by that basis. 

#The matrix A will be quite tall - with a signal sample rate of 20kHz and a window size of 26ms 
#(512 samples), the STFT will have 256 positive frequency channels and A will have 131,072 rows. 
#If there are assumed (or allowed) to be 10 sources present simultaneously, A will have 1.3 million
#elements. However, most of A and y are irrelevent, so one should try to remove the meaningless 
#components. 

#For a given frequency, the gain for most channels is very small, so one may assume that the support 
#of the gain of the basis column only covers a small number of nearby channels. This lets A be sparse.
#the signal y will also be negligible for most channels, so a simple thresholding step could allow one 
#to eliminate most of the rows of A and y. In general, any channel that isn't in the support of any of
#the basis columns should be eliminated. Next, one may downsample the signal. To get precise estimates
#for the basis, one should estimate them from a high sample rate signal. Once estimated, each channel and
#eigenvector can then be appropriately downsampled, probably by a very large amount. The final way to
#shrink A is to use an initial TF distribution that uses fewer channels. An IIR gammatone filterbank
#with 50-100 logarithmically spaced frequency bins could be equally informative as a STFT. The IIR 
#approach will also be faster and won't automatically downsample like the STFT with a step size larger 
#than 1. The complex gammatone in general is superior to the STFT for this method for numerous other 
#reasons. 

#This algorithm is adaptive, meaning the basis is updated for each window. This could get quite expensive
#if the covariance matrix and eigen decomposition were recalculated fully for each window, and most of the 
#time they won't change anyway. This method therefore uses an efficient online SVD scheme to form the basis 
#vectors. It will hopefully also support GPU usage in the future, and currently uses highly optimized 
#BLAS/LAPACK matrix operations from the LinearAlgebra package.

module AdaptiveSingularFilterBank
using DSP, StatsBase, LinearAlgebra, Peaks, Clustering, RollingFunctions, LowRankApprox

include("./TimeFrequencyTools.jl")
using ..TimeFrequencyTools





function ifreq_envelope(signal, cfs, fs, threshold, nn_oneside, sigma)
    amps = mag.(signal)
    signal ./= (eps() .+ amps)
    ifrs = zeros(Float32, size(amps)) 
    threshold *= maximum(mag.(amps))
    fdiff = (0.5*(cfs[2, 1] - cfs[1, 1]))
   
    dists = copy(ifrs)
    Threads.@threads for k in axes(ifrs, 2)
        ifreq!(@view(ifrs[:, k]), @view(signal[:, k]), fs)
    end

    ifrs = permutedims(ifrs)
    Threads.@threads for t in axes(ifrs, 2)
        for k in (nn_oneside + 1):(size(ifrs, 1) - nn_oneside)
            if amps[t, k] > threshold
                for n in -nn_oneside:nn_oneside
                    amps[t, k] *= exp(-((ifrs[k, t] - ifrs[k + n, t])^2) / sigma)
                end
            else 
                amps[t, k] = 0
            end

        end

        for k in 1:nn_oneside
            if amps[t, k] > threshold
                for n in (-k + 1):nn_oneside
                    amps[t, k] *= exp(-((ifrs[k, t] - ifrs[k + n, t])^2) / sigma)
                end
            else 
                amps[t, k] = 0
            end
        end
        
        for k in (size(amps, 2) - nn_oneside + 1):size(amps, 2)
            if amps[t, k] > threshold
                for n in -nn_oneside:0
                    amps[t, k] *= exp(-((ifrs[k, t] - ifrs[k + n, t])^2) / sigma)
                end
            else 
                amps[t, k] = 0
            end
        end
    end


    return ifrs, amps
end



########## sliding window SSA with frequency channel filtering
export trackSubspaceLR, link_comps, recreate_signal, envelopes


function updateHankel!(hk, xk, ar_lags, wl)
   
   # hk = zeros(ComplexF32, ar_lags, wl - ar_lags + 1)
    for j in 1:(wl - ar_lags + 1)
        hk[:, j] .= @view(xk[j:(j + ar_lags - 1)])
    end

    hk .-= mean(hk, dims = 2)
end

function trackSubspaceLR(tfd; ri, wl, ar_lags, step, overall_components, channel_components, min_eigval)
    
 
    s = zeros(eltype(tfd), size(tfd, 2))
    
    nWindows = size(ri, 2)
    
    bases = [zeros(ComplexF32, ar_lags, overall_components) for i in 1:nWindows]
    activations = [zeros(ComplexF32, wl - ar_lags + 1, overall_components) for i in 1:nWindows]
    comps = zeros(Int32, nWindows)

    explainedVar = [zeros(Float32, overall_components) for i in 1:nWindows]
    hankels = [zeros(ComplexF32, ar_lags, wl - ar_lags + 1) for i in axes(tfd, 1)]
   
    for w in 1:nWindows
        if w == nWindows
            times = (size(tfd, 2) - wl):(size(tfd, 2))
        else
            times = ((w - 1)*step + 1):((w - 1)*step + wl)
        end
     
        println(w)
        
        peaks = findmaxima(@view(ri[:, w]))
        bases[w] = zeros(ComplexF32, ar_lags, channel_components * length(peaks[1])) 
        explainedVar[w] = zeros(ComplexF32, channel_components * length(peaks[1])) 
        added = 0
        while !isempty(peaks[1])
            
            p = pop!(peaks[1])
            updateHankel!(hankels[p], @view(tfd[p, times]), ar_lags, wl)
   

    
            u, s, v  = psvd(hankels[p], rank = channel_components, sketch = :sub, maxdet_niter = 100000, maxdet_tol = 5 * eps(), sketchfact_adap = true)
            
            keep = findfirst(x -> x < (min_eigval .* maximum(s)), s) 
            if isnothing(keep)
                keep = length(s)
            else
                keep -= 1
            end

          #=
            for k in 1:keep
                phase = atan(imag(u[1, k]), real(u[1, k])) + π
                u[:, k] .*= cis(-phase)
                v[:, k] .*= cis(-phase)
            end
            =#
            bases[w][:, (added + 1):(added + keep)] = @view(u[:, 1:keep]) 
            activations[w][:, (added + 1):(added + keep)] = @view(v[:, 1:keep]) * diagm(s[1:keep])
            explainedVar[w][(added + 1):(added + keep)] = s[1:keep] 
            
            added += keep
            
            #=
            for k in 1:keep
                append!(tms, times[1] / fs)
                append!(frs, mean_ifreq(@view(u[:, k]), fs, ar_lags))
                append!(amps, s[k])
            end
            =#
        end
        
       
        
        bases[w] = @view(bases[w][:, 1:added]) 
        activations[w] =  @view(activations[w][:, 1:added]) 
        explainedVar[w] = @view(explainedVar[w][1:added])
        comps[w] = added
        
        
    end

    return bases, activations, comps
    #return explainedVar, bases, [tms frs amps]
end

export recreate_signal
function recreate_signal(bases, activations, comps, wl, ar_lags, threshold, fs)

    sigs = [zeros(ComplexF32, wl, comps[i]) for i in 1:size(bases, 1)]
    ifrs = [zeros(Float32, wl, comps[i]) for i in 1:size(sigs, 1)]
    amps = [zeros(Float32, wl , comps[i]) for i in 1:size(sigs, 1)]
    
    Threads.@threads for b in axes(bases, 1)
        if comps[b] > 0
    
            sig_hankel = zeros(ComplexF32, ar_lags, wl - ar_lags + 1)
  
            for c in 1:comps[b]
                
                counts = zeros(Int32, wl)  

                mul!(sig_hankel, @view(bases[b][:, c]), (@view(activations[b][:, c]))')
                
            
                for j in axes(sig_hankel, 2)
                    for i in axes(sig_hankel, 1)
                        sigs[b][i + j - 1, c] += sig_hankel[i, j]
                        counts[i + j - 1] += 1
                    end
                    
                end

                sigs[b][:, c] ./= counts
             
                
            end

            ifrs[b], amps[b], sigs[b] = window_envelopes(sigs[b], fs, threshold)

        end
    end

    return sigs, ifrs, amps
end


function link_comps(sigs, ifrs, amps, comps, wl, step, ovlp, siglength, similarity_threshold)

    start = findfirst(x -> x > 0, comps)
    strands =  zeros(ComplexF32, siglength, sum(comps))
    strands[1:wl, start:comps[start]] .= sigs[1]
    fstrands =  zeros(Float32, siglength, sum(comps))
    fstrands[1:wl, start:comps[start]] .= ifrs[1]
    astrands =  zeros(Float32, siglength, sum(comps))
    astrands[1:wl, start:comps[start]] .= amps[1]
    added = comps[start]
    active = 1:comps[start]

    startTimes = zeros(Int32, size(fstrands, 2))
    startTimes[1:comps[start]] .= 1
    endTimes = zeros(Int32, size(fstrands, 2)) .+ siglength
   
    for w in (start + 1):size(sigs, 1)
        

        if size(sigs[w], 2) > 0
            d = mag.((norm_cols(@view(strands[((w - 2)*step + wl - ovlp + 1):((w - 2)*step + wl), active])))' * norm_cols(@view(sigs[w][1:(ovlp), :])))
           
            mx = findmax(d)
            
            still_active = zeros(Bool, size(d, 1))
            new_comp = ones(Bool, size(d, 2))
            while mx[1] > similarity_threshold
                
                for t in 1:(wl - step)
                    #=
                    strands[((w - 1)*step + 1):((w - 2)*step + wl), active[mx[2][1]]] ./= 2
                    strands[((w - 1)*step + 1):((w - 2)*step + wl), active[mx[2][1]]] .+= (.5 .* @view(sigs[w][1:step, mx[2][2]]))
                    strands[((w - 2)*step + wl + 1):((w - 1)*step + wl), active[mx[2][1]]] .=  @view(sigs[w][(step + 1):end, mx[2][2]])
                =#

                    tc = (cospi(t / ((wl - step) * 2))) ^ 2
                    strands[((w - 1)*step + t), active[mx[2][1]]] *= tc
                    strands[((w - 1)*step + t), active[mx[2][1]]] += ((1 - tc) * sigs[w][t, mx[2][2]])

                    fstrands[((w - 1)*step + t), active[mx[2][1]]] *= tc
                    fstrands[((w - 1)*step + t), active[mx[2][1]]] += ((1 - tc) * ifrs[w][t, mx[2][2]])

                    astrands[((w - 1)*step + t), active[mx[2][1]]] *= tc
                    astrands[((w - 1)*step + t), active[mx[2][1]]] += ((1 - tc) * amps[w][t, mx[2][2]])
                end 

                strands[((w - 2)*step + wl + 1):((w - 1)*step + wl), active[mx[2][1]]] .=  @view(sigs[w][(wl - step + 1):end, mx[2][2]])
                fstrands[((w - 2)*step + wl + 1):((w - 1)*step + wl), active[mx[2][1]]] .=  @view(ifrs[w][(wl - step + 1):end, mx[2][2]])
                astrands[((w - 2)*step + wl + 1):((w - 1)*step + wl), active[mx[2][1]]] .=  @view(amps[w][(wl - step + 1):end, mx[2][2]])
            
                d[:, mx[2][2]] .= -1
                d[mx[2][1], :] .= -1
            
                still_active[mx[2][1]] = 1
                new_comp[mx[2][2]] = 0

                mx = findmax(d)
            
            end
        end

        endTimes[active[.! still_active]] .= (w)*step + wl
        active = active[still_active]
        
        newComps = (1:comps[w])[new_comp]
        for c in newComps
            added += 1
            startTimes[added] = (w)*step 
            append!(active, added)
            strands[((w - 1)*step + 1):((w - 1)*step + wl), added] .=  @view(sigs[w][:, c])
            fstrands[((w - 1)*step + 1):((w - 1)*step + wl), added] .=  @view(ifrs[w][:, c])
            astrands[((w - 1)*step + 1):((w - 1)*step + wl), added] .=  @view(amps[w][:, c])
            
        end
    
       # return active
    end

    return strands[:, 1:added], fstrands[:, 1:added], astrands[:, 1:added], [startTimes[1:added] endTimes[1:added]]
end

function envelopes(strands, ifrs, amps, fs, threshold, smoothingWindow = 6, swOvlp = 3)
  
    times = (1:size(strands, 1)) ./ fs
    threshold *= maximum(amps)
    #threshold *= mean(amps[amps .> 0])
    Threads.@threads for j in axes(strands, 2)
        
        for i in axes(amps, 1)
            if (amps[i, j] < threshold) | (isnan(ifrs[i, j]))
                ifrs[i, j] = 0
                amps[i, j] = 0
            end
        end

    end
    wgts = copy(ifrs) #weights for taking averages
    wgts[1:(end - 1), :] .-= @view(ifrs[2:end, :]) .+ eps()
    wgts .= abs.(wgts)
    wgts .^= (-1)
    wgts .*= amps

    ifrs = windowAverages(ifrs, wgts, smoothingWindow, swOvlp)
    amps = windowAverages(amps, wgts, smoothingWindow, swOvlp)
    times = windowAverages(times, nothing, smoothingWindow, swOvlp)

    fr = Vector{Float32}()
    am = Vector{Float32}()
    tm = Vector{Float32}()


    for j in axes(ifrs, 2)
        for i in axes(ifrs, 1)
            if amps[i, j] > threshold
                append!(fr, ifrs[i, j])
                append!(am, amps[i, j])
                append!(tm, times[i])
            end
        end
    end

    return [tm fr am]

    

end

function window_envelopes(strands, fs, threshold)
    ifrs = zeros(Float32, size(strands))
    amps = mag.(strands)
  
    tr = threshold * maximum(amps)
    #threshold *= mean(amps[amps .> 0])
    Threads.@threads for j in axes(strands, 2)
        ifrs[:, j] .= ifreq(@view(strands[:, j]), fs)
       
       
        for i in axes(amps, 1)
            if (amps[i, j] < tr) | (isnan(ifrs[i, j]))
                ifrs[i, j] = 0
                amps[i, j] = 0
            end
        end

    end

    #=
    ifrs = windowAverages(ifrs, smoothingWindow, swOvlp)
    amps = windowAverages(amps, smoothingWindow, swOvlp)
    
    times = windowAverages(strands, smoothingWindow, swOvlp)
=#

    return ifrs, amps, strands

    

end




export freqStrand_tv_smoothed


function freqStrand_tv_smoothed(freqs, amps, tfd, cfs, elasticity, maxiter = 1000)
    
    #forces =  DSP.conv(permutedims(tfd), [-.5; 0; -.5])[2:(end - 1), :]
    fRes = cfs[2] - cfs[1]
    minFreq = cfs[1]
    for j in axes(freqs, 2)
        println(j)
        startTime = findfirst(x -> x > 0, @view(amps[:, j])) + 1
        endTime = findfirst(x -> x < .0000001, @view(amps[(startTime + 1):end, j])) + startTime 
       
    
        if (endTime - startTime) > 10
            #freqDiffs = runmean(pushfirst!(diff(@view(freqs[startTime:endTime])), 0), 2)
            stretch = DSP.conv(freqs[startTime:endTime, j], [.50f0, -1.0f0, .50f0])[2:(end - 1)]
            stretch[1] += 0.5*freqs[startTime, j]
            stretch[end] += 0.5*freqs[end - 1, j] 

            peak = argmax(stretch[2:(end - 1)]) + 1
            moved = true
            peakChanged = true
            iters = 0
            while (moved || peakChanged) && (iters <= maxiter)
                iters += 1
                println(peak)
                current_spec_energy = interpolate_freqbins(hzToErb(freqs[peak + startTime, j]), peak, tfd, fRes, minFreq)
                proposed_spec_energy = interpolate_freqbins(hzToErb(freqs[peak + startTime, j] - stretch[peak]), peak, tfd, fRes, minFreq)
            
                if (abs(stretch[peak + 1]) * elasticity) > (proposed_spec_energy - current_spec_energy)
                    if peak > 2 
                        if peak < (size(stretch, 1) - 1)
                            freqs[peak + startTime, j] -= stretch[peak]
                            stretch[peak - 1] = mean([freqs[peak + startTime - 2, j], freqs[peak + startTime, j] ]) - freqs[peak + startTime - 1, j]
                            stretch[peak + 1] = mean([freqs[peak + startTime, j], freqs[peak + startTime + 3, j] ]) - freqs[peak + startTime + 1, j]
                    
                            peak = argmax([stretch[peak - 1], stretch[peak + 1]])
                        else
                            peak =  (size(stretch, 1) - 1)
                        end
                    else
                        peak = 2
                    end
                       

                    moved = true
                    
                else
                    moved = false
                    newPeak = argmax(stretch[2:(end - 1)]) + 1
                    peakChanged = newPeak != peak
                    peak = newPeak
                end
            end
        end
    end
     
    return fRes
end

function hzToErb(f)
    erb = 21.4 * log10((4.37 * f / 1000) + 1)
    return erb
end

function interpolate_freqbins(freq, time, tfd, fRes, minFreq)
    fBin = ((freq - minFreq) / fRes) + 1
    if fBin < 1
        fUp = 2
        fDown = 1
    else
        fUp = Int(ceil(fBin))
        fDown = Int(floor(fBin))
    end
    (tfd[fUp, time] - tfd[fDown, time]) * fBin

end
############ NLMS and trajectory smoothing stuff
export init_AR, plan_AR, tracker_AR, tracker_AR_complex, update_NLMS!, spectra_from_AR
 



function update_NLMS!(ar, plan, xNext, t; fs, fmin, fmax, fstep, threshold)
    append!(ar.predictions, dot(ar.h, ar.x))
    append!(ar.errors, xNext-ar.predictions[end])
    ar.V *= plan.gamma
    ar.V += (1 - plan.gamma)* real(ar.x[end] * conj(ar.x[end]))#(ar.x[end]^2)
    ar.h .+= ((plan.alpha/(ar.V + plan.beta))*(ar.errors[end])).*ar.x
    popfirst!(ar.x)
    #popfirst!(ar.predictions)
    #popfirst!(ar.errors)
    append!(ar.x, plan.sigma*ar.predictions[end] + (1 - plan.sigma)*xNext)

    ar.past_h[:, t] .= ar.h


end

function init_AR(plan, h = zeros(Float32, plan.p), x = zeros(Float32, plan.p); start = "linear", signal_length)
    if start == "linear"
        h[end - 1] = -1 * plan.sigma
        h[end] = 2 * plan.sigma
    end

    past_h = zeros(eltype(h), plan.p, signal_length)
    frequencies = zeros(Float32, plan.frs, signal_length)
    ar = tracker_AR(0, h, x, [0], [0], past_h, frequencies)
end

function spectra_from_AR(sigma, ar, fs, fmin, fmax, fstep)
    freqs = fmin:fstep:fmax
    amps = ones(ComplexF64, length(freqs))

    Threads.@threads for i in axes(amps, 1)
        for k in axes(ar, 1)
            amps[i] -= conj(ar[length(ar) - k + 1]) * cispi(-2 * k * freqs[i] / fs)
        end
        amps[i] *= conj(amps[i])
        amps[i] = sigma / amps[i]

    end

    return freqs, real.(amps)

end


mutable struct tracker_AR
    V::Float64
    h::Vector{Float64}
    x::Vector{Float64}
    predictions::Vector{Float64}
    errors::Vector{Float64}

    past_h::Matrix{Float64}
    frequencies::Matrix{Float64}
end

mutable struct tracker_AR_complex
    V::Float64
    h::Vector{ComplexF64}
    x::Vector{ComplexF64}
    predictions::Vector{ComplexF64}
    errors::Vector{ComplexF64}

    past_h::Matrix{ComplexF64}
    frequencies::Matrix{Float64}
end








end




####### Total variation denoising to obtain smooth frequency tracks
#This code is entirely copied from here: https://github.com/fundamental/TotalVariation.jl/blob/master/src/TotalVariation.jl
#It has been pasted in due to an unresolvable package version conflict.

module TotalVariation

using  DSP, SparseArrays, LinearAlgebra, RollingFunctions
export gstv, tv
#See ``Total Variation Denoising With Overlapping Group Sparsity'' by
# Ivan Selesnick and Po-Yu Chen (2013)

#Group Sparse Total Variation Denoising
function gstv(y::Vector, k, λ; show_cost = false, iter = 10)
    #Initialize Solution
    N  = length(y)
    x  = copy(y)

    #Differential of input
    b = diff(y)

    #Precalculate D D' where D is the first-order difference matrix
    DD::SparseMatrixCSC{Float64,Int} = spdiagm(-1=>-ones(N-2), 0=>2*ones(N-1), 1=>-ones(N-2))
  
   
    #Convolution Mask - spreads D x over a larger area
    #This regularizes the problem with applying a gradient to a larger area.
    #at k=1 the normal total variational sparse solution (for D x) is found.
   
    u = similar(b)
    r = similar(b)
    Λ = similar(b)
    F = sparse(Diagonal(1 ./ Λ))/λ + DD
    

    tmp = similar(b)
    dfb = diff(tmp)

    Threads.@threads for i=1:iter
        u = diff(x)
        r = sqrt.(max.(eps(), runmean(u, k))) 
        Λ = runmean(1 ./ r, k)
        F = sparse(Diagonal(1 ./ Λ))/λ + DD
        

        tmp = F\b
        dfb = diff(tmp)

        x[1]       = y[1]       + tmp[1]
        x[2:end-1] = y[2:end-1] + dfb[:] 
        x[end]     = y[end]     - tmp[end]
    end

    return x
end

export gstv_multi
function gstv_multi(y::Matrix, k, λ; show_cost = false, iter = 10, batchsize = size(y, 1))
    #Initialize Solution
   
    x  = copy(y)

    
    N  = batchsize
    batches = div(size(y, 1), batchsize)
    #Precalculate D D' where D is the first-order difference matrix
    DD::SparseMatrixCSC{Float64,Int} = spdiagm(-1=>-ones(N-2), 0=>2*ones(N-1), 1=>-ones(N-2))


    for batch in 1:batches
        Threads.@threads for j in axes(y, 2)
      
            #Differential of input
            b = diff(@view(y[((batch - 1)*batchsize + 1):((batch)*batchsize), j]))

            #=
            u = similar(b)
            r = similar(b)
            Λ = similar(b)
            F = sparse(Diagonal(1 ./ Λ))/λ + DD
            

            tmp = similar(b)
            dfb = diff(tmp)
                =#
            
            for i in 1:iter
                u = diff(@view(x[((batch - 1)*batchsize + 1):(batch*batchsize), j]))
                r = sqrt.(max.(eps(), runmean(u, k))) 
                Λ = runmean(1 ./ r, k)
                F = sparse(Diagonal(1 ./ Λ)) ./ λ .+ DD
                

                tmp = F\b
                dfb = diff(tmp)

                x[((batch - 1)*batchsize + 1), j] = y[((batch - 1)*batchsize + 1), j] + tmp[1]
                x[((batch - 1)*batchsize + 2):((batch*batchsize) - 1), j] .= @view(y[((batch - 1)*batchsize + 2):((batch*batchsize) - 1), j]) .+ dfb
                x[(batch*batchsize), j] = y[(batch*batchsize), j] - tmp[end]
            end
        end
    end

    return x
end

end # module