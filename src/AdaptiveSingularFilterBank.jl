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
using DSP, StatsBase, LinearAlgebra, Peaks, RollingFunctions, LowRankApprox, Revise

include("./TimeFrequencyTools.jl")
using .TimeFrequencyTools
#import ..TimeFrequencyTools.mag



########## sliding window SSA with frequency channel filtering
export ssa_windowed_peakchannels, ssa_windowed, mssa_windowed, link_comps, 
    envelopes, update_hankel!, link_comps_sparse, params_longformat, trackchannels, track_crosschannels

#=
#Uses phase derivative to estimate instantaneous frequency of a sampled complex exponential. 
#No phase unwrapping is required.
function ifreqn(y::Vector{ComplexF32}, fs)
    freqs = zeros(Float32, size(y, 1))
    
    Threads.@threads for i in 2:(size(y, 1) - 1)
        freqs[i] = fs * imag(conj(y[i]) * .5 * (y[i + 1] - y[i - 1])) / (π*2) #* fs / ( 2 * π )
        freqs[i] = (freqs[i] < 0) || (isnan(freqs[i])) ? 0 : freqs[i]
    end

    freqs[1] = freqs[2]
    freqs[end] = freqs[end - 1]

    return freqs
end

function ifreq(x::Vector{T}, fs) where T >: Complex
    y = x ./ mag.(x)
    freqs = zeros(Float32, size(y, 1))
    
    Threads.@threads for i in 2:(size(y, 1) - 1)
        freqs[i] = fs * imag(conj(y[i]) * .5 * (y[i + 1] - y[i - 1])) / (π*2) #* fs / ( 2 * π )
        freqs[i] = (freqs[i] < 0) || (isnan(freqs[i])) ? 0 : freqs[i]
    end

    freqs[1] = freqs[2]
    freqs[end] = freqs[end - 1]

    return freqs
end

function ifreq_sample(y, fs)
    fs * imag(conj(y[2]) * .5 * (y[3] - y[1])) / (π*2) #* fs / ( 2 * π )
end

function ifreq(x::Matrix{T}, fs) where T >: Complex
    y = x ./ (mag.(x) .+ eps())
    freqs = zeros(Float32, size(y))
    
    Threads.@threads for j in axes(x, 2)
        for i in 2:(size(y, 1) - 1)
            freqs[i, j] = fs * imag(conj(y[i, j]) * .5 * (y[i + 1, j] - y[i - 1, j])) / (π*2) #* fs / ( 2 * π )
            freqs[i, j] = (freqs[i, j] < 0) || (isnan(freqs[i, j])) ? 0 : freqs[i, j]
        end
    end

    freqs[1, :] .= @view(freqs[2, :])
    freqs[end, :] .= @view(freqs[end - 1, :])
    #replace!(x -> x < 0 )
    return freqs
end



function ifreq(x::AbstractArray, fs)
    y = x ./ mag.(x)
    freqs = zeros(Float32, size(y, 1))
    
    Threads.@threads for i in 2:(size(y, 1) - 1)
        freqs[i] = fs * imag(conj(y[i]) * .5 * (y[i + 1] - y[i - 1])) / (π*2) #* fs / ( 2 * π )
        freqs[i] = (freqs[i] < 0) || (isnan(freqs[i])) ? 0 : freqs[i]
    end

    freqs[1] = freqs[2]
    freqs[end] = freqs[end - 1]

    return freqs
end

#magnitude of complex value
function mag(a)
    sqrt(real(a)^2 + imag(a)^2)
end

function mag2(a)
    real(a)^2 + imag(a)^2
end

function threshold_power(val, threshold)
    mag(val) < threshold ? 0 : val
end



function windowAverages(aud, wgt, wl, ovlp)
    
    channels = zeros(eltype(aud), length(arraysplit(collect(1:size(aud, 1)), wl, ovlp)), size(aud, 2))

    if isnothing(wgt)
        Threads.@threads for j in axes(channels, 2)
            window = arraysplit(1:size(aud, 1), wl, ovlp)
    
            for (i, wind) in enumerate(window)
                channels[i, j] = mean(@view(aud[Int.(wind), j]))

                if isnan(channels[i, j])
                    channels[i, j] = 0
                end
            end
        end
    else

        Threads.@threads for j in axes(channels, 2)
            window = arraysplit(1:size(aud, 1), wl, ovlp)
    
            for (i, wind) in enumerate(window)
                channels[i, j] = mean(@view(aud[Int.(wind), j]), weights(@view(wgt[Int.(wind), j])))
                if isnan(channels[i, j])
                    channels[i, j] = 0
                end
            end
        end
    end
    
   
    return channels
end


function windowAverages(aud::Vector, wgt, wl, ovlp, mode = "mean")
    
    window = arraysplit(1:size(aud, 1), wl, ovlp)
    channels = zeros(eltype(aud), length(window))
  
     if isnothing(wgt)
          if mode == "mean"
             for (i, wind) in enumerate(window)
 
                 channels[i] = mean(@view(aud[Int.(wind)]))
 
                 if isnan(channels[i])
                     channels[i] = 0
                 end
             end
         elseif mode == "median"
             for (i, wind) in enumerate(window)
 
                 channels[i] = median(@view(aud[Int.(wind)]))
 
                 if isnan(channels[i])
                     channels[i] = 0
                 end
             end
         end
     else
 
         if mode == "mean"
             for (i, wind) in enumerate(window)
                 channels[i] = mean(@view(aud[Int.(wind)]), weights(@view(wgt[Int.(wind)])))
                 if isnan(channels[i])
                     channels[i] = 0
                 end
             end
         elseif mode == "median"
             for (i, wind) in enumerate(window)
                 channels[i] = median(@view(aud[Int.(wind)]), weights(@view(wgt[Int.(wind)])))
                 if isnan(channels[i])
                     channels[i] = 0
                 end
             end
         end
     end
     
    
     return channels
 end

 
=#
function mag2(a)
    real(a)^2 + imag(a)^2
end


function update_hankel!(hk, xk, embeddingDim, wl)
   
   # hk = zeros(ComplexF32, embeddingDim, wl - embeddingDim + 1)
    for j in 1:(wl - embeddingDim + 1)
        hk[:, j] .= @view(xk[j:(j + embeddingDim - 1)])
    end

    hk .-= mean(hk, dims = 2)
end
function update_hankel_mc!(hk, xk, embeddingDim, wl)
   
    for ch in axes(xk, 2)
        for j in (1:(wl - embeddingDim + 1)) .+ ((ch - 1) * (wl - embeddingDim))
            hk[:, j] .= @view(xk[j:(j + embeddingDim - 1), ch])
        end
 
        hk .-= mean(hk, dims = 2)
    end
end

 
function ssa_windowed_peakchannels(tfd; ri, wl, embeddingDim, step, components, channel_components, min_eigval)
    
 
    s = zeros(eltype(tfd), size(tfd, 2))
    
    nWindows = size(ri, 2) - 1
    
    bases = [zeros(ComplexF32, embeddingDim, components) for i in 1:nWindows]
    activations = [zeros(ComplexF32, wl - embeddingDim + 1, components) for i in 1:nWindows]
    comps = zeros(Int32, nWindows)

    explainedVar = [zeros(Float32, components) for i in 1:nWindows]
    hankels = [zeros(ComplexF32, embeddingDim, wl - embeddingDim + 1) for i in axes(tfd, 1)]
   
    maxeig = 0
    Threads.@threads for w in 1:nWindows
        if w == nWindows
           wl = size(tfd, 2) - ((w - 1)*step + 1)
           ri[:, w] .= 0.5 .* (ri[:, w] + ri[:, w + 1])
        end

        times = ((w - 1)*step + 1):((w - 1)*step + wl)
        
     
        println(w)
        
        peaks = findmaxima(@view(ri[:, w]))
        bases[w] = zeros(ComplexF32, embeddingDim, channel_components * length(peaks[1])) 
        explainedVar[w] = zeros(ComplexF32, channel_components * length(peaks[1])) 
        added = 0
        while !isempty(peaks[1])
            
            p = pop!(peaks[1])
            update_hankel!(hankels[p], @view(tfd[p, times]), embeddingDim, wl)
   

    
            u, s, v  = psvd(hankels[p], rank = channel_components, sketch = :sub, maxdet_niter = 100000, maxdet_tol = 5 * eps(), sketchfact_adap = true)
            if s[1] > maxeig
                maxeig = s[1]
            end
            keep = findfirst(x -> x < (min_eigval .* maxeig), s) 
            if isnothing(keep)
                keep = length(s)
            else
                keep -= 1
            end

            bases[w][:, (added + 1):(added + keep)] = @view(u[:, 1:keep]) 
            activations[w][:, (added + 1):(added + keep)] = @view(v[:, 1:keep]) 
            explainedVar[w][(added + 1):(added + keep)] = s[1:keep] 
            
            added += keep
            
         
        end
        
       
        
        bases[w] = @view(bases[w][:, 1:added]) 
        activations[w] =  @view(activations[w][:, 1:added])
        explainedVar[w] = @view(explainedVar[w][1:added])
        comps[w] = added
        
        
    end

    return bases, activations, explainedVar, comps
    #return explainedVar, bases, [tms frs amps]
end

function ssa_windowed(song; wl, embeddingDim, step, k, components, min_eigval, similarity_threshold)
    

    nWindows = Int(floor(((length(song)) - wl) / step))   #number of local SSA windows, i.e., the number of times SSA is performed
    coefs = [1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1] #used for convolution method for hankelization of u, s, and v 
    removed = Int(round((wl - step) / 2)) #number of elements truncated off each window's reconstructed components for the overlap-save concatenation
    
    
    #sigs = [zeros(ComplexF32, wl, components) for i in 1:nWindows]  #preallocating the output, i.e., the reconstructed segments from each eigenvector triplet
    sigs = Vector{Vector{ComplexF32}}() #initialize the output, i.e., the reconstructed segments from each eigenvector triplet
    hankel = zeros(ComplexF32, embeddingDim, wl - embeddingDim + 1)  #preallocating the actual sequence's hankel trajectory matrix
    starts = Vector{Int64}() #window start times

    old_u = zeros(eltype(song), embeddingDim, components) #initialize matrix of previous window's eigenvectors
    unused = collect(1:components) #columns of old_u that are inactive and can be overwritten 
    paired = Vector{Int64}()
    d = zeros(Float32, 2, 2) #initialize distance matrix of adjacent eigenvectors
    mx = 0 #max covariance between eigenvectors of adjacent windows, used to link components
    mapping = Dict{Int32, Int32}() #keys are rows of d, values are indices of sigs. Used to concatenate reconstructed components of window w to a pre-existing element of sigs if their basis vectors (u) are correlated enough
    added = 0 #cumulative number of individual reconstructed components added to sigs
    
    
    for w in 1:nWindows
        println(w)
        t1 = ((w - 1)*step + 1) #start time of the window
      
        if w == nWindows
            
            wl = length(song) - ((w - 1)*step + 1) #enlarge the last window if nWindows doesnt evenly divide the signal length
            times = t1:(t1 - 1 + wl)
            #removed = Int(round((wl - step) / 2)) #number of elements truncated off each window's reconstructed components for the overlap-save concatenation
    
    
            #resizing variables to handle increased window length
            hankel = zeros(ComplexF32, embeddingDim, wl - embeddingDim + 1)  
            coefs = [1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1]
            
        end
        times = t1:(t1 - 1 + wl)
        #startTimes[w] = t1

        update_hankel!(hankel, @view(song[times]), embeddingDim, wl)
        u, s, v  = psvd(hankel, rank = k, sketch = :sub, maxdet_niter = 100000, maxdet_tol = 5 * eps(), sketchfact_adap = true)
        keep = findfirst(x -> x < (min_eigval .* s[1]), s) 
        keep = isnothing(keep) ? length(s) : (keep - 1)
        
        d = mag.(old_u' * @view(u[:, 1:keep]))
      
        for c in 1:keep
            mx = findmax(@view(d[:, c]))

            if (mx[1] > similarity_threshold) && (mx[1] >= findmax(@view(d[mx[2], :]))[1])
                append!(sigs[mapping[mx[2]]], 
                    hankelize(@view(u[:, c]), s[c], @view(v[:, c]), coefs, removed + 1, removed + step))
                old_u[:, mx[2]] .= @view(u[:, c])
                append!(paired, mx[2])
            else
                added += 1
                push!(sigs, hankelize(@view(u[:, c]), s[c], @view(v[:, c]), coefs, 1, removed + step))
                col = pop!(unused)
                old_u[:, col] .= @view(u[:, c])
                mapping[col] = added
                append!(starts, [t1])
            end
            
            #    sigs[w][:, c] .= s[c] * (coefs .* DSP.conv(conj.(@view(v[:, c])), @view(u[:, c])))
        end
        
        unused = filter(x -> x ∉ paired, 1:components)
       # sigs[w] = @view(sigs[w][:, 1:keep]) 
        

    end

    return sigs, starts
end

function mssa_windowed(tfd; wl, embeddingDim, step, k, components, min_eigval, similarity_threshold)
    

    nWindows = Int(floor(((size(tfd, 1)) - wl) / step))   #number of local SSA windows, i.e., the number of times SSA is performed
    coefs = repeat([1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1], size(tfd, 2)) #used for convolution method for hankelization of u, s, and v 
    removed = Int(round((wl - step) / 2)) #number of elements truncated off each window's reconstructed components for the overlap-save concatenation
    
    
    #sigs = [zeros(ComplexF32, wl, components) for i in 1:nWindows]  #preallocating the output, i.e., the reconstructed segments from each eigenvector triplet
    sigs = Vector{Vector{ComplexF32}}() #initialize the output, i.e., the reconstructed segments from each eigenvector triplet
    hankel = zeros(ComplexF32, embeddingDim, (wl - embeddingDim + 1)*size(tfd, 2))  #preallocating the actual sequence's hankel trajectory matrix
    starts = Vector{Int64}() #window start times

    old_u = zeros(eltype(tfd), embeddingDim, components) #initialize matrix of previous window's eigenvectors
    unused = collect(1:components) #columns of old_u that are inactive and can be overwritten 
    paired = Vector{Int64}()
    d = zeros(Float32, 2, 2) #initialize distance matrix of adjacent eigenvectors
    mx = 0 #max covariance between eigenvectors of adjacent windows, used to link components
    mapping = Dict{Int32, Int32}() #keys are rows of d, values are indices of sigs. Used to concatenate reconstructed components of window w to a pre-existing element of sigs if their basis vectors (u) are correlated enough
    added = 0 #cumulative number of individual reconstructed components added to sigs
    
    
    for w in 1:nWindows
        println(w)
        t1 = ((w - 1)*step + 1) #start time of the window
      
        if w == nWindows
            
            wl = size(tfd, 1) - ((w - 1)*step + 1) #enlarge the last window if nWindows doesnt evenly divide the signal length
            times = t1:(t1 - 1 + wl)
            #removed = Int(round((wl - step) / 2)) #number of elements truncated off each window's reconstructed components for the overlap-save concatenation
    
    
            #resizing variables to handle increased window length
            hankel = zeros(ComplexF32, embeddingDim, (wl - embeddingDim + 1) * size(tfd, 2)) 
            coefs = repeat([1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1], size(tfd, 2))
            
        end
        times = t1:(t1 - 1 + wl)
        #startTimes[w] = t1

        update_hankel_mc!(hankel, @view(tfd[times, :]), embeddingDim, wl)
        u, s, v  = psvd(hankel, rank = k, sketch = :sub, maxdet_niter = 100000, maxdet_tol = 5 * eps(), sketchfact_adap = true)
        keep = findfirst(x -> x < (min_eigval .* s[1]), s) 
        keep = isnothing(keep) ? length(s) : (keep - 1)
        
        d = mag.(old_u' * @view(u[:, 1:keep]))
      
        for c in 1:keep
            mx = findmax(@view(d[:, c]))

            if (mx[1] > similarity_threshold) && (mx[1] >= findmax(@view(d[mx[2], :]))[1])
                append!(sigs[mapping[mx[2]]], 
                    hankelize(@view(u[:, c]), s[c], @view(v[:, c]), coefs, removed + 1, removed + step))
                old_u[:, mx[2]] .= @view(u[:, c])
                append!(paired, mx[2])
            else
                added += 1
                push!(sigs, hankelize(@view(u[:, c]), s[c], @view(v[:, c]), coefs, 1, removed + step))
                col = pop!(unused)
                old_u[:, col] .= @view(u[:, c])
                mapping[col] = added
                append!(starts, [t1])
            end
            
            #    sigs[w][:, c] .= s[c] * (coefs .* DSP.conv(conj.(@view(v[:, c])), @view(u[:, c])))
        end
        unused = filter(x -> x ∉ paired, 1:components)
       # sigs[w] = @view(sigs[w][:, 1:keep]) 
        

    end

    return sigs, starts
end

#=
function ssa_windowed_unlinked(song; wl, embeddingDim, step, components, min_eigval)
    

    nWindows = Int(floor(((length(song)) - wl) / step))   #number of local SSA windows, i.e., the number of times SSA is performed
    coefs = [1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1] #used for convolution method for hankelization of u, s, and v 
    removed = Int(round((wl - step) / 2)) #number of elements truncated off each window's reconstructed components for the overlap-save concatenation
    
    
    sigs = [zeros(ComplexF32, step, components) for i in 1:nWindows]  #preallocating the output, i.e., the reconstructed segments from each eigenvector triplet
    hankels = [zeros(ComplexF32, embeddingDim, wl - embeddingDim + 1) for i in 1:nWindows]  #preallocating the actual sequence's hankel trajectory matrix
    
    Threads.@threads for w in 1:nWindows
        println(w)
        t1 = ((w - 1)*step + 1) #start time of the window
      
        if w == nWindows
            
            wl = length(song) - ((w - 1)*step + 1) #enlarge the last window if nWindows doesnt evenly divide the signal length
            times = t1:(t1 - 1 + wl)
            #removed = Int(round((wl - step) / 2)) #number of elements truncated off each window's reconstructed components for the overlap-save concatenation
    
    
            #resizing variables to handle increased window length
            hankels[w] = zeros(ComplexF32, embeddingDim, wl - embeddingDim + 1)  
            coefs = [1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1]
            
        end
        times = t1:(t1 - 1 + wl)
        #startTimes[w] = t1

        update_hankel!(hankels[w], song[times], embeddingDim, wl)
        u, s, v  = psvd(hankels[w], rank = components, sketch = :sub, maxdet_niter = 100000, maxdet_tol = 5 * eps(), sketchfact_adap = true)
        keep = findfirst(x -> x < (min_eigval .* s[1]), s) 
        keep = isnothing(keep) ? length(s) : (keep - 1)
        
     
      
        for c in 1:keep
            sigs[w][:, c] .= hankelize(@view(u[:, c]), s[c], @view(v[:, c]), coefs, removed + 1, removed + step) 
        end
        
        sigs[w] = @view(sigs[w][:, 1:keep]) 
        

    end

    return sigs
end
=#

function hankelize(u, s, v, coefs, a, b)
   # coefs = [1:embeddingDim ; repeat([embeddingDim], wl - (2 * embeddingDim)); embeddingDim:-1:1]
    s .* @view(coefs[a:b]) .* @view(DSP.conv(conj.(v), u)[a:b])
end
        
#implements the hankelize method described in Korobeynikov, Anton. "Computation- and Space-Efficient Implementation of SSA."
function hankelize_all(bases, activations, explainedVar, comps, wl, embeddingDim)

    sigs = [zeros(ComplexF32, wl, comps[i]) for i in 1:size(bases, 1)]
   

    Threads.@threads for b in axes(bases, 1)
        if comps[b] > 0
            for c in 1:comps[b]
                sigs[b][:, c] .= explainedVar[b][c] * (coefs .* DSP.conv(conj.(@view(activations[b][:, c])), @view(bases[b][:, c])))
            end
        end
    end

    return sigs#, ifrs, amps
end



function params_longformat(sigs, times, fs, smoothingWl, smoothingOvlp)
  
    fr = Vector{Float32}()
    am = Vector{Float32}()
    tm = Vector{Float32}()

    for j in axes(sigs, 1)
        append!(fr, windowAverages(ifreq(sigs[j], fs), nothing, smoothingWl, smoothingOvlp, "median"))
        append!(am, windowAverages(mag.(sigs[j]), nothing, smoothingWl, smoothingOvlp, "median"))
        append!(tm, windowAverages(collect(times[j]:(times[j] + length(sigs[j]))) ./ fs, nothing, smoothingWl, smoothingOvlp, "median"))
    end
    am ./= maximum(am)
    return  [tm fr am]
end









############ NLMS and trajectory smoothing stuff

#export trackchannels, track_crosschannels


#Normalized least mean squares, one of the most popular adaptive filters
function update_NMLS!(predictions, errors, coefs, x, v, mu, e)

    #make prediction
    predictions[end] = dot(coefs, @view(x[1:(end - 1)]))

    #calculate error
    errors[end] = (x[end] - predictions[end])
   
    #adjust coefficients according to instantaneous gradient of mean square error
    coefs .+= ((mu * conj(errors[end])) / (v + e)) .* @view(x[1:(end - 1)])
   


end

function update_NMLS_multi!(predictions, errors, coefs, crossCoefs, x, v, ts, te, plan)
    
   #make prediction
   predictions[te + 1] = dot(crossCoefs, @view(x[ts:te]))

   #calculate error
   errors[te + 1] = (x[te + 1] - predictions[te + 1])
  
   #adjust coefficients according to instantaneous gradient of mean square error
   crossCoefs .+= ((plan.mu * conj(errors[te + 1])) / (v + plan.e) .* @view(x[ts:te]))
  
end





function trackchannels(tfd; mu, e, p, minAmplitude, ds = ds, saveEvery = 100);
    
    tfd = @view(tfd[1:ds:end, :])
    nWindows = Int(floor((size(tfd, 1) - p )/ saveEvery)) 
  
    power = mag2.(tfd)
    predictions = similar(tfd)
    errors = similar(tfd)
   
    Threads.@threads for k in axes(power, 2)
        power[:, k] .= (runmean(@view(power[:, k]), p) .* p)
    end

    threshold = maximum(power) * minAmplitude
    coefs = zeros(ComplexF32, p, size(tfd, 2))

    for w in 1:(nWindows - 1) #saveEvery:(size(tfd, 1) - plan.p)
        Threads.@threads for k in axes(tfd, 2)
            for t in ((w - 1) * saveEvery + 1):(w * saveEvery)
                if power[t, k] > threshold
                    update_NMLS!(@view(predictions[t:(t + p), k]), @view(errors[t:(t + p), k]), @view(coefs[:, k]),  @view(tfd[t:(t + p), k]), power[t + p - 1, k], mu, e)
                end

            end

            
        end
        #println(t)
    end
   
    return predictions, errors #, allCoefs
         
end

#export track_crosschannels
function track_crosschannels(tfd; mu, e, p, minPrediction, minAmplitude, ds = ds, saveEvery = 100);
    
    tfd = @view(tfd[1:ds:end, :])
    nWindows = Int(floor((size(tfd, 1) - p )/ saveEvery)) 
    power = mag2.(tfd)
    
    coefs = zeros(eltype(tfd), size(tfd, 2), p)

    #=
    coefs = Dict{Int, Vector{eltype(tfd)}}()
    whichChannels = Dict{Int, Vector{Int}}()
    whichCoefs = Dict{Int, Vector{Int}}()
    =#
    channelIsActive = zeros(Bool, size(tfd, 2))
    coefIsActive = zeros(Bool, size(tfd, 2))
    coefCov =  zeros(eltype(tfd), size(tfd, 2), size(tfd, 2))
    coefChannelCov =  zeros(eltype(tfd), size(tfd, 2), size(tfd, 2))

    Threads.@threads for k in axes(power, 2)
        power[:, k] .= (runmean(@view(power[:, k]), p))
    end

    tfd = permutedims(tfd)
    power = permutedims(power)
    signalComps = similar(tfd)
    predictions = similar(tfd)
    errors = copy(tfd)

    ampThreshold = maximum(power) * sqrt(minAmplitude)

    for t in 1:(size(tfd, 2) - p)
        salientChannels = findall(x -> x > ampThreshold, @view(tfd[k, t + p]))
        for k in salientChannels
            if channelIsActive[k]
                    
                if length(whichCoefs[k]) > 1
                    preds = @view(coefs[whichCoefs[k], :]) \  @view(tfd[k, t:(t + p - 1)]) #(conj.(@view(coefs[whichCoefs[k], :])) * @view(tfd[k, t:(t + p - 1)]))
                    signalComps[whichCoefs[k], t + p] .+= preds
                    predictions[k, t + p] = sum(preds)
                    errors[k, t + p] -= predictions[k, t + p]
    

                else
                    signalComps[c, t + p] += (conj.(@view(coefs[c, :])) * @view(tfd[k,  t:(t + p - 1)]))
                    predictions[k, t + p] = sum(preds)
                    errors[k, t + p] -= predictions[k, t + p]
        
                
                end

            else
                channelIsActive[k] = true

            end


        for c in eachindex(coefIsActive)
            for k in whichChannels[c]
                if length(whichCoefs[k]) > 1
                    
                    preds = @view(coefs[whichCoefs[k], :]) \  @view(tfd[k, (t - p):(t + p)]) #(conj.(@view(coefs[whichCoefs[k], :])) * @view(tfd[k, t:(t + p - 1)]))
                    signalComps[whichCoefs[k], t] .+= preds
                    predictions[k, t] = sum(preds)
                    errors[k, t] -= predictions[k, t]
    

                else
                    signalComps[c, t] += (conj.(@view(coefs[c, :])) * @view(tfd[k, (t - p):(t + p)]))
                    predictions[k, t] = sum(preds)
                    errors[k, t] -= predictions[k, t]
        
                
                end
               
            end
        end
    end
        #= 
        predictions_unrefined .= coefs * @view(tfd[t:(t + p - 1), k])
        residuals_unrefined .=  @view(tfd[t + p, :])
        channelMag .= mag.(residuals_unrefined)
        total = sum(channelMag)
        threshold = total * minAmplitude
        
        while total > threshold
            m = argmax(channelMag)
            b = argmin(@view(predictions_unrefined[:, m]))
            included[b] = true
            residuals_unrefined .-= @view(predictions_unrefined[b, :])
            channelMag .= mag.(residuals_unrefined)
            total = sum(channelMag)
        end
        
        
        for i in eachindex(included)
            if !included
                predictions_unrefined[i, :] .= 0
                #update_NMLS!(@view(predictions[:, k]), @view(errors[:, k]), @view(coefs[:, k]),  @view(tfd[:, k]), power[t + plan.p - 1, k], t, t + plan.p - 1, plan)
            end
        end

        Threads.@threads for k in axes(crossCoefs, 2)
            channelCrossCov = predictions_unrefined' * predictions_unrefined
            if included
                update_NMLS!(@view(predictions[:, k]), @view(errors[:, k]), @view(coefs[:, k]),  @view(tfd[:, k]), power[t + plan.p - 1, k], t, t + plan.p - 1, plan)
            else

            end
            predictions_unrefined[i, :] .= 0
            update_NMLS_multi!(@view(predictions[:, k]), @view(errors[:, k]), @view(coefs[:, k]), @view(crossCoefs[:, k]),  @view(tfd[:, k]), power[t + plan.p - 1, k], t, t + plan.p - 1, plan)
            
            
        end
    end
        
    
    


    return predictions, errors, coefs
=#
end

#export dft_matrix, ar_spectrum
function dft_matrix(fbins, ncoef, fs)

    m = zeros(ComplexF32, length(fbins), ncoef)

    for j in axes(m, 2)
        for i in axes(m, 1)
            m[i, j] = cispi(-2 * j * fbins[end - i + 1] / fs)
        end
    end
    m
end

function ar_spectrum(coefs, dftmat)

    spec = (1 .+ (dftmat * coefs))
    spec .= real.(spec) .^ 2 .+ imag.(spec) .^ 2

    return 1 ./ real.(spec)
end

mutable struct NLMSParam
    mu::Float64
    e::Float64
    p::Int64
    energySmoothing::Float64
    minAmplitude::Float64
end

end

#### component extraction
#=
export track_components


function track_components(tfd, fs; timeres, minLength, minAmplitude, burnin)

   
    wl = Int(round(timeres * fs))
    nWindows = Int(floor(size(tfd, 1) / wl))
    times = 1:wl
    ml = Int(round(minLength * fs))
    amps = mag.(tfd)
   
    tStarts = zeros(Int64, size(tfd, 2))
    strands = Dict{Tuple{Int64, Int64}, Matrix{Float32}}()

    inactive = [t*-rue for i in axes(tfd, 2)]
    #newlyActive = [false for i in axes(tfd, 2)]
    #newlyEnded = [false for i in axes(tfd, 2)]
   
    for w in 1:nWindows
        times = ((w - 1)*wl + 1):(w * wl)
    
        meanAmps = mean(@view(amps[times, :]), dims = 1)
        threshold = maximum(meanAmps) * minAmplitude
      
        println(w)
        for k in axes(tfd, 2)
        
            if inactive[k]
                if meanAmps[k]  > threshold

                    #newlyActive[k] = true
                    inactive[k] = false
                    tStarts[k] = times[1]
                end
            else
                if meanAmps[k] < threshold
                    #newlyEnded[k] = true
                    inactive[k] = true
        
                    #ts = tStarts[k]:times[end]
                  
                    if  (times[end] - tStarts[k]) > ml
                        strands[(tStarts[k], k)] = [((tStarts[k]:times[end]) ./ fs) ifreq(@view(tfd[tStarts[k]:times[end], k]), fs) @view(amps[tStarts[k]:times[end], k])]
                    end
                    
                    tStarts[k] = 0
                end
            end

        end


        for k in axes(tfd, 2)
            if !inactive[k]
                *

        end
    end

    for k in axes(tfd, 2)
        if !inactive[k] 
            #ts = collect(tStarts[k]:times[end]) 
            if (times[end] - tStarts[k]) > ml
                strands[(tStarts[k], k)] = [((tStarts[k]:times[end]) ./ fs) ifreq(@view(tfd[tStarts[k]:times[end], k]), fs) @view(amps[tStarts[k]:times[end], k])]
        
            end
        end
    end


    return strands
end




mutable struct ComponentTracker{T}
    
    tStart::Int64
    tEnd::Int64
    signal::Vector{T}

end




end
=#
#end
