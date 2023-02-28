

module TimeFrequencyTools

using WAV, DSP, FFTW, LinearAlgebra 
using Peaks, Distributions, Statistics, StatsBase, RollingFunctions
using DataStructures, Memoize
using CairoMakie
using Revise



### signal reading, generation, and modification in time domain
export getSongMono, compress_max!, downsample, clipAudio, createSignalFromInstFA, sampleCosine, 
    createEnvelope, midEarHighPass!, getSTFT, getPowerSpectrum, freq_reassignment_direction, 
    envelope_AC_DC, windowedSignal, plotSpectrogram, plotAudio, plotDFT, plotFB, plotStrandBook, 
    getAngle, windowAverages, mag, mag2, get_phase_shift, ifreq, mean_ifreq, freqresponses, norm_cols,  phase_align, 
    synchrosqueeze, synchrosqueeze_hpss, ispeak, hzToErb, erbToHz, nonlinearSmooth, nonlinearSmooth!


export AudioRecord


function getSongMono(fname, sTime, eTime, fs)

    song, fs_old = wavread(fname)
    s_start = Int(sTime * fs_old) + 1
    s_end = Int(eTime * fs_old)
    song = song[s_start:s_end, :]
    song = (song[:, 1] .+ song[:, 2]) / 2

    ds = round(fs / fs_old, digits = 1)
    song = resample(song, ds)
    return AudioRecord([0.0 0.0; 0.0 0.0], song, ds * fs_old, round(length(song)/fs, digits = 3), 1)

end

function compress_max!(aud, minMult, maxMult)
    zmax = maximum(aud.channels) * maxMult
    zmin = minimum(aud.channels) * minMult

    Threads.@threads for j in axes(aud.channels, 2)
        for i in axes(aud.channels, 1)
            aud.channels[i, j] = aud.channels[i, j] < zmin ? zmin : aud.channels[i, j]
            aud.channels[i, j] = aud.channels[i, j] > zmax ? zmax : aud.channels[i, j]
        end
    end

end

#=
function downsample(audio::AudioRecord, newFs)
    




    if size(audio.channels, 2) > 1
        
        #=
        rs1 = DSP.resample(@view(audio.channels[:, 1]), round(newFs / audio.fs, digits = 1))
        rs = zeros(eltype(audio.channels), length(rs1), size(audio.channels, 2))
        rs[:, 1] .= rs1

        Threads.@threads for j in 2:size(audio.channels, 2)
            rs[:, j] .= resample(@view(audio.channels[:, j]), round(newFs / audio.fs, digits = 1))
        end

        newFs = round(newFs / audio.fs, digits = 1) * audio.fs
        return AudioRecord(audio.cfs, rs, newFs, size(rs, 1) / newFs, audio.nchannels)
    =#
        newFs = round(newFs / audio.fs, digits = 1) * audio.fs
        audio.channels = DSP.resample(audio.channels,  round(newFs / audio.fs, digits = 1), dims = 1)
    else
        rs = resample(audio.channels, round(newFs / audio.fs, digits = 1))
        newFs = round(newFs / audio.fs, digits = 1) * audio.fs
        return AudioRecord(audio.cfs, rs, newFs, length(rs) / newFs, 1)
    end

    

    #AudioRecord(audio.cfs, resample(audio.channels, newFs / audio.fs), newFs, len)

    
end
=#

#clip a segment of an AudioRecord 
function clipAudio(aud, sTime, eTime)
    song = aud.channels
    fs = aud.fs
    s_start = Int(round(sTime * fs)) + 1
    s_end = Int(round(eTime * fs))
    song = song[s_start:s_end, :]
    

    return AudioRecord(aud.cfs, song, fs, round(length(song)/fs, digits = 3), aud.nchannels)
end

function exp_sawtooth_envelope(len, decay, period)
    env = ones(Float64, len)
    for i in 2:len
    
        if (i % period) == 0
            env[i] = 1
        else
            env[i] = decay * env[i-1]
        end

    end

    return env
end

function createSignalFromInstFA(frequencies, amplitudes, fs, compression)

    frequencies = convert.(eltype(frequencies), DSP.resample(frequencies, compression, dims = 1))
    amplitudes = convert.(eltype(amplitudes), DSP.resample(amplitudes, compression, dims = 1))
    song = sampleCosine(frequencies, amplitudes, fs * compression)
    s = AudioRecord([0.0 0.0; 0.0 0.0], song, fs, round(length(song)/fs, digits = 3), 1)
    
    return s
end

function createSignalFromInstFA(frequencies, amplitudes, fs)

    #frequencies = DSP.resample(frequencies, compression, dims = 1)
    #amplitudes = DSP.resample(amplitudes, compression, dims = 1)
    song = sampleCosine(frequencies, amplitudes, fs)
    s = AudioRecord([0.0 0.0; 0.0 0.0], song, fs, round(length(song)/fs, digits = 3), 1)
    
    return s
end


function sampleCosine(frequencies, amplitudes, fs)
    song = zeros(eltype(frequencies), size(frequencies, 1))

    phases = cumsum(frequencies .* ((2 * π) / fs), dims = 1)
    for j in axes(frequencies, 2)
       
        song .+=  amplitudes[:, j] .* cos.((phases[:, j]))
        
    end

    return song
end
function createEnvelope(InstVals, linModRate, errorRate, duration, fs)
    envelope = zeros(Int(round(duration*fs)), length(InstVals))
    
    #linModRate ./= fs #change from inverse seconds to inverse samples
    errorRate /= fs

    for j in eachindex(InstVals)
        ef = cumsum(rand(Normal(0, errorRate), size(envelope, 1)))
        linMod = cumsum(repeat([linModRate[j]/fs], inner =length(ef)))
        @view(envelope[:, j]) .= repeat([InstVals[j]], inner = length(ef)) .+ linMod .+ ef
    end

    return envelope
end

function midEarHighPass!(aud, alpha)
    Threads.@threads for i in length(aud.channels):-1:2
        aud.channels[i] -= alpha *  aud.channels[i-1]
    end
end

#time-frequency distributions

function getSTFT(songMono, winLen, ovlp, interp, fs)
    
    complexSTFT = stft(songMono,  
                    winLen, 
                    Int(round(ovlp*winLen)), 
                    onesided = true, 
                    nfft = Int(winLen * interp), 
                    fs = fs, 
                    window = hanning)

    
    
    freqs = round.((0:(Int(winLen  * .5 * interp))) .* fs ./ Int(winLen  * interp), digits = 3)
    H = round((1 - ovlp) * winLen) / fs
    times = round.(((0:(size(complexSTFT, 2) - 1)) .+ (.5/(1 - ovlp))) .* H, digits = 4)
    
    
    #spc = power(spec) .^ 2
    return complexSTFT, freqs, times
end 

function getPowerSpectrum(complexSTFT)

    mag = Matrix{Float64}(undef, size(complexSTFT, 1),  size(complexSTFT, 2))

    Threads.@threads for j in axes(complexSTFT, 2)
        for i  in axes(complexSTFT, 1)
            mag[i, j] = complexSTFT[i, j] * conj(complexSTFT[i, j] )
        end
    end

    return mag
end


function freq_reassignment_direction(ifreq)

    reassignment = downsample(ifreq, ifreq.fs)
    
    Threads.@threads for j in axes(reassignment.channels, 2)
        for i in axes(reassignment.channels, 1)
            reassignment.channels[i, j] = (ifreq.cfs[j, 2] - ifreq.channels[i, j])/mean([ifreq.cfs[j, 2], ifreq.channels[i, j], 1*10^-14])
        end
        
    end

    return reassignment
end


function envelope_AC_DC(tfd, wl, step)
    nWindows = Int(div(size(tfd, 1) - wl, step))
    AC = zeros(Float64, nWindows, size(tfd, 2))
    DC = zeros(Float64, nWindows, size(tfd, 2))

    Threads.@threads for k in axes(AC, 2)
        for t in axes(AC, 1)
            DC[t, k] = mean(@view(tfd[((t - 1)*step + 1):((t - 1)*step + wl), k]))
            AC[t, k] = std(@view(tfd[((t - 1)*step + 1):((t - 1)*step + wl), k]))
        end
    end

    return AC, DC
end

function windowedSignal(aud, tfdNew, step)
    fsNew = aud.fs/step
    dNew = size(tfdNew, 1) / fsNew
    return AudioRecord(aud.cfs, tfdNew, fsNew, dNew, aud.nchannels)

end


### visualization functions


function plotSpectrogram(aud; 
    tmin, 
    tmax, 
    fmin = aud.cfs[1, 2], 
    fmax = aud.cfs[end, 2], 
    maxbins = 5000,
    fticks = 10,
    tticks = 10,
    zScale,  
    logY = false, 
    showNotes = false)
    
    if size(aud.channels, 1) > maxbins
        ds = Int(ceil(size(aud.channels, 1) / maxbins))
        channels = aud.channels[1:ds:end, :]
        fs = aud.fs / ds
       
    else
        channels = aud.channels
        fs = aud.fs
    end

   
    frs = aud.cfs[:, 2]
    tms = collect(1:size(channels, 1)) ./ fs
 
    fRes = hzToErb(frs[3])-hzToErb(frs[2])
    tRes = tms[3]-tms[2]

    #frs = [frs ; fRes+frs[end]]
    append!(frs, erbToHz(fRes) + frs[end])
    append!(tms, tRes+tms[end])
    #tms = [tms ; tRes+tms[end]]
    #append!(Vector(frs), [frs[end] + fRes])
    #append!(Vector(tms), [tms[end] + tRes])
   
  


    f = Figure(backgroundcolor = :white, resolution = (1100, 700))
    if logY
        ax = Makie.Axis(f[1, 1], title = "Spectrogram", xlabel = "Time (s)", ylabel = "Frequency (Hz)", yscale = log)
        ax.yticks = round.(exp.(range(log(minimum(frs)), log(maximum(frs)), fticks)))
    
    else
        ax = Makie.Axis(f[1, 1], title = "Spectrogram", ylabel = "Frequency (Hz)", xlabel = "Time (s)")
        ax.yticks = round.(range(minimum(frs), maximum(frs), fticks))
    
    end
    
    ax.xticks = round.(range(tms[1],tms[end], tticks), digits = 1)
    

    
    if zScale == "log"

        CairoMakie.heatmap!(tms, 
        frs, 
        channels, 
        colormap = cgrad(:viridis, 20, scale = :exp10, rev = false))
        Colorbar(f[:, 2], colormap = cgrad(:viridis, 20, scale = :exp10))

    else
        CairoMakie.heatmap!(tms, 
        frs, 
        channels)
        Colorbar(f[:, 2])

        #colormap = cgrad(:haline, 20, rev = false))
        #Colorbar(f[:, 2], colormap = cgrad(:haline, 20))

    end


    ylims!(fmin, fmax)
    xlims!(tmin, tmax)
    f
end

function plotAudio(aud, channel, sTime, eTime)
    sIndex = maximum([convert(Int32, round(sTime * aud.fs)), 1])
    eIndex = minimum([convert(Int32, round(eTime * aud.fs)), size(aud.channels[:, channel], 1)])
    f = Figure(backgroundcolor = :white, resolution = (1100, 700))

    channelFreq = string(round(aud.cfs[channel, 2]))
    Axis(f[1, 1], title = ("Waveform of " * channelFreq * " Hz frequency channel"), xlabel = "Time (s)", ylabel = "Amplitude")
    lines!(range(sTime, eTime, eIndex - sIndex + 1), aud.channels[sIndex:eIndex, channel])
    f
end

function plotDFT(mag, freqs, times, t, fmax)
    
    
    
    ti = findfirst(x -> x > t, times)
    fm = findfirst(x -> x > fmax, freqs)
    s = mag[1:fm, ti]



    f = Figure(backgroundcolor = :white, resolution = (1100, 700))
    ax = Makie.Axis(f[1, 1], title = "Fourier spectrum",  xlabel = "Frequency (Hz)", ylabel = "Amplitude")
    
    ax.xticks = round.(0:(freqs[fm]/10):freqs[fm])
    ax.yticks = round.(0:(maximum(s)/10):maximum(s), digits = 3)
    

    barplot!(freqs[1:fm], s)
    f

end

function plotFB(fb, ampmin, fmin, fmax, numFreqs, fs)
    f = Figure(backgroundcolor = :white, resolution = (1100, 700))

    Axis(f[1, 1], title = "Frequency response", xlabel = "Frequency (Hz)", ylabel = "Amplitude (dB)")
    frs = LinRange(fmin, fmax, numFreqs)

    for filt in fb.filters
        H = freqresp(filt, frs .* (2*pi/fs))
        hmag = 20 * log.(convert.(Float32, sqrt.(H .* conj.(H))))
        keepAmp = findall(x -> x >= ampmin, hmag)
        #keepFreq = findall(x -> x <= fmax && x >= fmin, W)
        #keep = findall(x -> (x in keepAmp) && ( x in keepFreq), 1:length(hmag))

        lines!(frs[keepAmp], hmag[keepAmp], linewidth = 2)
    end

    f
    

end

function plotChannels(channelRMS, showChannels, alpha, gap)
    vals = channelRMS.channels[:, showChannels]
    frs = string.(convert.(Int64, round.(channelRMS.cfs[showChannels, 2], digits = 0)))
    samples = size(channelRMS.channels, 1)
    times = round.(LinRange(0, channelRMS.duration, samples), digits = 6)
    yrange = extrema(vals)

    fig = Figure(backgroundcolor = :white, resolution = (1100, 700))

    f = GridLayout(fig[1:length(showChannels), 1])
    gls = [GridLayout(f[row, 1]) for row in 1:length(showChannels)]

    axs = Vector{Axis}(undef, length(gls))
    for a in eachindex(axs)
        axs[a] = Axis(gls[a][1, 1])
        if a > 1
            linkxaxes!(axs[a], axs[a - 1])
        end
        ylims!(axs[a], yrange)

        lines!(axs[a], times, vals[:, length(showChannels) - a + 1], color = (:black, alpha), linewidth = .75)
    
        #Box(f[a, 1, Left()], color = :gray90)
        rowgap!(f, gap)
        Label(f[a, 1, Left()], frs[length(gls) - a + 1], valign = :bottom, halign = :left)
    end

    hidespines!.(axs, :t, :r, :b)
    hidexdecorations!.(axs[1:(end - 1)])
    hideydecorations!.(axs)





    Label(fig[1, :, Top()], "Channel amplitude", valign = :bottom,
        font = "TeX Gyre Heros Bold",
        padding = (0, 0, 5, 0))
    Label(fig[length(showChannels) + 1, :, Bottom()], "Time (s)", valign = :top,
        font = "TeX Gyre Heros Bold",
        padding = (0, 0, 0, 0))
    Label(fig[length(showChannels) + 1, :, TopLeft()], "Freq. (Hz)", valign = :top,
        font = "TeX Gyre Heros Bold",
        padding = (0, 0, 0, 0))

    fig

    return fig

end

### utility functions

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

#smooth a time-frequency distribution nonlinearly by estamating instantaneous frequency and amplitude envelope
#of each channel, low pass filtering those values, and then resampling complex sinusoids in each channel using 
#the estimated amplitude envelope and phase (found by taking the cumulative sum of frequency), assuming an 
#initial phase of 0.
function nonlinearSmooth(signal, magWindow, freqWindow)

    m = mag.(signal)
    fr = ifreq(signal, 1)
    s = copy(signal)
    Threads.@threads for j in axes(m, 2)
        m[:, j] .= runmean(@view(m[:, j]), magWindow)
        fr[:, j] .= cumsum(runmean(@view(fr[:, j]), freqWindow))
        s[:, j] .= @view(m[:, j]) .* cispi.(2 .* @view(fr[:, j]))
    end

    s
end


function nonlinearSmooth!(signal, magWindow, freqWindow)

    m = mag.(signal)
    fr = ifreq(signal, 1)

    for j in axes(m, 2)
        m[:, j] .= runmean(@view(m[:, j]), magWindow)
        fr[:, j] .=  cumsum(runmean(@view(fr[:, j]), freqWindow))
        signal[:, j] .= @view(m[:, j]) .*  cispi.(2 .* @view(fr[:, j]))
    end


end


#phase angle of complex value
function getAngle(m1, m2)
    θ = atan((m1-m2) / (1 + m1*m2))
end

#magnitude of complex value
function mag(a)
    sqrt(real(a)^2 + imag(a)^2)
end

function mag2(a)
    real(a)^2 + imag(a)^2
end



export thresholder, thresholder!

function thresholder(val::T, threshold::Number) where T <: Real
    val < threshold ? 0.0 : val
end

function thresholder!(vals::AbstractArray{T}, threshold::Number) where T <: Real
   
    Threads.@threads for i in eachindex(vals)
        if vals[i] < threshold
           vals[i] = 0
        end
    end
end

function thresholder!(vals::AbstractArray{T}, threshold::AbstractArray) where T <: Real
   
    Threads.@threads for i in eachindex(vals)
        if vals[i] < threshold[i]
           vals[i] = 0
        end
    end
end

function thresholder!(x::AbstractArray{T}, vals::AbstractArray{T}, threshold::Number) where T <: Real
   
    Threads.@threads for i in eachindex(vals)
        if vals[i] < threshold
            x[i] = 0
        else
            x[i] = vals[i]
        end
    end
end


function thresholder(val::T, threshold::Number) where T <: Complex
    mag(val) < threshold ? 0 : val
end

function thresholder!(vals::AbstractArray{T}, threshold::Number) where T <: Complex
   
    Threads.@threads for i in eachindex(vals)
        if mag(vals[i]) < threshold
           vals[i] = 0
        end
    end
end

function thresholder!(x::AbstractArray{T}, vals::AbstractArray{T}, threshold::Number) where T <: Complex
   
    Threads.@threads for i in eachindex(vals)
        if mag(vals[i]) < threshold
            x[i] = 0
        else
            x[i] = vals[i]
        end
    end
end

function thresholder!(vals::AbstractArray{T}, threshold::AbstractArray) where T <: Complex
   
    Threads.@threads for i in eachindex(vals)
        if mag(vals[i]) < threshold[i]
           vals[i] = 0
        end
    end
end


function hzToErb(f)
    erb = 21.4 * log10((4.37 * f / 1000) + 1)
    return erb
end

function erbToHz(erb)
    f = (10 ^ (erb / 21.4) - 1) * 1000 / 4.37 
    return f
end


function get_phase_shift(x1::Vector{T}, x2::Vector{T}) where T <: Real
    a = [ones(eltype(x1), length(x1), 1) x1]
    beta = a \ x2
    
    return acos(beta[2])/pi 
end

function get_phase_shift(x1::Vector{T}, x2::Vector{T}) where T <: Complex
    a = [ones(eltype(x1), length(x1), 1) x1]
    beta = a \ x2
    
    return atan(imag(beta[2]), real(beta[2])) / pi
end


function phase_align(x1, x2) 
    a = [ones(eltype(x1), length(x1), 1) x1]
    beta =  a \ x2
    
    return a * beta
end


#Uses phase derivative to estimate instantaneous frequency of a sampled complex exponential. 
#No phase unwrapping is required.
function ifreq(x::Vector, fs, wl) 
    y = x ./ mag.(x)
    freqs = zeros(Float32, size(y, 1))
    
    Threads.@threads for i in 2:(size(y, 1) - 1)
        freqs[i] = fs * imag(conj(y[i]) * .5 * (y[i + 1] - y[i - 1])) / (π*2) #* fs / ( 2 * π )
        freqs[i] = (freqs[i] < 0) || (isnan(freqs[i])) ? 0 : freqs[i]
    end

    freqs[1] = freqs[2]
    freqs[end] = freqs[end - 1]

    freqs .= runmean(freqs, wl)
    return freqs
end

function ifreq(x::Matrix, fs, wl = 10)
    y = x ./ (mag.(x) .+ eps())
    freqs = zeros(Float32, size(y))
    
    Threads.@threads for j in axes(x, 2)
        for i in 2:(size(y, 1) - 1)
            freqs[i, j] = fs * imag(conj(y[i, j]) * .5 * (y[i + 1, j] - y[i - 1, j])) / (π*2) #* fs / ( 2 * π )
            freqs[i, j] = (freqs[i, j] < 0) || (isnan(freqs[i, j])) ? 0 : freqs[i, j]
        end
        freqs[1, j] = freqs[2, j]
        freqs[end, j] = freqs[end - 1, j]

        freqs[:, j] .= runmean(@view(freqs[:, j]), wl)
    end

    
    #replace!(x -> x < 0 )
    return freqs
end

function ifreq(x::AbstractArray, fs, wl) 
    y = x ./ mag.(x)
    freqs = zeros(Float32, size(y, 1))
    
    Threads.@threads for i in 2:(size(y, 1) - 1)
        freqs[i] = fs * imag(conj(y[i]) * .5 * (y[i + 1] - y[i - 1])) / (π*2) #* fs / ( 2 * π )
        freqs[i] = (freqs[i] < 0) || (isnan(freqs[i])) ? 0 : freqs[i]
    end

    freqs[1] = freqs[2]
    freqs[end] = freqs[end - 1]

    freqs .= runmean(freqs, wl)

    return freqs
end

#slightly more efficient if just the mean instantaneous frequency for the sample is needed
function mean_ifreq(x, fs, len)
    x ./= mag.(x)
    freqs = 0.0
    
    for i in 2:(size(x, 1) - 1)
        freqs += (fs * imag(conj(x[i]) * .5 * (x[i + 1] - x[i - 1])) / ((len - 2) * 2 * π)) 
    end

    return freqs
end


#get pairwise frequency responses between each filter (frequency response of filter i at the center frequency of filter j)
function freqresponses(fb, fs)
    freqr = zeros(ComplexF32, length(fb.filters), length(fb.filters))
    
    Threads.@threads for j in axes(freqr, 2)
        freqr[:, j] .= freqresp(fb.filters[j], fb.cfs[:, 2] .* (2 * π) ./ fs)
        #freqr[j, j] = 0 
    end

    #probs = copy(freqr)
    #for i in 2:order
    #    freqr[:, j] 

    #end

    return freqr
end


#normalize columns of matrix
function norm_cols(x, p = 2)
    y = deepcopy(x)
    for j in axes(y, 2)
        y[:, j] = normalize!(y[:, j], p)
    end
    return y
end



function synchrosqueeze(ifreqs,  amps::Matrix{T}, cfs; threshold = .01, maxdist = 10) where T <: Real
    squeezed = zeros(eltype(amps), size(ifreqs))
    tr = threshold * maximum(amps)
    fdiff = (cfs[2, 1] - cfs[1, 1])
    Threads.@threads for t in axes(squeezed, 2)
        for k in axes(squeezed, 1)
            if amps[t, k] > tr
                newBin =  ((hzToErb(ifreqs[k, t]) - cfs[1, 1])/fdiff) + 1

                if checkindex(Bool, 1:size(squeezed, 1), newBin) && (abs(newBin - k) < maxdist) #&& (newBin != k)
                    squeezed[floor(Int32, newBin), t] += amps[t, k] * (newBin - floor(newBin))
                    squeezed[ceil(Int32, newBin), t] += amps[t, k] * (ceil(newBin) - newBin)
                
                end
            end
        
        end
    
    end
    return permutedims(squeezed)
end

function synchrosqueeze(ifreqs, amps::Matrix{T}, cfs; threshold = .01, maxdist = 10) where T <: Complex
    squeezed = zeros(eltype(amps), size(ifreqs))
    tr = threshold * maximum(mag.(amps))
    fdiff = (cfs[2, 1] - cfs[1, 1])
    Threads.@threads for t in axes(squeezed, 2)
        for k in axes(squeezed, 1)
            if mag(amps[t, k]) > tr
                newBin =  ((hzToErb(ifreqs[k, t]) - cfs[1, 1])/fdiff) + 1

                if checkindex(Bool, 1:size(squeezed, 1), newBin) && (abs(newBin - k) < maxdist) #&& (newBin != k)
                    squeezed[floor(Int32, newBin), t] += amps[t, k] * (newBin - floor(newBin))
                    squeezed[ceil(Int32, newBin), t] += amps[t, k] * (ceil(newBin) - newBin)
                
                end
            end
        
        end
    
    end
    return permutedims(squeezed)
end

function synchrosqueeze_hpss(squeezed, wl)

    squeezed = permutedims(squeezed)

    Threads.@threads for t in axes(squeezed, 2)
        squeezed[:, t] .-= runmin(@view(squeezed[:, t]), wl)
    
    end
    squeezed = permutedims(squeezed)

   return squeezed
end


function ispeak(tfd, threshold)

    tfd = permutedims(tfd)
    peaks = zeros(Bool, size(tfd))
    minAmp = maximum(tfd) * threshold
    Threads.@threads for t in axes(peaks, 2)
        for k in 2:(size(peaks, 1) - 1)
            peaks[k, t] = (tfd[k, t] > minAmp) && (tfd[k, t] > tfd[k - 1, t]) && (tfd[k, t] > tfd[k + 1, t])
        end
    end
    tfd = permutedims(tfd)

   return permutedims(peaks)
end
### filterbank related data structures
export AudioRecord


mutable struct AudioRecord
    cfs::Matrix{Float64}
    channels::Array{Float64}
    fs::Float64
    duration::Float64
    nchannels::Int64
end



#=
function copy(aud::AudioRecord)
    
    return AudioRecord(copy(aud.cfs), copy(inst_amp), copy(aud.fs), copy(aud.duration), copy(aud.nchannels)) 

end
=#

end
