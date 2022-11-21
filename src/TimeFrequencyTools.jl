

module TimeFrequencyTools

using WAV, DSP, FFTW, LinearAlgebra 
using Peaks, Distributions, Statistics, StatsBase, RollingFunctions
using DataStructures, Memoize
using CairoMakie
using Revise



### signal reading, generation, and modification in time domain
export getSongMono, compress_max!, downsample, clipAudio, createSignalFromInstFA, sampleCosine, createEnvelope, midEarHighPass!


function getSongMono(fname, sTime, eTime)

    song, fs = wavread(fname)
    s_start = Int(sTime * fs) + 1
    s_end = Int(eTime * fs)
    song = song[s_start:s_end, :]
    song = (song[:, 1] .+ song[:, 2]) / 2

    return audioRecord([0.0 0.0; 0.0 0.0], song, fs, round(length(song)/fs, digits = 3), 1)

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

function downsample(audio, newFs, lowpass = true)
    

    if lowpass

        rs1 = DSP.resample(@view(audio.channels[:, 1]), newFs / audio.fs)
        rs = zeros(eltype(audio.channels), length(rs1), size(audio.channels, 2,))
        rs[:, 1] .= rs1

        Threads.@threads for j in 2:size(audio.channels, 2)
            rs[:, j] .= resample(@view(audio.channels[:, j]), newFs / audio.fs)
        end

    else
        factor = Int(ceil(audio.fs / newFs))

        rs = @view(audio.channels[1:factor:end, :])
        newFs = audio.fs / factor
    end

    return audioRecord(audio.cfs, rs, newFs, size(rs, 1) / newFs, audio.nchannels)
end

function clipAudio(aud, sTime, eTime)
    song = aud.channels
    fs = aud.fs
    s_start = Int(round(sTime * fs)) + 1
    s_end = Int(round(eTime * fs))
    song = song[s_start:s_end, :]
    

    return audioRecord(aud.cfs, song, fs, round(length(song)/fs, digits = 3), aud.nchannels)
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

function createSignalFromInstFA(frequencies, amplitudes, fs)

    song = sampleCosine(frequencies, amplitudes, fs)
    s = audioRecord([0.0 0.0; 0.0 0.0], song, fs, round(length(song)/fs, digits = 3), 1)
    
    return s
end

function sampleCosine(frequencies, amplitudes, fs)
    song = zeros(size(frequencies, 1))

    for j in axes(frequencies, 2)
        for t in eachindex(song)
            song[t] +=  amplitudes[t, j] * cos((2 * π * frequencies[t, j]) * t / fs)
        end
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
export getSTFT, getPowerSpectrum, freq_reassignment_direction

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

import LinearAlgebra.dot

export envelope_AC_DC, windowedSignal

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
    return audioRecord(aud.cfs, tfdNew, fsNew, dNew, aud.nchannels)

end

#synchrony strands
export getPeakList, connectStrands

function getPeakList(aud, threshold)
    ch = transpose(aud.channels)
    minAmplitude = maximum(ch)*threshold
    peakList = Vector{Dict{Int64, audStrand}}(undef, size(ch, 2))

    Threads.@threads for t in axes(ch, 2)

        pkDict = Dict{Int64, audStrand}()
        pks = findmaxima(view(ch, :, t))[1]

        #pks = findpeaks(ch, t, minAmplitude)
        
        if !isempty(pks)
            for f in pks
                if ch[f, t] >= minAmplitude
                    frs = aud.cfs[(f-1):(f+1), 2]
                    amps = ch[(f-1):(f+1), t] ./ sum(ch[(f-1):(f+1), t] )
                    fr = sum(frs .* amps)
                    freqEnv = MutableLinkedList{Float64}(fr)
                    ampEnv = MutableLinkedList{Float64}(ch[f, t])
                    freqDeriv = MutableLinkedList{Float64}(0.0)
                    pkSegment = audStrand(t, t, 0.0, freqEnv, ampEnv, freqDeriv)
                    pkDict[f] = pkSegment
                else
                    pkDict[0] = audStrand(0, 0, 0.0, MutableLinkedList{Float64}(0.0), MutableLinkedList{Float64}(0.0), MutableLinkedList{Float64}(0.0))
                end
            end
        else
            pkDict[0] = audStrand(0, 0, 0.0, MutableLinkedList{Float64}(0.0), MutableLinkedList{Float64}(0.0), MutableLinkedList{Float64}(0.0))
        end
            peakList[t] = pkDict
        
    end
    return peakList
end
function findpeaks(ch, t, threshold)
    pks = MutableLinkedList{Int64}()
    for f in axes(ch, 1)
        
        if (f > 1) && (f < size(ch, 1)) && (t > 1) && (t < size(ch, 2)) && (ch[f, t] >= threshold)
            if (ch[f, t] > (.5*ch[f-1, t] + ch[f+1, t])) | (ch[f, t] > (.5*ch[f, t - 1] + .5*ch[f, t+1]))
                push!(pks, f)
            end
        end
    end

    return collect(pks)
end
function appendLL!(s1::MutableLinkedList, s2::MutableLinkedList, alpha)

    for el in s2
        push!(s1, alpha*collect(s1)[end] + (1-alpha)*el)
    end
end
function mergeSegments(s1, s2, alpha = .2)

    if s1.t_end <= s2.t_start
        s1.t_end = s2.t_end
        appendLL!(s1.freq_envelope, s2.freq_envelope, alpha)
        appendLL!(s1.amp_envelope, s2.amp_envelope, alpha)

        
        fdiff = s2.freq_envelope[1] - s1.freq_envelope[end]
        setindex!(s2.freq_deriv, fdiff, 1)
        appendLL!(s1.freq_deriv, s2.freq_deriv, 1)
    end

    return s1

end

function connectStrands(peakList, minLength, alpha, beta)
    for t in axes(peakList, 1)
        
        for pk in peakList[t]
            f = pk[1]
            s1 = pk[2]
            findLink(peakList, s1, t, f, alpha, beta)
        end
        
    end
    strandBook = Dict{Int, Pair{Int, audStrand}}()
    
    strands = 0
    for window in peakList
        while !isempty(window)
            strands += 1
            str = pop!(window)
            if (str[2].t_end - str[2].t_start) >= minLength
                strandBook[strands] = str
            end
        end
    end

    return strandBook

end

function findLink(peakList, s1, t, f, alpha, beta)

    if t == length(peakList)
        return s1
    else 
        check_f = f .+ [0, 1, -1, 2, -2]
        f_scores = ones(Float64, length(check_f)) .* Inf
        for i in eachindex(check_f)
            if haskey(peakList[t+1], check_f[i])
                s2 = peakList[t+1][check_f[i]]
                slopeScore = getAngle(s1.freq_deriv[end], s2.freq_deriv[1]) / (0.5*π)
                ampScore = (s1.amp_envelope[end]-s2.amp_envelope[1])/mean([s1.amp_envelope[end], s2.amp_envelope[1]])
                f_scores[i] = (beta*abs(slopeScore)) + ((1-beta)*abs(ampScore))
            else
                f_scores[i] = Inf
            end
        end
        
        best = findmin(f_scores)

        if best[1] < Inf
            fbest = check_f[best[2]]
            best_s2 = peakList[t+1][fbest]

            if (t+1) == length(peakList)
                return best_s2
            else
                newSeg = mergeSegments(s1, findLink(peakList, best_s2, t+1, fbest, alpha, beta), alpha)
                delete!(peakList[t+1], fbest)
                return newSeg
            end

            
        else 
            return s1
        end
    end

end




### visualization functions
export plotSpectrogram, plotAudio, plotDFT, plotFB, plotStrandBook


function plotSpectrogram(aud; 
    tmin, 
    tmax, 
    fmin = aud.cfs[1, 2], 
    fmax = aud.cfs[end, 2], 
    maxbins = 500,
    fticks = 10,
    tticks = 10,
    zScale,  
    logY = false, 
    showNotes = false)
    
    if size(aud.channels, 1) > maxbins
        newFs = aud.fs / ceil(size(aud.channels, 1) / maxbins)
        aud = downsample(aud, newFs, false)
        if tmax > aud.duration
            tmax = aud.duration
        end
    end

   
    frs = aud.cfs[:, 2]
    tms = collect(1:size(aud.channels, 1)) ./ aud.fs
 
    fRes = frs[3]-frs[2]
    tRes = tms[3]-tms[2]

    #frs = [frs ; fRes+frs[end]]
    append!(frs, fRes+frs[end])
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
        aud.channels, 
        colormap = cgrad(:haline, 20, scale = :exp10, rev = false))
        Colorbar(f[:, 2], colormap = cgrad(:haline, 20, scale = :exp10))

    else
        CairoMakie.heatmap!(tms, 
        frs, 
        aud.channels,
        colormap = cgrad(:haline, 20, rev = false))
        Colorbar(f[:, 2], colormap = cgrad(:haline, 20))

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

function plotStrandBook(strandBook, aud; tmin = 0, tmax = -1, fmin = 0, fmax = -1, logY = false)

    if tmax == -1 
        tmax = aud.duration
    end

    if fmax == -1
        fmax = aud.cfs[end, 2]
    end

    f = plotSpectrogram(aud, tmin = tmin, tmax = tmax, fmin = fmin, fmax = fmax, fticks = 10, tticks = 10, zScale = "linear", logY = logY)

    for i in eachindex(strandBook)
        str = strandBook[i]
        s = str[2]

        freqs = collect(s.freq_envelope)
        amps = collect(s.amp_envelope)
        times = range(s.t_start, s.t_end, length(freqs)) ./ aud.fs
        lines!(times, freqs, color = :red, linewidth = 3, alpha = amps)
        #scatter!(times, freqs, color = :red, size = 1, alpha = amps)

    end
    f
end


### utility functions
export getAngle, windowAverages, mag, get_phase_shift, ifreq, mean_ifreq, norm_cols

function windowAverages(aud, wgt, wl, ovlp)
    
    channels = zeros(eltype(aud), length(arraysplit(collect(1:size(aud, 1)), wl, ovlp)), size(aud, 2))

    if isnothing(wgt)
        Threads.@threads for j in axes(channels, 2)
            window = arraysplit(1:size(aud, 1), wl, ovlp)
    
            for (i, wind) in enumerate(window)
                channels[i, j] = mean(@view(aud[Int.(wind), j]))
            end
        end
    else

        Threads.@threads for j in axes(channels, 2)
            window = arraysplit(1:size(aud, 1), wl, ovlp)
    
            for (i, wind) in enumerate(window)
                channels[i, j] = mean(@view(aud[Int.(wind), j]), weights(@view(wgt[Int.(wind), j])))
            end
        end
    end
    
   
    return channels
end


#phase angle of complex value
function getAngle(m1, m2)
    θ = atan((m1-m2) / (1 + m1*m2))
end

#magnitude of complex value
function mag(a)
    sqrt(real(a)^2 + imag(a)^2)
end


function get_phase_shift(x1, x2)
    x1p = transpose(x1 .^ (-1))
    em = x2 * x1p
    diags = div(size(em, 1) + size(em, 2) - 2, 2)
    
    for i in -diags:diags
        em[diagind(em, i)] .= mean(diag(em, i))
    end

    em ./= size(em, 2)
    return em
end



#Uses phase derivative to estimate instantaneous frequency of a sampled complex exponential. 
#No phase unwrapping is required.
function ifreq(x, fs)
    y = x ./ mag.(x)
    freqs = zeros(Float32, size(y, 1))
    
    Threads.@threads for i in 2:(size(y, 1) - 1)
        freqs[i] = fs * imag(conj(y[i]) * .5 * (y[i + 1] - y[i - 1])) / (π*2) #* fs / ( 2 * π )
    end

    freqs[1] = freqs[2]
    freqs[end] = freqs[end - 1]

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


#normalize columns of matrix
function norm_cols(x, p = 2)
    y = copy(x)
    for j in axes(y, 2)
        y[:, j] = normalize!(y[:, j], p)
    end
    return y
end




### filterbank related data structures
export audioRecord, audStrand


mutable struct audioRecord
    cfs::Matrix{Float64}
    channels::Array{Float64}
    fs::Float64
    duration::Float64
    nchannels::Int64
end


mutable struct audStrand
    t_start::Int64
    t_end::Int64
    init_phase::Float64
    freq_envelope::MutableLinkedList{Float64}
    amp_envelope::MutableLinkedList{Float64}
    freq_deriv::MutableLinkedList{Float64}
end


#=
function copy(aud::audioRecord)
    
    return audioRecord(copy(aud.cfs), copy(inst_amp), copy(aud.fs), copy(aud.duration), copy(aud.nchannels)) 

end
=#

end
