

module timeFrequencyAnalysis

using WAV, DSP, FFTW #fast fourier transform backend
using Peaks, Distributions, Statistics, StatsBase, RollingFunctions
using DataStructures, Memoize
using CairoMakie
using Revise



### signal reading, generation, and modification in time domain
export getSongMono, compress_max!, downsample, clipAudio, createSignalFromInstFA, sampleCosine, createEnvelope, midEarHighPass!


getSongMono = function(fname, sTime, eTime)

    song, fs = wavread(fname)
    s_start = Int(sTime * fs) + 1
    s_end = Int(eTime * fs)
    song = song[s_start:s_end, :]
    song = (song[:, 1] .+ song[:, 2]) / 2

    return audioRecord([0.0 0.0; 0.0 0.0], song, fs, round(length(song)/fs, digits = 3), 1)

end

compress_max! = function(aud, minMult, maxMult)
    zmax = maximum(aud.channels) * maxMult
    zmin = minimum(aud.channels) * minMult

    Threads.@threads for j in axes(aud.channels, 2)
        for i in axes(aud.channels, 1)
            aud.channels[i, j] = aud.channels[i, j] < zmin ? zmin : aud.channels[i, j]
            aud.channels[i, j] = aud.channels[i, j] > zmax ? zmax : aud.channels[i, j]
        end
    end

end

downsample = function(audio, new_fs)
    
    rs1 = DSP.resample(audio.channels[:, 1], new_fs / audio.fs)
    rs = zeros(Float64, length(rs1), size(audio.channels, 2,))
    rs[:, 1] = rs1

    Threads.@threads for j in 2:size(audio.channels, 2)
        rs[:, j] = resample(audio.channels[:, j], new_fs / audio.fs)
    end

    return audioRecord(audio.cfs, rs, new_fs, audio.duration, 1)
end

clipAudio = function(aud, sTime, eTime)
    song = aud.channels
    fs = aud.fs
    s_start = Int(round(sTime * fs)) + 1
    s_end = Int(round(eTime * fs))
    song = song[s_start:s_end, :]
    

    return audioRecord(aud.cfs, song, fs, round(length(song)/fs, digits = 3), aud.nchannels)
end

exp_sawtooth_envelope = function(len, decay, period)
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

createSignalFromInstFA= function(frequencies, amplitudes, fs)

    song = sampleCosine(frequencies, amplitudes, fs)
    s = audioRecord([0.0 0.0; 0.0 0.0], song, fs, round(length(song)/fs, digits = 3), 1)
    
    return s
end

sampleCosine = function(frequencies, amplitudes, fs)
    song = zeros(size(frequencies, 1))

    for j in axes(frequencies, 2)
        for t in eachindex(song)
            song[t] +=  amplitudes[t, j] * cos((2 * π * frequencies[t, j]) * t / fs)
        end
    end

    return song
end
createEnvelope = function(InstVals, linModRate, errorRate, duration, fs)
    envelope = zeros(Int(round(duration*fs)), length(InstVals))
    
    #linModRate ./= fs #change from inverse seconds to inverse samples
    errorRate /= fs

    for j in eachindex(InstVals)
        ef = cumsum(rand(Normal(0, errorRate), size(envelope, 1)))
        linMod = cumsum(repeat([linModRate[j]/fs], inner =length(ef)))
        envelope[:, j] .= repeat([InstVals[j]], inner = length(ef)) .+ linMod .+ ef
    end

    return envelope
end

function midEarHighPass!(aud, alpha)
    Threads.@threads for i in length(aud.channels):-1:2
        aud.channels[i] -= alpha *  aud.channels[i-1]
    end
end

#time-frequency distributions
export getSTFT, getPowerSpectrum, amp_from_windowPeaks, get_inst_power_exponential, get_hilbert_amp

getSTFT = function(songMono, winLen, ovlp, interp, fs)
    
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

getPowerSpectrum = function(complexSTFT)

    mag = Matrix{Float64}(undef, size(complexSTFT, 1),  size(complexSTFT, 2))

    Threads.@threads for j in axes(complexSTFT, 2)
        for i  in axes(complexSTFT, 1)
            mag[i, j] = complexSTFT[i, j] * conj(complexSTFT[i, j] )
        end
    end

    return mag
end

amp_from_windowPeaks = function(aud, wl, ovlp)
    channels = zeros(Float64, length(arraysplit(collect(1:size(aud.channels, 1)), wl, ovlp)), aud.nchannels)

    Threads.@threads for j in axes(channels, 2)
        window = arraysplit(@view(aud.channels[:, j]), wl, ovlp)
    
        for (i, wind) in enumerate(window)
            wind_ave = mean(findmaxima(wind)[2])
            if isnan(wind_ave)
                wind_ave = 0.0
            end
            channels[i, j] = wind_ave
        end
    end
    fsNew = aud.fs/(wl - ovlp)
    dNew = size(channels, 1) / fsNew
    return audioRecord(aud.cfs, channels, fsNew, dNew, aud.nchannels)
end

get_inst_power_exponential = function(aud, alpha)

    #inst_amp = zeros(Float64, size(aud.channels))
    inst_amp = copy(aud.channels)
    
    Threads.@threads for j in axes(inst_amp, 2)
    
        for i in axes(inst_amp, 1)
            if i==1
                inst_amp[1, j] = abs(inst_amp[1, j])
            else
                inst_amp[i, j] = alpha*inst_amp[i-1, j] + (1-alpha)*abs(inst_amp[i, j])
            end
        end
    end

    

    return audioRecord(aud.cfs, inst_amp, aud.fs, aud.duration, aud.nchannels) #, audioRecord(aud.cfs, inst_amp, aud.fs, aud.duration, aud.nchannels)

end

get_hilbert_amp = function(aud; lowpass = false, cutoff = 500, bwOrder = 4)

    inst_amp = zeros(Float64, size(aud.channels, 1),  size(aud.channels, 2))
    #inst_freq = zeros(Float64, size(aud.channels, 1),  size(aud.channels, 2))
    
    responsetype = Lowpass(cutoff; fs=aud.fs)
    designmethod = Butterworth(bwOrder)

    Threads.@threads for j in axes(inst_amp, 2)
        x = hilbert(@view(aud.channels[:, j]))
        for i in eachindex(x)
            inst_amp[i, j] = sqrt(x[i] * conj(x[i]))
        end

        if lowpass
            inst_amp[:, j] = filt(digitalfilter(responsetype, designmethod), inst_amp[:, j])
            #inst_amp[:, j] = filt(digitalfilter(responsetype, designmethod), inst_amp[:, j])
            #inst_freq[:, j] = filt(digitalfilter(responsetype, designmethod), inst_freq[:, j])
        end
    end
    

    return audioRecord(aud.cfs, inst_amp, aud.fs, aud.duration, aud.nchannels) #, audioRecord(aud.cfs, inst_amp, aud.fs, aud.duration, aud.nchannels)

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





### gammatone filter banks
#bandwidth values
#coefficients from https://www.scinapse.io/papers/396690109#fullText

export create_gt_cascade_filterBank, applyFilterbank, meddis_IHC
export hzToErb, erbWidth, erbToHz, getErbBins

erbWidth = function(f)
    erb = 24.7 * ((4.37 * f / 1000) + 1)
    return erb
end

hzToErb = function(f)
    erb = 21.4 * log10((4.37 * f / 1000) + 1)
    return erb
end

erbToHz = function(erb)
    f = (10 ^ (erb / 21.4) - 1) * 1000 / 4.37 
    return f
end

getErbBins = function(fmin, fmax, bins)
    
    erbBins = LinRange(hzToErb(fmin), hzToErb(fmax), bins)
    hzBins = erbToHz.(erbBins)
    freqs = [erbBins hzBins]

    return freqs
end

getGT_transferFunc = function(params, bandwidth)
    
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

get_4thOrder_gammatone = function(cf, fs, bandwidth)
    

    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 1, "Glasberg"), bandwidth)
    filt1 =  Biquad(transFun...)


    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 2, "Glasberg"), bandwidth)
    filt2 =  Biquad(transFun...)

    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 3, "Glasberg"), bandwidth)
    filt3 = Biquad(transFun...)


    transFun = getGT_transferFunc(biquadParams(cf, 1/fs, 4, "Glasberg"), bandwidth)
    filt4 = Biquad(transFun...)


# o2 = freqresp(SecondOrderSections([filt1, filt2, filt3, filt4], 1), cf*2*π/fs)

    o2 = freqresp(SecondOrderSections([filt1, filt2, filt3, filt4], 1), cf * 2 * pi / fs)
    peakGain = convert.(Float64, sqrt.(o2 .* conj.(o2)))
    
    
    

    return SecondOrderSections([filt1, filt2, filt3, filt4], 1/peakGain)

end

create_gt_cascade_filterBank = function(fmin, fmax, fs, bins, bandwidth)
    freqBins = getErbBins(fmin, fmax, bins)
    filts = Vector{SecondOrderSections{:z, Float64, Float64}}(undef, bins)
    
    
    for i in axes(freqBins, 1)
        filts[i] = get_4thOrder_gammatone(freqBins[i, 2], fs, bandwidth)
    
    end

    fb = gt_cascaded_filterbank(freqBins, filts)
    return fb
end

applyFilterbank = function(s, fb)
    newNumChannels = size(fb.cfs, 1)
    splitAudio = Matrix{Float64}(undef, size(s.channels, 1), newNumChannels)
    Threads.@threads for j in axes(splitAudio, 2)
        x = DSP.filt(fb.filters[j], s.channels)
        splitAudio[:, j] = x
    end

    return audioRecord(fb.cfs, splitAudio, s.fs, s.duration, newNumChannels)
end

#meddis inner hair cell model
meddis_IHC = function(aud;
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

    dt=1/aud.fs #time step
    sTime = 0
    eTime = aud.duration + sTime - dt
    times = sTime:dt:eTime



    Cs = zeros(Float64, length(times),  aud.nchannels) 
    Qt = zeros(Float64, aud.nchannels) 
    Wt = zeros(Float64, aud.nchannels) 
    

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

    for ch in 1:aud.nchannels
        for t in 2:size(Cs, 1)
        
            st = aud.channels[t, ch]
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


### visualization functions
export plotSpectrogram, plotAudio
export plotDFT, plotFB, plotStrandBook


function plotSpectrogram(aud; 
    tmin, 
    tmax, 
    fmin = aud.cfs[1, 2], 
    fmax = aud.cfs[end, 2], 
    fticks = 10,
    tticks = 10,
    zScale,  
    logY = false, 
    showNotes = false)
    
    if aud.fs > 500
        aud = downsample(aud, 500)
    end

    
    frs = aud.cfs[:, 2]
    tms = collect(1:size(aud.channels, 1)) ./ aud.fs
    
    

    

    fRes = frs[3]-frs[2]
    tRes = tms[3]-tms[2]

    frs = [frs ; fRes+frs[end]]
    tms = [tms ; tRes+tms[end]]
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
        hmag = 20 * log.(convert.(Float64, sqrt.(H .* conj.(H))))
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
export getAngle, windowAverages

function getAngle(m1, m2)
    θ = atan((m1-m2) / (1 + m1*m2))
end

function windowAverages(aud, wl, ovlp)
    channels = zeros(Float64, length(arraysplit(collect(1:size(aud.channels, 1)), wl, ovlp)), aud.nchannels)

    Threads.@threads for j in axes(channels, 2)
        window = arraysplit(@view(aud.channels[:, j]), wl, ovlp)
    
        for (i, wind) in enumerate(window)
            channels[i, j] = mean(wind)
        end
    end
    fsNew = aud.fs/(wl - ovlp)
    dNew = size(channels, 1) / fsNew
    return audioRecord(aud.cfs, channels, fsNew, dNew, aud.nchannels)
end


### filterbank related data structures
export gt_cascaded_filterbank, audioRecord, audStrand

mutable struct biquadParams

    cf::Float64
    T::Float64
    order::Int64
    erbModel::String
    
end

mutable struct gt_cascaded_filterbank

    cfs::Matrix{Float64}
    filters::Vector{SecondOrderSections{:z, Float64, Float64}}
    
end

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



end
