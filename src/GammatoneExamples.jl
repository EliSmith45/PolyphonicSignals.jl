using CairoMakie 
cd("PolyphonicSignals")
using Revise

includet("./TimeFrequencyTools.jl")
includet("./GammatoneFilterBanks.jl")
includet("./AdaptiveSingularFilterBank.jl")

using .TimeFrequencyTools
using .GammatoneFilterBanks
using .AdaptiveSingularFilterBank



fr = createEnvelope([200, 300, 400, 600],  [10, -10, 10, -10], 20, 2, 20000)
am = createEnvelope([100, 50, 80, 30],  [10, 5, 10, 5], 15, 2, 20000)
song = createSignalFromInstFA(fr, am, 20000)
plotAudio(song, 1, 0.0, .1)

#Short-time Fourier transform as a baseline for comparison
complexSTFT, frs, times = getSTFT(song.channels, 2048, .9, 5, song.fs)
mag = getPowerSpectrum(complexSTFT)
mag = mag ./ maximum(mag)
spec = audioRecord([hzToErb.(frs) frs], transpose(mag), round(1/(times[4]-times[3])), song.duration, length(frs))

plotSpectrogram(spec, tmin = 0, tmax = 1.8, fmin = 0, fmax = 1000, zScale = "linear", logY = false)


### testing the gammatone filter bank
gammatoneFB = create_gt_cascade_filterBank(40, 1000, song.fs, 200, .2)
plotFB(gammatoneFB, -200, 0, 200, 2000, song.fs)

gtSpectra = downsample(song, song.fs)
gtSpectra.channels = applyFilterbank(song.channels, gammatoneFB)
gtSpectra.nchannels = size(gtSpectra.channels, 2)
gtSpectra.cfs = gammatoneFB.cfs
audioSplit = get_inst_power_exponential(gtSpectra, .99)
plotSpectrogram(audioSplit, tmin = 0, tmax = 1, fmin = 0, fmax = 1000, zScale = "linear", logY = false)


spectraPeaks = amp_from_windowPeaks(gtSpectra, 500, 497) #estimate amplitude envelope using windowed averages of signal peaks
plotSpectrogram(spectraPeaks, tmin = 0, tmax = spectraPeaks.duration, fmin = spectraPeaks.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)

gtSpectra = audioSplit = spectraPeaks = mag = spec = complexSTFT = 0
GC.gc()



#testing the complex gammatone

complexGammatoneFB = createComplexGTFilterBank(40, 1000, song.fs, 200, .35, .016)
plotFB(complexGammatoneFB, -200, 0, 200, 2000, song.fs)

cgtSpectra = downsample(song, song.fs)
signal = applyComplexFilterbank(song.channels, complexGammatoneFB, "signals")
cgtSpectra.channels = amp = mag.(signal)

cgtSpectra.nchannels = size(cgtSpectra.channels, 2)
cgtSpectra.cfs = complexGammatoneFB.cfs
#complexAudioSplit = get_inst_power_exponential(cgtSpectra, .99)
plotSpectrogram(cgtSpectra, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)


windowLength = 20
overlap = 10

ds =  downsample(cgtSpectra,  8000)

@time ifrs1, ifrs2, ifrs3 = clean_crossTalk!(cgtSpectra, signal, windowLength, overlap; neighborhood = 6, sigma_c = 1.5, sigma_r = 2.5, cohesionOrder = 14, smoothingWindow = 115, threshold = .5, step = .1)
coherence = downsample(cgtSpectra,  .1*(cgtSpectra.fs/(windowLength - overlap)))
repulsion = downsample(cgtSpectra,  (cgtSpectra.fs/(windowLength - overlap)))


ifrs.channels = 2 .* (ifrs1) ./ (ifrs1 .+ ifrs2)# ./ifrs2# ./ ifrs2# ./ (eps() .+ ifrs2))
plotSpectrogram(coherence, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)
plotSpectrogram(ifrs, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)
ifrs3 = windowAverages(ifrs3, windowLength, overlap, true, ifrs.channels)

amp = windowAverages(cgtSpectra.channels, windowLength, overlap)
amps = downsample(ifrs, ifrs.fs)
amps.channels = copy(amp)


amps.channels  ./= maximum(amps.channels )

amps.channels .*= ifrs.channels
#amps.channels ./= ifrs2
plotSpectrogram(amps, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)

ifrs = (ifrs3)
tm = (1:size(amps.channels, 1)) / (amps.fs)
timeCoord = repeat(collect(tm), outer = size(amps.channels, 2))
freqCoord = reshape(ifrs, length(ifrs)) 
sizeCoord = reshape(amp, length(amp)) .^(1)
keep = findall(x -> x > 0.1, sizeCoord)
timeCoord = timeCoord[keep]
freqCoord = freqCoord[keep]
sizeCoord = sizeCoord[keep]
f = plotSpectrogram(amps, tmin = 0, tmax = amps.duration, fmin = amps.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)
scatter!(timeCoord, freqCoord, markersize = sizeCoord .* 10, color = [(:red, alpha) for alpha in sizeCoord])
f

squeezed = downsample(cgtSpectra, (cgtSpectra.fs/(windowLength - overlap)))
amps = downsample(squeezed, squeezed.fs)
amps.channels = amp
@time squeezed.channels = synchrosqueeze(ifrs, amp, cgtSpectra.cfs, .5)

plotSpectrogram(squeezed, tmin = 0, tmax = squeezed.duration, fmin = squeezed.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)
plotSpectrogram(amps, tmin = 0, tmax = amps.duration, fmin = amps.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)










###### AdaptiveFB #######
using Distributions


n = 400000
fs = 16000
s = zeros(Float32, n)
s = 2.0f0 .* sin.((0:n)./fs .*(2π*100) ) .+ 
   # 1 .* sin.((0:n)./fs .*(2π*400)) .+ 
    #1.0 .* sin.((0:n)./fs .*(2π*350)) .+ 
    1.02 .* sin.((0:n)./fs .*(2π*150)) #.+ 
   # sin.(10 .* sin.((0:n)./fs .*(2π*35))) .+ 
    #rand(Normal(0, .005), n + 1) 
s = [zeros(Float32, 5000) ; s ; zeros(Float32, 5000)]

fb = createComplexGTFilterBank(75, 200, fs, 200, .25, .016)


@time signal = applyComplexFilterbank(s, fb);

amp = mag.(signal)


wl = 3000
step = 1000

spec = audioRecord(fb.cfs, amp, fs, (n+1 + 10000)/fs, size(fb.cfs, 1))
@time AC, DC = envelope_AC_DC(amp, wl, step)
reduced_interference = windowedSignal(spec, DC .- AC, step)
AC = windowedSignal(spec, AC, step)
DC = windowedSignal(spec, DC, step)

plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(AC, tmin = 0, tmax = AC.duration, fmin = AC.cfs[1,2], fmax = AC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(DC, tmin = 0, tmax = DC.duration, fmin = DC.cfs[1,2], fmax = DC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(reduced_interference, tmin = 0, tmax = DC.duration, fmin = DC.cfs[1,2], fmax = DC.cfs[end,2], zScale = "linear", logY = false)





ar_lags = 300
channel_components = 1
overall_components = 50
min_eigval = 0.05


tfd = permutedims(signal)
ri = permutedims(reduced_interference.channels)


@time bases, activations, comps  = trackSubspaceLR(tfd; ri = ri, wl, ar_lags, step = step, overall_components, channel_components, min_eigval);
@time sigs, ifrs, amps = recreate_signal(bases, activations, comps, wl, ar_lags, .0025, fs)
@time linked_sinusoids, linked_freqs, linked_amps = link_comps(sigs, ifrs, amps, comps, wl, step, 100, size(tfd, 2), .9)
@time points = envelopes(linked_sinusoids, linked_freqs, linked_amps, fs, .0025, 20, 10)

points[:, 3] .^= (1/2)
points[:, 3] ./= maximum(points[:, 3])

f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 25, color = :red)

f





################# testing on a real song ###########
fs = 16000
song = downsample(getSongMono("../porterRobinson_shelter.wav", 1, 3), fs, false)

midEarHighPass!(song, .95)

fb = createComplexGTFilterBank(40, 1000, fs, 100, .25, .016)
signal = applyComplexFilterbank(vec(song.channels), fb)
amp = mag.(signal)

wl = 2000
step = 1000

spec = audioRecord(fb.cfs, amp, fs, song.duration, size(fb.cfs, 1))
@time AC, DC = envelope_AC_DC(amp, wl, step)
reduced_interference = windowedSignal(spec, DC .- AC, step)
AC = windowedSignal(spec, AC, step)
DC = windowedSignal(spec, DC, step)


plotSpectrogram(AC, tmin = 0, tmax = AC.duration, fmin = AC.cfs[1,2], fmax = AC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(DC, tmin = 0, tmax = DC.duration, fmin = DC.cfs[1,2], fmax = DC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(reduced_interference, tmin = 0, tmax = DC.duration, fmin = DC.cfs[1,2], fmax = DC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)


#@time ifrs, squeezed = ifreq_envelope(signal, fb.cfs, song.fs, .5, 3, 1)


ar_lags = 250
channel_components = 1
overall_components = 50
projection_period = 700
min_eigval = 0.1


tfd = permutedims(signal)
ri = permutedims(DC.channels)
@time bases, activations, comps = trackSubspaceLR(tfd; ri = ri, wl, ar_lags, step = step, overall_components, channel_components, min_eigval);
@time sigs, ifrs, amps = recreate_signal(bases, activations, comps, wl, ar_lags, .0025, fs);
@time linked_sinusoids, linked_freqs, linked_amps, times = link_comps(sigs, ifrs, amps, comps, wl, step, 100, size(tfd, 2), .9)

times

linked_freqs


@time points = envelopes(linked_sinusoids, linked_freqs, linked_amps, fs, .0025, 4, 2)
points[:, 3] .^= (1/2)
points[:, 3] ./= maximum(points[:, 3])

f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 25, color = :red)
f



smoothed = copy(linked_freqs)
@time aaa = freqStrand_tv_smoothed(smoothed, linked_amps, mag.(tfd), fb.cfs[:, 1], .0000005)


@time points = envelopes(linked_sinusoids, smoothed, linked_amps, fs, .0025, 4, 2)
points[:, 3] .^= (1/2)
points[:, 3] ./= maximum(points[:, 3])

f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 25, color = :red)
f






ifrs = copy(linked_freqs)

times = (1:size(linked_freqs, 1)) ./ fs
ifrs = windowAverages(linked_freqs, nothing, 20, 10)
amps = windowAverages(linked_amps, nothing, 20, 10)
ls = windowAverages(linked_sinusoids, nothing, 20, 10)
times = windowAverages(times, nothing, 20, 10)

@time points = envelopes(ls, ifrs, amps, fs/10, .0025, 4, 2)
points[:, 3] .^= (1/2)
points[:, 3] ./= maximum(points[:, 3])

f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 25, color = :red)

f



######## smoothing the frequency trajectories with a Kalman-like smoother
dlinked_freqs
linked_amps


freqSmoother = init_trajectory_smoother(plan, zeros(Float32, plan.p), zeros(Float32, plan.p); start = "linear")
ampSmoother = init_trajectory_smoother(plan, zeros(Float32, plan.p), zeros(Float32, plan.p); start = "linear")
freqData = linked_freqs[:, 2]
ampData = linked_amps[:, 2]

@time run_smoother!(freqSmoother, ampSmoother, plan, @view(freqData[:, 2]), @view(ampData[:, 2]))


freqSmoother.x
freqSmoother.self_confidence


freqSmoother.predictions
lines(freqSmoother.errors[5:end])
lines(freqSmoother.predictions[5:end])

lines(freqSmoother.self_confidence[5:end])


f = lines(freqData[freqSmoother.activation_start:(freqSmoother.activation_end - 25)])
lines!(freqSmoother.predictions[5:(freqSmoother.activation_end - 25)])
f


plan = plan_smoother(.0001, 10, .9, .9, 1, 20, 3)
freqTrackerBank =  [init_trajectory_smoother(plan, zeros(Float32, plan.p), zeros(Float32, plan.p); start = "linear") for i in axes(linked_freqs, 2)]
ampTrackerBank = [init_trajectory_smoother(plan, zeros(Float32, plan.p), zeros(Float32, plan.p); start = "linear") for i in axes(linked_freqs, 2)]
freqData = copy(linked_freqs)
ampData = copy(linked_amps)

Threads.@threads for j in axes(freqData, 2)
    println(j)
    
    run_smoother!(freqTrackerBank[j], ampTrackerBank[j], plan, @view(freqData[:, j]), @view(ampData[:, j]))
    
    #freqData[freqTrackerBank[j].activation_start:freqTrackerBank[j].activation_end, j] .= freqTrackerBank[j].predictions
    #ampData[ampTrackerBank[j].activation_start:ampTrackerBank[j].activation_end, j] .= ampTrackerBank[j].predictions
    #ampData[:, j] .= ampTrackerBank[j].predictions
end



f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)

for j in axes(freqData, 2)
    freqTrackerBank[j], ampTrackerBank[j]
    scatter!((freqTrackerBank[j].activation_start:freqTrackerBank[j].activation_end) ./ fs,
        freqTrackerBank[j].predictions, color = :red)#, markersize = points[:, 3] .* 25, color = :red)
end
f


#initialize tuning parameters, set initial energy and x 
#(the subset of data used for predicting the next time step)
#to zero, initialize autoreg coefficients to 1st order taylor 
#series (length of coefficient vector is the model order),
#set predictions and errors to empty linked list.
using Peaks

n = 40000
fs = 8000
s = 2.0 .* sin.((0:n)./fs .*(2π*60) ) .+ 
   # 1 .* sin.((0:n)./fs .*(2π*400)) .+ 
    #1.0 .* sin.((0:n)./fs .*(2π*350)) .+ 
    1.02 .* sin.((0:n)./fs .*(2π*67)) .+ 
   # sin.(10 .* sin.((0:n)./fs .*(2π*35))) .+ 
    rand(Normal(0, .005), n + 1) 
s = hilbert(s)


as = real.(s .* conj.(s))
lines(as)
lines(real.(s))

f=lines(real.(s[1:1000]))
lines!(imag.(s[1:1000]))
f



arPlan = plan_AR(.0001, 10.5, .5, 0, 1000, 5)

h = cispi.(-2*300/fs .* (arPlan.p:-1:1)) ./ arPlan.p
ar = init_AR(arPlan, h, signal_length = length(s))


trackerBankAmp = [init_ar(arPlan; start = "linear") for i in axes(linked_freqs, 2)]
trackerBankFreq = [init_ar(arPlan; start = "linear") for i in axes(linked_freqs, 2)]

Threads.@threads for k in axes(amp, 2)
    for t in axes(amp, 1)
        updateNLMS!(trackerBankAmp[k], arPlan, amp[t, k])
        updateNLMS!(trackerBankFreq[k], arPlan, amp[t, k])
    end
end


@time for t in eachindex(s)
    update_NLMS!(ar, arPlan, s[t], t, fs = fs, fmin = 50, fmax = 500, fstep = .5, threshold = .1)
end
ar.h

lines(real.(ar.past_h[:, 15010]))


arSpec = spectra_from_AR(1, ar.past_h[:, 24010], fs, 250, 500, .1)

lines(arSpec[1], arSpec[2])
arSpec[1][argmaxima(arSpec[2])]



@time ar.predictions[end]


f=lines(s[1:1000], linewidth = .5)
lines!(collect(ar.predictions)[1:1000])
f
lines(ar.h)


fb = createComplexGTFilterBank(40, 1000, song.fs, 200, .35, .016)
cgtSpectra = downsample(song, song.fs)
amp, phase = applyComplexFilterbank(song.channels, fb)
cgtSpectra.channels = amp
cgtSpectra.nchannels = size(cgtSpectra.channels, 2)
cgtSpectra.cfs = fb.cfs


windowLength = 200
overlap = 150

ifrs = matrixGradient(phase) .* (cgtSpectra.fs)
ifrs = windowAverages(ifrs, windowLength, overlap, true)
amp = windowAverages(cgtSpectra.channels, windowLength, overlap)
cgtSpectra.channels = amp
cgtSpectra.fs = (cgtSpectra.fs/(windowLength - overlap))
f = plotSpectrogram(cgtSpectra, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)

freqMovement = ifreqMovement(ifrs, amp, fb, cgtSpectra.fs, .00000007)

correctedSpec = downsample(cgtSpectra, cgtSpectra.fs)
correctedSpec.channels = ((freqMovement .^ 1.51) .* amp) .^ (.325)
plotSpectrogram(correctedSpec, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)

heatmap(amp)
heatmap(ifrs)


arPlan = ar_plan(.1, 10.5, .99, .1, 6)

trackerBankAmp = [init_ar(arPlan; start = "linear") for i in axes(amp, 2)]
trackerBankFreq = [init_ar(arPlan; start = "linear") for i in axes(amp, 2)]

Threads.@threads for k in axes(amp, 2)
    for t in axes(amp, 1)
        updateNLMS!(trackerBankAmp[k], arPlan, amp[t, k])
        updateNLMS!(trackerBankFreq[k], arPlan, amp[t, k])
    end
end

f = heatmap(amp)
tm = (0:size(amp, 1)) ./ cgtSpectra.fs
timeCoord = repeat(collect(tm), outer = size(ifrs, 2))

for k in axes(amp, 2)
    println(k)
    scatter!(tm, trackerBankFreq[k].predictions, markersize = trackerBankAmp[k].predictions .* 10)#, color = [(:red, alpha) for alpha in sizeCoord])

end
f


f=lines(s[1:1000], linewidth = .5)
lines!(collect(ar.predictions)[1:1000])
f






bank = createTrackerBank(.1, .9, .9, 4, 5)


a = [i*1.1 for i in 1:10000000]
@time popfirst!(a)
@time append!(a, 1)
@time a[end]
fill!(a, 1)
append!(a, 2)


