using CairoMakie 
cd("PolyphonicSignals")
using Revise



using PolyphonicSignals

fs = 40000
fr = createEnvelope([200, 300, 400, 600],  [10, -10, 10, -10], 20, 2, fs)
am = createEnvelope([100, 50, 80, 30],  [10, 5, 10, 5], 15, 2, fs)
song = createSignalFromInstFA(fr, am, fs)

fr = createEnvelope([1400],  [0], 20, 2, fs)
am = createEnvelope([100],  [10], 15, 2, fs)

song = createSignalFromInstFA(fr, am, fs)
plotAudio(song, 1, 0.0, .1)

#Short-time Fourier transform as a baseline for comparison
complexSTFT, frs, times = getSTFT(song.channels, 2048, .9, 5, song.fs)
magspec = getPowerSpectrum(complexSTFT)
magspec = magspec ./ maximum(magspec)
spec = AudioRecord([hzToErb.(frs) frs], transpose(magspec), round(1/(times[4]-times[3])), song.duration, length(frs))

plotSpectrogram(spec, tmin = 0, tmax = 1.8, fmin = 0, fmax = 2000, zScale = "linear", logY = false)
magspec = 0
GC.gc()

#testing the complex gammatone

fb = createComplexGTFilterBank(40, 1700, song.fs, 200, .15, .02)
plotFB(fb, -200, 0, 200, 2000, song.fs)

cgtSpectra = downsample(song, song.fs)
signal = applyComplexFilterbank(song.channels, fb, "signals")
cgtSpectra.channels = amp = mag.(signal)



cgtSpectra.nchannels = size(cgtSpectra.channels, 2)
cgtSpectra.cfs = fb.cfs
#complexAudioSplit = get_inst_power_exponential(cgtSpectra, .99)
plotSpectrogram(cgtSpectra, tmin = 0, tmax = cgtSpectra.duration, fmin = cgtSpectra.cfs[1,2], fmax = cgtSpectra.cfs[end, 2], zScale = "linear", logY = false)


################# testing on a real song ###########

fs = 30000
startT = 1
endT = 10
song = getSongMono("../porterRobinson_shelter.wav", startT, endT) #reading in a song's .WAV file
midEarHighPass!(song, .95)
song = downsample(song, fs)

fb = createComplexGTFilterBank(40, 2000, fs, 200, .3, .016);
plotFB(fb, -200, 0, 300, 2000, song.fs)

@time signal = applyComplexFilterbank(vec(song.channels), fb)
tfd = mag.(signal)
tfd ./= maximum(abs.(tfd))

ds = 100
ifrs = ifreq(signal, fs, 100)
squeezed = synchrosqueeze(permutedims(ifrs), tfd, fb.cfs, .0001)
spec = AudioRecord(fb.cfs, tfd[1:ds:end, :], fs / ds, song.duration, size(fb.cfs, 1))
squeezed = AudioRecord(fb.cfs, squeezed[1:ds:end, :], fs / ds, song.duration, size(fb.cfs, 1))
#peaks = AudioRecord(fb.cfs, peakmask .* squeezed.channels, fs * ds, song.duration, size(fb.cfs, 1))

plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "log", logY = false)
plotSpectrogram(squeezed, tmin = 0, tmax = squeezed.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "log", logY = false)
#plotSpectrogram(peaks, tmin = 0, tmax = squeezed.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)




################# testing on a real song ###########

using RollingFunctions
fs = 20000
song = downsample(getSongMono("../porterRobinson_shelter.wav", 1,5), fs, false)
midEarHighPass!(song, .95)

fb = createComplexGTFilterBank(40, 1500, fs, 100, .15, .016);
@time signal = applyComplexFilterbank(vec(song.channels), fb)
tfd = mag.(signal)
tfd ./= maximum(abs.(tfd))


n = 90000
fs = 30000
amp_ramprate = 0.001 #amplitude increase per sample for smooth envelope
amp_ramplength = 1000 #duration of ramp in samples


s = zeros(ComplexF32, n)
env1 = [(1:amp_ramplength) .* amp_ramprate; ones(Float32, n - amp_ramplength)]
s = cispi.((1:n)./fs .*(2*400) ) .+ 
    .5 * cispi.((1:n)./fs .*(2*800) .+ .25) 
   # sin.(10 .* sin.((0:n)./fs .*(2π*35))) .+ 
    #rand(Normal(0, .0005), n) 

@time signal = applyComplexFilterbank(vec(s), fb)
tfd = mag.(signal)
tfd ./= maximum(abs.(tfd))

ifrs =  ifreq(signal, fs) 
squeezed = synchrosqueeze(permutedims(ifrs), tfd, fb.cfs, .01)
#@time squeezed = synchrosqueeze_hpss(squeezed, 5)
peakmask = ispeak(squeezed, .01)
ifrs .*= peakmask


spec = AudioRecord(fb.cfs, tfd, fs, song.duration, size(fb.cfs, 1))
squeezed = AudioRecord(fb.cfs, squeezed, fs, song.duration, size(fb.cfs, 1))
peaks = AudioRecord(fb.cfs, peakmask .* squeezed.channels, fs, song.duration, size(fb.cfs, 1))
#peaks.channels = mapslices(x -> runmean(x, 200), peaks.channels, dims = 1)



#spec.channels = mapslices(x -> runmean(x, 300), spec.channels, dims = 1)
#ifrs.channels = mapslices(x -> runmean(x, 300), ifrs.channels, dims = 1)
plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(squeezed, tmin = 0, tmax = squeezed_harmonic.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(peaks, tmin = 0, tmax = squeezed_harmonic.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)
