using CairoMakie 
cd("PolyphonicSignals")
using Revise



using PolyphonicSignals



###### Singular spectrum analysis algorithms and complex gammatone #######
using Distributions


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

lines(imag.(s[1:1000]))
lines(real.(s[1:1000]))



fb = createComplexGTFilterBank(200, 1000, fs, 100, .35, .016)
@time signal = applyComplexFilterbank(s, fb);
amp = mag.(signal)
spec = AudioRecord(fb.cfs, amp, fs, (n)/fs, size(fb.cfs, 1))
f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)


### single channel windowed SSA



wl = 10000 #window length
embeddingDim = 300 # length of each eigenvector (column of U)
step = 500 #step size
step <= (wl - 2*embeddingDim) #check that step size is small enough
min_eigval = .05 #smallest kept eigenvalue as a proportion of the largest
components = 50 #maximum kept eigenvectors
k = 10
similarity_threshold = .8 #minimum similarity to group two eigenvectors of adjacent windows
@time sigs, times = ssa_windowed(s; wl, embeddingDim, step, k, components, min_eigval, similarity_threshold)

points = params_longformat(sigs, times, fs, 100, 50)
f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 10, color = :red)
f

 



################# testing on a real song ###########

fs = 30000
startT = 1
endT = 10
song = getSongMono("../porterRobinson_shelter.wav", startT, endT) #reading in a song's .WAV file
midEarHighPass!(song, .95)
song = downsample(song, fs)

fb = createComplexGTFilterBank(40, 3000, fs, 200, .15, .016);
#plotFB(fb, -200, 0, 300, 2000, song.fs)

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


mu = .001
e = eps()
p = 50

minAmplitude = .0001 
ds = 2
saveEvery = 100

#@time predictions, errors, allCoefs = trackchannels(signal, fs; method = "ANFIR", plan = plan, ds = ds, saveEvery = 100);
@time predictions, errors = trackchannels(signal; mu, e, p, minAmplitude, ds, saveEvery);

pr = AudioRecord(fb.cfs, mag.(predictions), fs / ds, song.duration, size(fb.cfs, 1))
er = AudioRecord(fb.cfs, mag.(errors), fs / ds, song.duration, size(fb.cfs, 1))
plotSpectrogram(pr, tmin = 0, tmax = pr.duration, fmin = pr.cfs[1,2], fmax = pr.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(er, tmin = 0, tmax = er.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)






@time predictions, errors, coefs = track_crosschannels(signal, fs; method = "NLMS", mu, e, p, energySmoothing, similaritySmoothing, minAmplitude, ds = ds, saveEvery = 100);
coefs = track_crosschannels(signal, fs; mu, e, p, energySmoothing, similaritySmoothing, minAmplitude, ds = ds, saveEvery = 100);
heatmap(mag.(coefs))

a = sum(mag.(coefs), dims = 1)

pr = AudioRecord(fb.cfs, mag.(predictions), fs / ds, song.duration, size(fb.cfs, 1))
er = AudioRecord(fb.cfs, mag.(errors), fs / ds, song.duration, size(fb.cfs, 1))
plotSpectrogram(pr, tmin = 0, tmax = pr.duration, fmin = pr.cfs[1,2], fmax = pr.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(er, tmin = 0, tmax = er.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)











### old SSA stuff

##### modified multi-channel SSA #####
#This is a novel adaptation of the SSA algorithm. It first segements the time-frequency amplitude distribution using 
#the watershed transformation, which locates ridges. Normal SSA is then carried out on each ridge/segment to determine
#its instantaneous frequencies and amplitude envelopes


@time sigs, times = mssa_windowed(s; wl, embeddingDim, step, k, components, min_eigval, similarity_threshold)
points = params_longformat(sigs, times, fs, 100, 50)
f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 10, color = :red)
f





wl = 2000 #window length
embeddingDim = 500 # length of each eigenvector (column of U)
step = 750 #step size
step <= (wl - 2*embeddingDim) #check that step size is small enough
min_eigval = .05 #smallest kept eigenvalue as a proportion of the largest
components = 500 #maximum kept eigenvectors
k = 20
similarity_threshold = .8 #minimum similarity to group two eigenvectors of adjacent windows
@time sigs, times = ssa_windowed(song.channels; wl, embeddingDim, step, k, components, min_eigval, similarity_threshold)
collect(times[100]:length(sigs[100])) ./ fs
lines(real.(sigs[100]))
lines(ifreq(sigs[100], fs))

points = params_longformat(sigs, times, fs, 100, 50)
f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[:, 1], points[:, 2], markersize = points[:, 3] .* 10, color = :red)
f

spec = AudioRecord(fb.cfs, amp, fs, song.duration, size(fb.cfs, 1))
@time AC, DC = envelope_AC_DC(amp, wl, step)
reduced_interference = windowedSignal(spec, DC .- AC, step)
AC = windowedSignal(spec, AC, step)
DC = windowedSignal(spec, DC, step)


plotSpectrogram(AC, tmin = 0, tmax = AC.duration, fmin = AC.cfs[1,2], fmax = AC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(DC, tmin = 0, tmax = DC.duration, fmin = DC.cfs[1,2], fmax = DC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(reduced_interference, tmin = 0, tmax = DC.duration, fmin = DC.cfs[1,2], fmax = DC.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)


#@time ifrs, squeezed = ifreq_envelope(signal, fb.cfs, song.fs, .5, 3, 1)


embeddingDim = 250
channel_components = 1
components = 50
projection_period = 700
min_eigval = 0.1


tfd = permutedims(signal)
ri = permutedims(DC.channels)
@time bases, activations, comps = ssa_windowed_peakchannels(tfd; ri = ri, wl, embeddingDim, step = step, components, channel_components, min_eigval);
@time sigs, ifrs, amps = hankelize(bases, activations, comps, wl, embeddingDim, .0025, fs);
@time linkedSinusoids, linkedFreqs, linkedAmps = link_comps(sigs, ifrs, amps, comps, wl, step, 500, size(tfd, 2), .8);



ifreqStrands, ampStrands, times = get_component_params(linkedAmps, linkedFreqs, 0.0000000001, 1, 100, 50)
@time points =  params_longformat(ifreqStrands, ampStrands, times, fs)



points[:, 3] ./= maximum(points[:, 3])

f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
scatter!(points[1:5:end, 1], points[1:5:end, 2], markersize = points[1:5:end, 3] .* 25, color = :red)
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


















n = 90000
fs = 30000
amp_ramprate = 0.001 #amplitude increase per sample for smooth envelope
amp_ramplength = 1000 #duration of ramp in samples


s = zeros(Float32, n)
env1 = [(1:amp_ramplength) .* amp_ramprate; ones(Float32, n - amp_ramplength)]
s = cospi.((1:n)./fs .*(2*400) ) .+ 
    .5 * cospi.((1:n)./fs .*(2*500) .+ .25) 
   # sin.(10 .* sin.((0:n)./fs .*(2π*35))) .+ 
    #rand(Normal(0, .0005), n) 

fb = createComplexGTFilterBank(300, 600, fs, 50, .35, .016);
@time signal = applyComplexFilterbank(vec(s), fb)
tfd = mag.(signal)

tfd ./= maximum(abs.(tfd))


spec = AudioRecord(fb.cfs, tfd, fs, song.duration, size(fb.cfs, 1))
plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)

qr!(signal)
heatmap(tfd[1:50, :])


covmat = signal' * signal
partialCor = inv(covmat)
heatmap(mag.(covmat))
heatmap(mag.(partialCor))
decor = signal * conj.(partialCor)
decor = AudioRecord(fb.cfs, mag.(decor), fs, song.duration, size(fb.cfs, 1))
plotSpectrogram(decor, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)
