using CairoMakie 
cd("PolyphonicSignals")
using Revise

using PolyphonicSignals



###### Singular spectrum analysis algorithms and complex gammatone #######
using Distributions, DSP 


n = 60000
fs = 30000
amp_ramprate = 0.001 #amplitude increase per sample for smooth envelope
amp_ramplength = 1000 #duration of ramp in samples


#s = zeros(ComplexF32, n)
#nv1 = [(1:amp_ramplength) .* amp_ramprate; ones(Float32, n - amp_ramplength)]
s = #cispi.((1:n)./fs .*(2*400) ) .+ 
     cospi.((1:n)./fs .*(2*90) .+ .35)  .+
    1 .* cospi.((1:n)./fs .*(2*92.5) .+ .9) #.+ 
     #cospi.((1:n)./fs .*(2*100) .+ 0) 
   # sin.(10 .* sin.((0:n)./fs .*(2π*35))) .+ 
    #rand(Normal(0, .0005), n) 
s = [zeros(2000) ; s]
#lines(imag.(s[1:1000]))
#lines(real.(s[1:1000]))



fb = createComplexGTFilterBank(70, 160, fs, 405, .15, .016)
@time signal = applyComplexFilterbank(s, fb, forwards_backwards = true);
amp = mag.(signal)
amp ./= maximum(amp)

sigdiff = diff(signal, dims=1)
sigdiff2 = [zeros(eltype(signal), length(s))  diff(signal, dims=2)]
sigdiff = abs.(real.(sigdiff2))
sigdiff ./= maximum(sigdiff)
heatmap(real.(abs.(sigdiff[100:10:end, :])))
#sigdiff2 ./=2

amSignal = copy(signal)
amSignal .= amp .+ (sigdiff2 .* im)
heatmap(imag.(sigdiff2[100:10:10000, :]))
heatmap(amp[100:10:10000, :])
heatmap(real.(mag.(amSignal[1:10:20000, :])))

heatmap(real.(signal[100:10:10000, :]))
heatmap(real.(sigdiff2[100:10:10000, :]))

heatmap(atan.(imag.(amSignal[1:10:40000, :]), real.(amSignal[1:10:40000, :])))

spec = AudioRecord(fb.cfs, amp, fs, (n)/fs, size(fb.cfs, 1))
f = plotSpectrogram(spec, tmin = 0, tmax = spec.duration, fmin = spec.cfs[1,2], fmax = spec.cfs[end,2], zScale = "linear", logY = false)


f=amp=spec=signal=fb=s=env1=0
GC.gc()
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

 






mu = .01
e = eps()
p = 5

minAmplitude = .0001 
ds = 2
saveEvery = 100

#@time predictions, errors, allCoefs = trackchannels(signal, fs; method = "ANFIR", plan = plan, ds = ds, saveEvery = 100);
@time predictions, errors = trackchannels(signal, fb.cfs, fs; mu, e, p, minAmplitude, ds, saveEvery);

pr = AudioRecord(fb.cfs, mag.(predictions), fs / ds, song.duration, size(fb.cfs, 1))
er = AudioRecord(fb.cfs, mag.(errors), fs / ds, song.duration, size(fb.cfs, 1))
plotSpectrogram(pr, tmin = 0, tmax = pr.duration, fmin = pr.cfs[1,2], fmax = pr.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(er, tmin = 0, tmax = er.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)


############# SSA with flux


mu = .1
e = eps()
p = 50
minAmplitude = .01
ds = 2
saveEvery = 100
@time predictions = trackchannels_flux(signal[10000:50000], fb.cfs, fs; mu, e, p, minAmplitude, ds = ds, wl = 100);

pr = AudioRecord(fb.cfs, mag.(predictions), fs / ds, song.duration, size(fb.cfs, 1))
er = AudioRecord(fb.cfs, mag.(errors), fs / ds, song.duration, size(fb.cfs, 1))
plotSpectrogram(pr, tmin = 0, tmax = pr.duration, fmin = pr.cfs[1,2], fmax = pr.cfs[end,2], zScale = "linear", logY = false)
plotSpectrogram(er, tmin = 0, tmax = er.duration, fmin = fb.cfs[1,2], fmax = fb.cfs[end,2], zScale = "linear", logY = false)


[Conv((1, p), 1 => 1; bias = false) for i in axes(tfd, 2)]
    


########## Flux deconvolution



    







@time  deconvolved, reconvolved, residual = track_crosschannels(signal[1:5000, :], fb.cfs, fs; mu, e, p, minAmplitude, ds = ds, wl = 100);

de = AudioRecord(fb.cfs, mag.(deconvolved), fs / ds, song.duration, size(fb.cfs, 1))
rec = AudioRecord(fb.cfs, mag.(reconvolved), fs / ds, song.duration, size(fb.cfs, 1))

residual = AudioRecord(fb.cfs, mag.(errors), fs / ds, song.duration, size(fb.cfs, 1))
plotSpectrogram(de, tmin = 0, tmax = pr.duration, fmin = pr.cfs[1,2], fmax = pr.cfs[end,2], zScale = "linear", logY = false)
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






##################### sinusoids and exponents ##############

fs = 10000.0
t0 = 0
te = 15
numDerivs = 5
times = collect((1 / fs):(1 / fs):te)

y1 = sin.(times)

yp = zeros(Float64, 2, length(times))
yv = copy(yp)
ya = copy(yp)
d = zeros(Float64, length(times))

yp[:, 1] = [1, -1]
yv[:, 1] = [0, 1]
ya[:, 1] = [-1, 0]

τ =  .000000005

for j in 2:size(yh, 2) 
 
   
    d[j - 1] = yp[1, j - 1] - yp[2, j - 1]
    ya[:, j] .= [-d[j - 1], d[j - 1]]
    yv[:, j] .= yv[:, j - 1] .+ ya[:, j] .* τ
    
    yp[:, j] .= yp[:, j] .+ yv[:, j] .* τ

    
    
    for i in 2:size(yh, 1)
        yh[i, j] = (yh[i - 1, j] * cis(r0))
    #yh[:, i] .= 1 .* sin.(@view(yh[:, i - 1]))
    #yh = yh[div(size(yh, 1), 2), :]
    end

end
yp
yv
ya

f= lines(yp[1, 1:10:10000])
lines!(yp[2, 1:100:end])
f

#yh[:, numDerivs] .= exp.(yh[:, numDerivs])
lines(d)
lines(yh[31400:1:31500, 2])
lines(yh[1:100:end, 3])
lines(yh[1:100:end, 4])

y1 .= exp.(y1) 
y1 .-= mean(y1)
#y1 ./= std(y1, corrected = false)
y1 ./= maximum(abs.(y1))
y1 .*= (-1) ^ (numDerivs) 

y2 = cos.(2 .* times)

yh
lines(y1[1:100:end])
lines(y2[1:100:end])

sum(abs.(y2 .- y1)) ./ length(y2)



f=lines(y1[1:end])
lines!(y2[1:end])
f


yh = zeros(Float64, length(times), numDerivs)

r0 = .5
r1 = .5
r2 = .5

yh[1, 1] = r0


for i in 2:size(yh, 1)
    yh[i, 1] = yh[i - 1, 1] * r1
  
end


for j in axes(yh, 2)
    for i in 2:size(yh, 1)
        yh[i, j] = yh[i - 1, j] * r1
      
    end
end



lines(yh[1:100])
lines(yhd[1:1000])


θ = 30.0 #time scaling, amplitude scaling, frequency, initial phase, sinusoid center
a = .5
b = 2.0

y1 = exp.(r .* times)
y2 = exp.(-1 .* y1 .* times)
y3 = exp.(-1 .* y2 .* times)
y4 = exp.(-1 .* y3 .* times)




lines(y1[1:end])
lines(y2[1:end])
lines(y3[1:end])
lines(y4[1:end])


f=lines(y1)
lines!(y2)
f


using CUDA, Flux

mutable struct Sinusoid
    θ
end
(m::Sinusoid)(times) = (-1 .* cos.(m.θ .* times)) .+ 2
Flux.@functor Sinusoid
MSE(x, y) = sum((x .- y).^2)


r = 20
fs = 1000.0
t0 = 0
te = .1



η = .01
θ = 30.0 #time scaling, amplitude scaling, frequency, initial phase, sinusoid center
a = .5
b = 2.0
times = collect(t0:(1 / fs):te)
y1 = exp.(r .* times)
lines(y1)
θ = cu(θ)
times = cu(times)
y1 = cu(y1)

m1 = Sinusoid(θ) 
m1 = fmap(cu, m1)
MSE = fmap(cu, MSE)

yn = cpu(y1)
yhat = cpu(m1(times ./ .7))
f=lines(cpu(y1))
lines!(yhat)
f




opt_state = Flux.setup(Adam(), m1)
reconstruct_loss(x, y) = sum((x .- y).^2)

@time reconstruct_loss(Ac, AcHat)

gr = gradient(m -> reconstruct_loss(m(Ac), Ac), m1)
Flux.update!(opt_state, m1, gr[1])


reconstruct_loss(m1(Ac), Ac)






