using Revise

#=
using WAV
using DSP
using FFTW #fast fourier transform backend
using Peaks
using Distributions
using Statistics
using StatsBase
using DataStructures
using RollingFunctions
using Memoize
using CairoMakie
=#


includet("./timeFreqDists.jl")


using .timeFrequencyAnalysis
using .TQWT
using .MorphologicalComponents


### testing out some basic TFD visualization functionsfr = createEnvelope([200, 300, 400, 600],  [10, -10, 10, -10], 20, 2, 20000)


fr = createEnvelope([200, 300, 400, 600],  [10, -10, 10, -10], 20, 2, 20000)
am = createEnvelope([100, 50, 80, 30],  [10, 5, 10, 5], 15, 2, 20000)
song = createSignalFromInstFA(fr, am, 20000)
plotAudio(song, 1, 0.0, .1)


complexSTFT, frs, times = getSTFT(song.channels, 2048, .9, 5, song.fs)
mag = getPowerSpectrum(complexSTFT)
mag = mag ./ maximum(mag)
spec = audioRecord([hzToErb.(frs) frs], transpose(mag), round(1/(times[4]-times[3])), song.duration, length(frs))

plotSpectrogram(spec, tmin = 0, tmax = 1.8, fmin = 0, fmax = 1000, zScale = "linear", logY = false)
plotDFT(mag, frs, times, .1, 1000)


### testing the gammatone filter bank
gammatoneFB = create_gt_cascade_filterBank(40, 1000, song.fs, 100, .2)
plotFB(gammatoneFB, -200, 0, 200, 2000, song.fs)

spectra = applyFilterbank(song, gammatoneFB)
audioSplit = get_inst_power_exponential(spectra, .99)
plotSpectrogram(audioSplit, tmin = 0, tmax = 1, fmin = 0, fmax = 1000, zScale = "linear", logY = false)

gtSpectrum = get_inst_power_exponential(audioSplit, .5)
gtAveSpec = windowAverages(gtSpectrum, 200, 10)
spectraPeaks = amp_from_windowPeaks(spectra, 500, 497)
plotSpectrogram(gtSpectrum, tmin = 0, tmax = audioSplit.duration, fmin = 0, fmax = 1000, zScale = "linear", logY = false)
plotSpectrogram(gtAveSpec, tmin = 0, tmax = gtAveSpec.duration, fmin = 0, fmax = 1000, zScale = "linear", logY = false)
plotSpectrogram(spectraPeaks, tmin = 0, tmax = spectraPeaks.duration, fmin = spectraPeaks.cfs[1,2], fmax = 1000, zScale = "linear", logY = false)


### testing the Meddis IHC model####### Hair cell model #######
Cs = meddis_IHC(spectraPeaks;
m=1.0, #maximum storage of transmitter pool
a=2, #permeability average firing rate
b=300, #permeability firing rate compressor
g=2000, #permeability scale
y=8, #factory production rate
l=2500, #cleft loss rate
r=6580, #cleft reprocessing rate
x=66.31, #store release rate
h=50000 #firing rate scale
)

ihc = audioRecord(spectraPeaks.cfs, Cs, spectraPeaks.fs, spectraPeaks.duration, spectraPeaks.nchannels)
plotSpectrogram(ihc, tmin = 0, tmax = 1, fmin = 0, fmax = 1000, zScale = "linear", logY = false)


#synchrony strands - https://staffwww.dcs.shef.ac.uk/people/G.Brown/pdf/endeavour.
spectra = applyFilterbank(song, gammatoneFB)
spectraPeaks = amp_from_windowPeaks(spectra, 3000, 2000)
plotSpectrogram(spectraPeaks, tmin = 0, tmax = spectraPeaks.duration, fmin = spectraPeaks.cfs[1, 2], fmax = 1000, zScale = "linear", logY = false)
peakList = getPeakList(spectraPeaks, .05)
spectraPeaks.fs * .01
strandBook = connectStrands(peakList, 15, .02, .5)
plotStrandBook(strandBook, spectraPeaks; tmin = 0, tmax = spectraPeaks.duration, fmin = spectraPeaks.cfs[1, 2], fmax = 1000) 




### synchrony strands with a real song
song = getSongMono("audioProcessing/porterRobinson_shelter.wav", 0, 20)
plotAudio(song, 1, 0, .5)
midEarHighPass!(song, .95)
gammatoneFB = create_gt_cascade_filterBank(10, 2000, song.fs, 200, .25)
spectra = applyFilterbank(song, gammatoneFB)
#spectra = windowAverages(get_inst_power_exponential(spectra, .99), 500, 250)
spectra = amp_from_windowPeaks(spectra, 1000, 500)
plotSpectrogram(spectra, tmin = 0, tmax = spectra.duration, fmax = 2000, fticks = 10, tticks = 10, zScale = "linear", logY = false)


peakList = getPeakList(spectra, .01)
strandBook = connectStrands(peakList, 5, .02, .5)
plotStrandBook(strandBook, spectra; tmin = 0, tmax = spectra.duration, fmin = 10, fmax = 2000)






### fast TQWT testing

seq = collect(1:20000)
song = getSongMono("audioProcessing/porterRobinson_shelter.wav", 1, 2)
songDownsamp = downsample(song, song.fs/2)
seq = convert.(Float32, vec(songDownsamp.channels))
seq ./= sqrt(sum(seq.^2))

seq, p=init_plan(seq; Q = 5, r = 5, J = 80)

coeffs = init_coefficients(p)
highPass = init_highPassVec(p)
lowPass = init_lowPassVec(p)


@time tqwt!(coeffs; highPass=highPass, lowPass=lowPass, seq=seq, p=p)
@time signal = itqwt!(coeffs; highPass=highPass, lowPass=lowPass, p=p)




wv = getWavelets(length(seq); p = p, whichLevel = 81)
lines(wv[1:end])




#using MCA to remove transients6songDownsamp = downsample(song, song.fs/4)
using DSP
song = getSongMono("audioProcessing/porterRobinson_shelter.wav", 1, 20)
midEarHighPass!(song, .95)


wl=.5 #seconds
ovlp = .5 #proportion overlap


songDownsamp = downsample(song, song.fs/4)
resonant = downsample(songDownsamp, songDownsamp.fs)
transient = downsample(songDownsamp, songDownsamp.fs)
Y = convert.(Float32, vec(songDownsamp.channels))
Y ./= sqrt(sum(Y.^2))
wl *= div(songDownsamp.fs, 2)*2
wl = convert(Int32, round(wl))
ovlp *= wl
ovlp = convert(Int32, round(ovlp))

#solution vectors
Y1 = similar(Y)
Y2 = similar(Y)
Y1 .= 0
Y2 .= 0

windows = arraysplit(Y, wl, ovlp)
windowIndices = Matrix{Int32}(undef, length(windows[1]), length(windows))
wI = arraysplit(1:length(Y), wl, ovlp)

for j in axes(windowIndices, 2)
    windowIndices[:, j] = round.(Int32, wI[j])
end


weights = daubechies_memoized.(collect(ovlp:-1:1) ./ ovlp).^2
wgt = [weights ; reverse(weights)]
complete = 0
for wnd in axes(windowIndices, 2)
   
    if wnd == 1
        wgt = [ones(Float64, ovlp) ; reverse(weights)]
    elseif wnd == length(windows)
        wgt = [weights ; ones(Float64, ovlp) ]
    else 
        wgt = [weights ; reverse(weights)]
    end
 
    y = windows[wnd]
    samples = windowIndices[:, wnd]
    y, p1 = init_plan(y; Q = 5, r = 3, J = 80)
    y, p2 = init_plan(y; Q = 2.1, r = 3, J = 40)

    y1, y2, res = salsa(y; l1= .1, l2= .1, mu=.5, plan1 = p1, plan2 = p2, targetError = .002, maxIter=100, check_every = 10, printIter = false)

    for i in eachindex(windows[wnd])
        Y1[samples[i]] += (y1[i] * wgt[i])
        Y2[samples[i]] += (y2[i] * wgt[i])
    end
    complete += 1
    println(complete)
end


resonant.channels = Y1
transient.channels = Y2
resonant = downsample(resonant, resonant.fs*(4))
transient = downsample(transient, resonant.fs*4)

plotAudio(resonant, 1, 0, 10)
plotAudio(transient, 1, 0, 10)

yboth = Y1+Y2
error = Y - yboth
lines(Y1[20:1:end])
lines(Y2)
lines(yboth[1:100:end])
lines(error[1:100:end])
sum(abs.(Y .-Y1.-Y2))


gammatoneFB = create_gt_cascade_filterBank(40, 2000, song.fs, 200, .25)
harmonic = applyFilterbank(resonant, gammatoneFB)
harmonic = amp_from_windowPeaks(harmonic, 1000, 500)
plotSpectrogram(harmonic, tmin = 0, tmax = harmonic.duration, fmax = 1200, fticks = 10, tticks = 10, zScale = "linear", logY = false)


percussive = applyFilterbank(transient, gammatoneFB)
percussive = amp_from_windowPeaks(percussive, 1000, 500)
plotSpectrogram(percussive, tmin = 0, tmax = percussive.duration, fmax = 1200, fticks = 10, tticks = 10, zScale = "linear", logY = false)
