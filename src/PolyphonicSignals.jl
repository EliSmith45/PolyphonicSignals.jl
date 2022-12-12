module PolyphonicSignals

#using Revise
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
#using CairoMakie



include("./TimeFrequencyTools.jl")
include("./GammatoneFilterBanks.jl")
include("./TQWaveletTransformFunctions.jl")
include("./MorphologicalComponentsFunctions.jl")
include("./AdaptiveSingularFilterBank.jl")

#export from time-frequency analysis module 
export getSongMono, compress_max!, downsample, clipAudio, createSignalFromInstFA, sampleCosine, 
    createEnvelope, midEarHighPass!, getSTFT, getPowerSpectrum, freq_reassignment_direction, 
    envelope_AC_DC, windowedSignal, plotSpectrogram, plotAudio, plotDFT, plotFB, plotStrandBook, 
    getAngle, windowAverages, mag, get_phase_shift, ifreq, mean_ifreq, norm_cols,  phase_align, 
    synchrosqueeze, synchrosqueeze_hpss, ispeak, hzToErb, erbToHz, AudioRecord



export create_gt_cascade_filterBank, applyFilterbank, getComplexGammatone, createComplexGTFilterBank, 
    applyComplexFilterbank, gt_complex_filterbank, meddis_IHC, getGTFreqResponse, erbWidth,
    getErbBins, biquadParam #gt_cascaded_filterbank




#export from TQWT module
export analysisFB!, synthesisFB!, tqwt!, itqwt!, tqwtParams, scale_lowpass, scale_lowpass!, 
    scale_highpass, scale_highpass!, daubechies_memoized, get_dft_sizes, get_sequence_sizes, 
    init_lowPassVec, init_highPassVec, init_coefficients, init_plan, getWavelets, daubechies_memoized, 
    nextPower2

#export from MCA module
export salsa

#export from AdaptiveSingularFilterBank
export ssa_windowed_peakchannels, ssa_windowed, mssa_windowed, link_comps, 
    envelopes, update_hankel!, link_comps_sparse, params_longformat, trackchannels, track_crosschannels

#import modules
using .TimeFrequencyTools
using .TQWT
using .MorphologicalComponents
using .GammatoneFilterBanks
using .AdaptiveSingularFilterBank



end
