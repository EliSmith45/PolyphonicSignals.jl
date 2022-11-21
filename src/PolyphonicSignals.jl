module AutomaticMusicTranscription

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
    createEnvelope, midEarHighPass!,  getPeakList, connectStrands, plotSpectrogram, plotAudio,
    plotDFT, plotFB, plotStrandBook, getAngle, windowAverages, audioRecord, 
    audStrand, getSTFT, getPowerSpectrum, amp_from_windowPeaks, get_inst_power_exponential, get_hilbert,
    ifreq_from_phase, freq_reassignment_direction, getRidgeTops, ridgeTop, 
    freq_resp_basis

export create_gt_cascade_filterBank, applyFilterbank, getComplexGammatone, createComplexGTFilterBank, 
    applyComplexFilterbank, gt_complex_filterbank, meddis_IHC, getGTFreqResponse, hzToErb, erbWidth, 
    erbToHz, getErbBins, gt_cascaded_filterbank



#export from TQWT module
export analysisFB!, synthesisFB!, tqwt!, itqwt!, tqwtParams, scale_lowpass, scale_lowpass!, 
    scale_highpass, scale_highpass!, daubechies_memoized, get_dft_sizes, get_sequence_sizes, 
    init_lowPassVec, init_highPassVec, init_coefficients, init_plan, getWavelets, daubechies_memoized, 
    nextPower2

#export from MCA module
export salsa

using .TimeFrequencyTools
using .TQWT
using .MorphologicalComponents
using .GammatoneFilterBanks
using .AdaptiveSingularFilterBank
using .TotalVariation


end
