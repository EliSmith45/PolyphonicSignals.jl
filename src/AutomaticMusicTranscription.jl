module AutomaticMusicTranscription


include("./timeFrequencyAnalysisFunctions.jl")
include("./tunableQWaveletTransformFunctions.jl")
include("./MorphologicalComponentsFunctions.jl")

#export from time-frequency analysis module 
export getSongMono, compress_max!, downsample, clipAudio, createSignalFromInstFA, sampleCosine, 
    createEnvelope, midEarHighPass!,  getPeakList, connectStrands, create_gt_cascade_filterBank,
    applyFilterbank, meddis_IHC, hzToErb, erbWidth, erbToHz, getErbBins, plotSpectrogram, plotAudio
    plotDFT, plotFB, plotStrandBook, getAngle, windowAverages, gt_cascaded_filterbank, audioRecord, 
    audStrand



#export from TQWT module
export analysisFB!, synthesisFB!, tqwt!, itqwt!, tqwtParams, scale_lowpass, scale_lowpass!, 
    scale_highpass, scale_highpass!, daubechies_memoized, get_dft_sizes, get_sequence_sizes, 
    init_lowPassVec, init_highPassVec, init_coefficients, init_plan, getWavelets, daubechies_memoized, 
    nextPower2

#export from MCA module
export salsa

using .timeFrequencyAnalysis
using .TQWT
using .MorphologicalComponents



end
