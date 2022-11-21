# AutomaticMusicTranscription

[![Build Status](https://github.com/EliSmith45/AutomaticMusicTranscription.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EliSmith45/AutomaticMusicTranscription.jl/actions/workflows/CI.yml?query=branch%3Amain)


This package contains a variety of modern signal processing algorithms as well as a few novel methods of mine. Most methods are fairly general but the intent is to facilitate analysis of polyphonic music. Includes functionality for calculation and visualisation of time-frequency distributions, generation of AM/FM signals, blind source separation, tracking the time-varying parameters of an unknown number of sinusoids, and grouping sources following a computational auditory scene analysis (CASA) framework. It will soon include methods for timbre learning and transfer, which will allow one to learn timbres from any audio recording and apply them to another (possibly in real-time at some point in the future. Finally, it will include music transcription functionality after the source separation methods are finalized.

*** WARNING: THIS PACKAGE IS IN EARLY DEVELOPMENT AND IS NOT FULLY FUNCTIONAL OR DOCUMENTED YET. MUCH OF THIS PACKAGE IS WORKING, BUT EXPECT STRUCTURAL CHANGES AND THE ADDITION OF MANY NEW ALGORITHIMS. ***



The overwhelming majority of signal processing algorithms fail in the presence of overlapping sinusiods, i.e., sinusoids with close but not identical frequencies. The nonlinearity of the amplitude of the sum of sinusoids makes them quite challenging to resolve. Unfortunately, polyphonic signals are rich in overlapping harmonics. Therefore, in order to develop algorithms for automatic music transcription or timbre learning from polyphonic audio, one must first resolve each harmonic, even with an unknown number of sources. This package implements a variety of modern methods, as well as a few of my own novel (and completely experimental) algorithms for source separation and superresolution time-frequency distributions. 

This package consists of five modules and a main parent module "PolyphonicSignals.jl." They are:

  1. TimeFrequencyTools.jl -- These are basic methods for calculating short-time Fourier transforms (STFTs),  visualizing time-frequency distributions (such as STFTs or any other filter bank), sampling from sinusoids with optional amplitude/frequency modulations, harmonics, and gaussian or random walk noise. In general, this is where all the simple utility methods are. 
  
  2. GammatoneFilterBanks.jl -- Two IIR implementations of the gammatone filterbank, which is modeled after the basilar membrane in the human cochlea. The first is the more common real-valued filter who's output is the real-valued filtered signal. The second is a more accurate perfect-reconstruction complex valued filter bank. It compensates for the phase shift and group delay, making it one of very few IIR filterbanks capable of (near) perfet reconstruction. Being complex valued, the output for each frequency channel is analytic, so you essentially get a free Hilbert transform when using it. As such, it may be used to calculate instantaneous frequencies and amplitude envelopes. Being an IIR filter, its output generally will have a much higher sample rate than a typical STFT, so the instantaneous frequency can be calculated much more reliably using finite differences on the phase values. Plus, it is extremely efficient, and the user has complete control of the filter's bandwidth and the number of/spacing of the filters. To read more about the real valued gammatone filterbank, see [1], and for the complex version, see [2].
  
  3. TQWaveletTransformFunctions.jl -- A highly performant implementation of the tunable Q wavelet transform using radix-2 FFTs, derived by Selesnick [3].
  
  4. MorphologicalComponentsFunctions.jl -- Implementation of morphological components analysis for resonance-based source separation [4]. I implemented this method to see if it could be useful for harmonic-percussive source separation, but alas, I haven't had much success with it at least for the parameters I've tried. However, I am leaving it in because it also implements the SALSA method for sparse L1 penalized optimization, which could have other uses down the road in this package. See the source paper [4] for more details. 
  
  5. AdaptiveSingularFilterBank.jl -- a novel method of mine for source separation and spectrogram superresolution. This is the most significant module in this package as its main algorithm can resolve any number of overlapping sinusoids and track their parameters overtime. 
  
  The main method 'trackSubspace' is based on singular spectrum analysis [5], with several significant modifications to improve its performance with musical signals. The method works by cleaning up the output of a complex valued filterbank such as a short-time Fourier transform or gammatone filterbank. It takes a windowed segment of each frequency channel whose energy is above some threshold, then performs singular spectrum analysis. That is, it performs an eigen decomposition on the covariance matrix of the windowed signal. The leading eigenvectors will be the exact frequency components, so they act like the columns in a Fourier basis, except with the exact frequency found in the signal. The leading eigenvectors are found for each channel and are then clustered, as neighboring channels will detect some of the same frequency components.

  The centroids of each cluster represent the true frequency components found in the signal and act like a new Fourier basis, except with exact instantaneous frequencies extracted from the signal. One then needs to express the signal as a linear combination of these basis vectors to decompose the signal. The Fourier transform does this with a dot product, which is prone to spectral leakage whenever the basis isn't truly orthogonal - which requires an infinitely long signal. To circumvent this issue, the new decomposition is found via optimization. Rather than solving the optimization problem for the original windowed signal, we continue to work with the windowed channels produced by the original filtering. 

For this, we assume that the signal is stationary throughout the window, and that each basis component will leak through to several channels. We can use the frequency response of the filters and the (optionally averaged) instantaneous frequency of each basis vector (calculated as the derivative of the phase, which is implemented in this package in a manner that doesn't require unwrapping) to map the true amplitude to that which is observed in each frequency channel. The optimization problem is then min ||Ax - y||₂ where y is the windowed filterbank output written as a tall vector y = [y1, y2, ..., yK]ᵀ for K frequency channels, where yₖ = [yₖ₁, yₖ₂, ..., yₖₘ] 
where m is the signal length. Each column of A is a basis vector found as before, but they are repeated K times and stacked vertically to match the dimension of y. Each duplicate corresponds to a filterbank channel and must be multiplied by the frequency response of that channel at the frequency represented by that basis. 

The matrix A will be quite tall - with a signal sample rate of 20kHz and a window size of 26ms (512 samples), the STFT will have 256 positive frequency channels and A will have 131,072 rows. If there are assumed (or allowed) to be 10 sources present simultaneously, A will have 1.3 million elements. However, most of A and y are irrelevent, so one should try to remove the meaningless components. 

For a given frequency, the gain for most channels is very small, so one may assume that the support of the gain of the basis column only covers a small number of nearby channels. This lets A be sparse. The signal y will also be negligible for most channels, so a simple thresholding step could allow one to eliminate most of the rows of A and y. In general, any channel that isn't in the support of any of the basis columns should be eliminated. Next, one may downsample the signal. To get precise estimates for the basis, one should estimate them from a high sample rate signal. Once estimated, each channel and eigenvector can then be appropriately downsampled, probably by a very large amount. The final way to shrink A is to use an initial TF distribution that uses fewer channels. An IIR gammatone filterbank with 50-100 logarithmically spaced frequency bins could be equally informative as a STFT. The IIR approach will also be faster and won't automatically downsample like the STFT with a step size larger than 1. The complex gammatone in general is superior to the STFT for this method for numerous other reasons. 

This algorithm is adaptive, meaning the basis is updated for each window. This could get quite expensive 
since SVD runs in O(n^3), and the algorithm requires many thousands of SVDs to be calculated. It therefore uses an extremely efficient approximate truncated SVD on a compressed version of the data formed by matrix sketching. Empirically I've found the results to be essentially identical to the typical SVD algorithm, but it runs many orders of magnitude faster. All other matrix operations are carried out using BLAS/LAPACK implementations from the LinearAlgebra package. In all I've found this algorithm to be significantly more performant than I thought it would be, running in a couple seconds to process ten seconds of audio with a sample rate of 20kHz on an intel i7 8th gen CPU. 

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  1. Slaney, Malcolm. An Efficient Implementation of the Patterson-Holdsworth Auditory Filter Bank.
  2. Chen, Zhangli and Hohmann, Volker. Online Monaural Speech Enhancement Based on Periodicity Analysis and A        Priori SNR Estimation. 
  3. Selesnick, Ivan W. Wavelet Transform with Tunable Q-Factor. 
  4. Selesnick, Ivan W. Resonance-Based Signal Decomposition: A New Sparsity-Enabled Signal Analysis
     Method
  5. Vautard, Yiou, Ghil. Singular-spectrum analysis: A toolkit for short, noisy chaotic signals
