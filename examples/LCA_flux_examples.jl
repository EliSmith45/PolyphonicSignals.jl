using CairoMakie, Pkg
cd("PolyphonicSignals")
Pkg.activate(".")
using Revise

using PolyphonicSignals

######################## Locally competitive algorithm
using LinearAlgebra, DSP




function mergepeaks(x)
    xm = zeros(eltype(x), size(x))
    energyBetweenRuns = 0.0f0
   
    for j in axes(xm, 2)
        i = 0
        while i < size(x, 1)
        
        
            while (i < (size(x, 1))) &&  (x[i + 1, j] == 0)
                i += 1
            end

        # hillrun = flatrun
            k=0
            while  (i < (size(x, 1))) && (x[i + 1, j] > 0)
                i += 1
                k += 1
                energyBetweenRuns += x[i, j]
            end

            xm[i - div(k, 2), j] = energyBetweenRuns
            energyBetweenRuns = 0
        end
    end
    xm
end
function mergepeaks!(x)
    #xm = zeros(eltype(x), size(x))
    energyBetweenRuns = 0.0f0
   
    for j in axes(x, 2)
        i = 0
        while i < size(x, 1)
        
        
            while (i < (size(x, 1))) &&  (x[i + 1, j] == 0)
                i += 1
                x[i, j] = 0
            end

        # hillrun = flatrun
            k=0
            while  (i < (size(x, 1))) && (x[i + 1, j] > 0)
                i += 1
                k += 1
                energyBetweenRuns += x[i, j]
                x[i, j] = 0
            end

            x[i - div(k, 2), j] = energyBetweenRuns
            energyBetweenRuns = 0
        end
    end
    
end

len = 2.5
nPad = 100
fs = 30000



fb = createComplexGTFilterBank(100, 400, fs, 200, .5, .016)
fb.cfs = Float32.(fb.cfs)
f1 = fb.cfs[50, 2]
f2 = fb.cfs[150, 2]


fr = createEnvelope([f1, f2]; linModRate = [0.0f0, 0], modulationFreq = [0.0f0, 0], modulationAmp = [0.0f0, 0], errorRate = 0.0, duration = len, fs = fs)
am = createEnvelope([1.0f0, 1.0]; linModRate = [0.0f0, 0], modulationFreq = [0.0f0, 0], modulationAmp = [0.0f0, 0], errorRate = 0, duration = len, fs = fs)
s = createSignalFromInstFA(fr, am, fs)
#s = [zeros(eltype(s), nPad) ; s ; zeros(eltype(s), nPad) ; s ; zeros(eltype(s), nPad)]

ds = 10
@time signal = applyComplexFilterbank(s, fb; forwards_backwards = true);
amp = mag.(signal)


heatmap(amp[10000:ds:20010, :])
#heatmap(real.(signal[32000:ds:45000, 1:250]))



function freqresponses_complex(fb, fs; interp = 1)

    numFreqs = size(fb.filters, 1) * interp
    frs = erbToHz.(collect(range(fb.cfs[1, 1], fb.cfs[end, 1], length = numFreqs))) .* ((2 * π) / fs)
    freqr = zeros(ComplexF32, numFreqs, length(fb.filters))
    
    Threads.@threads for j in axes(freqr, 2)
        freqr[:, j] .= freqresp(fb.filters[j], frs)
        #freqr[j, j] = 0 
    end

    
    #phaseShifts = conj.(cispi.(frs))
    #frResp = mag2.(freqr)

    return mag2.(freqr) #permutedims(frResp)#, phaseShifts
end

D = freqresponses_complex(fb, fs, interp = 1)
#foreach(x -> normalize!(x), eachcol(D))
D[diagind(D)] .= 0
D .*= -1

l1 = size(D, 1)
n = 1000
st = 10000
x = permutedims(amp[st:(st + n - 1), :])
dnn = Lca1((l1, l1), n; W = D, λ = .03, τ = .1)
dnn.A


#dnn = Dnn((l1, l1), n; W = [[0.5 .* I, .5 .* I], [I, D]], λ = [.05, .05], τ = [.1, .1])
#dnn.A
#dnn(x)


iters = 100
@time for i in 1:iters
    dnn(x)
end

heatmap(x')
heatmap(dnn.A')

D1 = copy(D)
D1[diagind(D1)] .= -1
D1 .*= -1
#D[diagind(D)] .= -1

dnn2 = LiquidRnn12((l1, l1), n; W0 = D1, W1 = D, τ = 10, delta = .1)

dnn2.bw .= 1.0f0
dnn2.b .= 0.0000000f0
dnn2.W0 .*= 0
dnn2.W0[diagind(dnn2.W0)] .= 1
dnn2.W1
iters = 100
@time for i in 1:iters
    dnn2(x)
end

#heatmap(x')
heatmap(dnn2.U')
heatmap(dnn.A')
heatmap(x')

lines(x[:, 100])
lines(dnn2.U[:, 100])
lines(dnn.A[:, 100])





################# testing on a real song ###########

function harmonicDistances(f0, N = 5, B = .01, σ = 7)
    harmonics = [f0 * (k * sqrt(1 + B*k)) for k in 1:N]
    x -> maximum(exp.(-1 .* ( ((x .- harmonics) ./ σ) .^ 2)))
end

function harmonics(fb;  N = 5, B = .01, σ = 7)

    basis = zeros(eltype(fb.cfs), size(fb.cfs, 1), size(fb.cfs, 1))

    @views for j in axes(basis, 2)
        basis[:, j] .= harmonicDistances(fb.cfs[j, 2], N, B, σ).(fb.cfs[:, 2])

    end
    basis
end

fs = 8000
startT = .1
endT = 17
song = getSongMono("../porterRobinson_shelter.wav", startT, endT, fs) #reading in a song's .WAV file
#song2 = getSongMono("../porterRobinson_shelter.wav", startT, endT, fs) #reading in a song's .WAV file
#song2.channels = song2.channels[end:-1:1]
#midEarHighPass!(song, .95)

fb = createComplexGTFilterBank(40, 2800, fs, 500, .3, .016);
@time sig1 = applyComplexFilterbank(vec(song.channels), fb; forwards_backwards = true)
ds = 100
X1 = mag.(sig1)[1:ds:end, :]
heatmap((X1))
heatmap(sqrt.(X1))


D = freqresponses_complex(fb, fs, interp = 1)
D = sqrt.(D)
#foreach(x -> normalize!(x), eachcol(D))
D[diagind(D)] .= 0
D .*= -1

l1 = size(D, 1)
n = size(X1, 1)
x = sqrt.(X1')


dnn = Dnn((l1, l1), n; W = [[I, 0], [I, D]], λ = [.01, .01], τ = [.05, .05])
dnn.A
dnn(x)
smoother = hanning((5, 15))
smoother ./= sum(smoother)
dnn.U[1] .= DSP.conv(dnn.U[1], smoother)[3:(end - 2), 8:(end - 7)]
dnn.A[1] .= DSP.conv(dnn.A[1], smoother)[3:(end - 2), 8:(end - 7)]



iters = 110
@time for i in 1:iters
    dnn()
    if (i % 50) == 0
        mergepeaks!(dnn.A[2])
    end
    println(i)
end

heatmap(x')
heatmap((dnn.A[2]'))
aa = (dnn.A[2]') .^ 2


heatmap(1:size(aa, 1), fb.cfs[10:end, 2], aa[:, 10:end])
heatmap(1:size(aa, 1), fb.cfs[10:300, 2], aa[:, 10:300])

H = harmonics(fb, N = 6, B = 0.005, σ = [4, 5, 6, 7, 7, 7])
foreach(x -> normalize!(x), eachcol(H))










.32
A = (aa * H)
heatmap(A)







D = H' * H
#D = sqrt.(D)
#foreach(x -> normalize!(x), eachcol(D))
D[diagind(D)] .= 0
D .*= -1
#D = D[1:200, 1:200]


l1 = size(D, 1)
n = size(A, 1)
x = A'[1:end, :]


dnn2 = Dnn((l1, l1), n; W = [[I, 0], [I, D']], λ = [.02, .02], τ = [.05, .05])
dnn2.A
dnn2(x)
smoother = hanning((5, 15))
smoother ./= sum(smoother)
#dnn2.U[1] .= DSP.conv(dnn2.U[1], smoother)[3:(end - 2), 8:(end - 7)]
#dnn2.A[1] .= DSP.conv(dnn2.A[1], smoother)[3:(end - 2), 8:(end - 7)]



iters = 200
@time for i in 1:iters
    dnn2()
    println(i)
end


heatmap(x')
heatmap((dnn2.A[2]')[:, 1:end] .^ .5)
aa = (dnn2.A[2]') .^ 2
aa ./= maximum(aa)
heatmap(1:size(aa, 1), fb.cfs[1:end, 2], aa[:, 1:end])
heatmap(aa[:, 10:end] .^ .2)



aa = (dnn.A[2]') .^ 2
aa ./= maximum(aa)
heatmap(1:size(aa, 1), fb.cfs[10:end, 2], aa[:, 10:end])





fb2 = createComplexGTFilterBank(40, 1600, fs, 500, .9, .016);
@time sig2 = applyComplexFilterbank(vec(song.channels), fb2; forwards_backwards = true)
X2 = reshape([X1 X1 X1 X1], 530, 500, 4, 1)

X2[:, :, 1, 1] .= real.(sig1[1:ds:end, :])
X2[:, :, 2, 1] .= imag.(sig1[1:ds:end, :])
X2[:, :, 3, 1] .= real.(sig2[1:ds:end, :])
X2[:, :, 4, 1] .= imag.(sig2[1:ds:end, :])
Y2 = model(cu(X2)) |> cpu
heatmap(Y2[:, :, 1, 1])

X1

G = mag.(freqresponses(fb, fs)) 



τ = .01
threshold = .02
iters = 10



#@time A1 = lca_iir(X1, G; minWeight = .0001, τ = τ, threshold = threshold, iters = iters, save_every = 1);
@time A2 = lca_iir2(X1, G; τ = τ, threshold = threshold, iters = iters);
heatmap(X1)
#heatmap(A1)
heatmap(A2)



τ = .01
threshold = .03
iters = 15
ds=10

Y = sig1[4000:ds:25100, :]
G2 = freqresponses(fb, fs)
#G2 = permutedims(G2)
phaseShifts = cispi.(fb.cfs[:, 2] .* 2 .* (ds/fs))



@time A3, residual = lca_iir(Y, G2, phaseShifts; τ = τ, threshold = threshold, iters = iters);
A3
heatmap(A3)
heatmap(permutedims(residual))
heatmap(mag.(Y))
residual
mag.(Y)

A3=A2=A1=G=G2=phaseShifts=song=residual=Y=X1=sig1=0
GC.gc()
-