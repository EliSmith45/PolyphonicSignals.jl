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

 



######################## Locally competitive algorithm
using LinearAlgebra, DSP



len = .5
nPad = 10000
fs = 30000



fr = createEnvelope([40, 100],  [0, 0], 0, len, fs)
am = createEnvelope([1, 1.5],  [0, 0], 0, len, fs)
s = createSignalFromInstFA(fr, am, fs).channels
s = [zeros(eltype(s), nPad) ; s ; zeros(eltype(s), nPad) ; s ; zeros(eltype(s), nPad)]


fr = createEnvelope([200], [0], 0, len, fs)
am = createEnvelope([1.5],  [0], 0, len, fs)
s2 = createSignalFromInstFA(fr, am, fs).channels
nPad2 = length(s) - length(s2)
s2 = [zeros(eltype(s), nPad2) ; s2]

s .+= s2







fb = createComplexGTFilterBank(10, 500, fs, 500, .5, .016)
@time signal = applyComplexFilterbank(s, fb; forwards_backwards = true);
amp = mag.(signal)
ds = 10
heatmap(amp[12000:ds:end, :])
heatmap(real.(signal[32000:ds:45000, 1:250]))



τ = .04
threshold = .03
iters = 35
ds=5

Y = signal[38000:ds:44000, :]
G2 = mag2.(freqresponses(fb, fs))
phaseShifts = conj.(cispi.(fb.cfs[:, 2] .* 2 .* (ds/fs)))
#heatmap(real.(G2))



@time A3, residual = lca_iir(Y, G2, phaseShifts; τ = τ, threshold = threshold, iters = iters, save_every = 5, delta = .0001);
A3
heatmap(A3)
heatmap(mag.(Y))
lines(residual)
maximum(residual)

heatmap(mag.(Y))
residual
mag.(Y)



####### solving LCA with ODE package
using OrdinaryDiffEq

function lca_diffeq!(du, u, p, t)

    p[4] = findall(x -> x > p[1], u)
    du .=  p[2] .- u .- (p[3][:, p[4]] * u[p[4]])

end

threshold = .030f0
ds = 10
fs = 30000

fr = createEnvelope([40, 100],  [0, 0], 0, len, fs)
am = createEnvelope([1, 1.5],  [0, 0], 0, len, fs)
s = createSignalFromInstFA(fr, am, fs).channels
s = [zeros(eltype(s), nPad) ; s ; zeros(eltype(s), nPad) ; s ; zeros(eltype(s), nPad)]
fb = createComplexGTFilterBank(10, 200, fs, 400, .5, .016)
@time signal = applyComplexFilterbank(s, fb; forwards_backwards = true);

Y = signal[38000:ds:44000, :]
heatmap(mag.(Y))

G2 = mag2.(freqresponses(fb, fs))
G2[diagind(G2)] .= 0
#phaseShifts = conj.(cispi.(fb.cfs[:, 2] .* 2 .* (ds/fs)))
data = mag.(Y[120, :])
data ./= maximum(data)
lines(G2[:, 30])
lines(data)


u0 = zeros(Float32, size(fb.cfs, 1))
p = [.03, data, G2, []]
tspan = (15.0, 105.20)
prob = ODEProblem(lca_diffeq!, u0, tspan, p)
@time sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveevery = 20);

us = sol(105)
k = zeros(eltype(us), size(us, 1)) 
k[p[4]] .= 1
#maximum(us)
lines(us .* k)


active = findall(x -> x > 200.0, [1.1, 1.2, 3.0, 4.0])
p[3][:, active] * u[active]


a
#=


fs = 40000
len = .5
fr = createEnvelope([200, 220], [0, 0], 0, len, fs)
am = createEnvelope([1.5, 1.5],  [0, 0], 0, len, fs)
s = createSignalFromInstFA(fr, am, fs).channels
s = [zeros(eltype(s), 5000) ; s]


fr = createEnvelope([250], [0], 0, len, fs)
am = createEnvelope([1.5],  [0], 0, len, fs)
s2 = createSignalFromInstFA(fr, am, fs).channels
s2 = [s2; zeros(eltype(s2), 5000)]

s .+= s2

#percussive s = [zeros(eltype(s2), 15000) ; ones(eltype(s2), 1000) .* 0.99 .^ (1:1000); zeros(eltype(s2), 15000)]
s = [zeros(eltype(s2), 15000) ; cos.((1:25000) ./ 30) .* .998 .^ (1:25000); zeros(eltype(s2), 15000)]

lines(s[15000:19000])

fb = createComplexGTFilterBank(100, 500, fs, 500, .5, .016)
@time signal = applyComplexFilterbank(s, fb; forwards_backwards = true);
amp = mag.(signal)

ds = 10
st = Int(round(size(signal, 1) * .2))
en = Int(round(size(signal, 1) * .4))

heatmap(real.(signal[st:ds:en, :]))
heatmap(amp[st:ds:en, :])


A3=A2=A1=G=G2=phaseShifts=song=residual=Y=X1=sig1=0
GC.gc()

=#

fr=am=s=signal=amp=X=s2=0
GC.gc()









################# testing on a real song ###########

fs = 8000
startT = 1
endT = 7
song = getSongMono("../porterRobinson_shelter.wav", startT, endT, fs) #reading in a song's .WAV file
#song2 = getSongMono("../porterRobinson_shelter.wav", startT, endT, fs) #reading in a song's .WAV file
#song2.channels = song2.channels[end:-1:1]
#midEarHighPass!(song, .95)

fb = createComplexGTFilterBank(40, 1600, fs, 500, .3, .016);
@time sig1 = applyComplexFilterbank(vec(song.channels), fb; forwards_backwards = true)
ds = 100
X1 = mag.(sig1)[1:ds:end, :]
heatmap(X1)


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



ds = 200
τ = .05
threshold = .02
iters = 25

Y = sig1[1:ds:end, :]
G2 = freqresponses(fb, fs)
#G2 = permutedims(G2)
phaseShifts = cispi.(fb.cfs[:, 2] .* 2 .* (ds/fs))
#G2 = conj.(G2)

@time A3 = lca_iir2(Y, G2, phaseShifts; τ = τ, threshold = threshold, iters = iters);
A3 = mag.(A3)
heatmap(mag.(Y))
#heatmap(A2)
heatmap(A3)





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






############Flux stuff ###########
using Flux
using CUDA

fs = 8000
fs_envelope = 500
duration = 2
fmin = 40
fmax = 1600

fb = createComplexGTFilterBank(fmin, fmax, fs, 500, .4, .016);

function makeTrainingSet(fb1, fb2; fs = 8000,
                        fs_envelope = 800,
                        duration = 2,
                        numSamples = 10,
                        
                        noise = .1,
                        comps_per_sample = 5)

    nFilts = size(fb1.cfs, 1)

    compression = div(fs, fs_envelope)
    fs_envelope = div(fs, compression)
    #return fs_envelope, compression
    xTrain = zeros(Float32, Int64(round(fs * duration)), size(fb1.cfs, 1), 4, numSamples)
    yTrain = zeros(Float32,  round(Int64(fs * duration)), size(fb1.cfs, 1) , numSamples)

    temp = zeros(ComplexF32, round(Int64(fs * duration)), size(fb1.cfs, 1))
    #gains = zeros(Float32, comps_per_sample)

    step = div(nFilts, comps_per_sample)
    freqInd = ones(Int64, comps_per_sample) 
    freqs = Float32.(fb1.cfs[freqInd, 2]) #* 2 * π / fs
    #freqSweep = 1 .* freqs .* rand(-.10f0:.010f0:.150f0, length(freqs))
    freqEnv = zeros(Float32, size(yTrain, 1), comps_per_sample)
    ampEnv = zeros(Float32, size(yTrain, 1), comps_per_sample)

    amps = Float32.(rand(.1:.01:1, length(freqs)))
    #ampSweep = Float32.(amps .* rand(-.5:.01:.5, length(freqs)))
    #ampEnv = Float32.(createEnvelope(amps, ampSweep, 0.1, duration, fs_envelope))

    startTime = ones(Int64, size(amps, 1), 4)
    endTime = ones(Int64, size(amps, 1), 4)
    s = zeros(Float32, size(yTrain, 1))
    

    #return freqInd
    for m in 1:numSamples
        println(m)

        amps .= rand(.10f0:.010f0:10f0, length(freqs))
        startTime .= rand(collect(1000:(size(yTrain, 1) - 15000)), length(freqs))
        amps .= rand(.10f0:.010f0:10f0, length(freqs))
       
        for g in 1:(size(freqInd, 1) - 1)
           
            freqInd[g] = rand(collect(((g - 1)*step + 1):(g*step)))
            freqs[g] = Float32.(fb1.cfs[freqInd[g], 2])
        

            freqEnv[:, g] .= freqs[g]
            endTime[g] = rand(collect((startTime[g] + 5000):(size(yTrain, 1) - 5000)))
            ampEnv[startTime[g]:endTime[g], g] .= amps[g] #.+ ampSweep[g]*(1:(1 +endTime[g] - startTime[g]))
            startTime[g] = rand(collect((endTime[g] + 800):(size(yTrain, 1) - 200)))
            endTime[g] = rand(collect((startTime[g] + 100):(size(yTrain, 1) - 1)))
            
            amps[g] = rand(.10f0:.010f0:10f0)
            ampEnv[startTime[g]:endTime[g], g] .= amps[g] #.+ ampSweep[g]*(1:(1 +endTime[g] - startTime[g]))
           
           
        end

        s .= createSignalFromInstFA(freqEnv, ampEnv, fs).channels
        s .+= rand(Normal(0, noise), size(s, 1)) 
        

        Threads.@threads for j in axes(yTrain, 2) 

            for k in axes(ampEnv, 2)
                      
                yTrain[:, j, m] .+= (@view(ampEnv[:, k]) .* mag2.(DSP.freqresp(fb1.filters[j], (2 * π / fs) .* freqs[k])))
                   
      
            end
        end

        #xTrain[:, :, i] .= tfd
        temp .= applyComplexFilterbank(s, fb1; forwards_backwards = true)
        xTrain[:, :, 1, m] .= real.(temp)
        xTrain[:, :, 2, m] .= imag.(temp)

        temp .=  applyComplexFilterbank(s, fb2; forwards_backwards = true)
        xTrain[:, :, 3, m] .= real.(temp)
        xTrain[:, :, 4, m] .= imag.(temp)
       
        #return xTrain, yTrain, freqs
        
    end

   

    return xTrain, yTrain#, yTrain
  
end


function getPatches(x, y; patchHeight = 499, patchWidth = size(x, 2), N = 100, startBuffer = 500, endBuffer = size(x, 1) - patchHeight - 1)
    xTrain = zeros(eltype(x), patchHeight, patchWidth, 4, N)
    yTrain = zeros(eltype(x), patchHeight, patchWidth, 1, N)

    start = 1
    whichLevel = 1

    for n in 1:N
        start = rand(startBuffer:endBuffer)
        whichLevel = rand(1:size(x, 4))
        yTrain[:, :, 1, n] .= @view(y[start:(start + patchHeight - 1), :, whichLevel])
    
        for q in 1:4
            xTrain[:, :, q, n] .= @view(x[start:(start + patchHeight - 1), :, q, whichLevel])
        end
    end

    return xTrain, yTrain
end


fs = 8000
fb1 = createComplexGTFilterBank(fmin, fmax, fs, 200, .3, .016);
fb2 = createComplexGTFilterBank(fmin, fmax, fs, 200, .95, .016);

@time x, y = makeTrainingSet(fb1, fb2; fs = fs,
                        fs_envelope = fs_envelope,
                        duration = 4,
                        numSamples = 3,
                        noise = .2,
                        comps_per_sample = 10);

#y = Float32.(DSP.resample(y, fs/fs_envelope; dims = 1))
#y[y .<= 0] .= 0

heatmap(y[1:10:end, :, 1])
heatmap(sqrt.(x[1:10:end, :, 4, 1] .^2 .+ x[1:10:end, :, 3, 1].^2))
heatmap(y[1:10:end, :, 2])
heatmap(sqrt.(x[1:10:end, :, 1, 2] .^2 .+ x[1:10:end, :, 2, 2].^2))

@time xp, yp = getPatches(x, y, patchHeight = 500, N = 500);
x=y=0
GC.gc(true)

heatmap(xp[:, :, 1, 1])
yp[:, :, 1, 1]

l = 60
heatmap(xp[:, :, 3, l])
heatmap(sqrt.(xp[1:10:end, :, 3, l] .^2 .+ xp[1:10:end, :, 4, l].^2))




### getting fancy here

b0 = BatchNorm(4, relu)
c0 = Conv((1, 1), 4 => 2, tanh; pad = SamePad(), stride = (1, 1), dilation = 1)
c1 = DepthwiseConv((1, 5), 2 => 4, tanh; pad = SamePad(), stride = (1, 1), dilation = 1)
c2 = DepthwiseConv((1, 5), 4 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 2))

c3 = DepthwiseConv((1, 5), 16 => 32, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 2))
b3 = BatchNorm(32, relu)

c4 = DepthwiseConv((1, 7), 32 => 64, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 2))
b4 = BatchNorm(64, relu)

#c4 = DepthwiseConv((1, 9), 32 => 32; pad = SamePad(), stride = (1, 1), dilation = (1, 8))

c5 = Conv((15, 1), 64 => 32, relu; pad = SamePad(), stride = (1, 1), dilation = (8, 1))
c6 = Conv((11, 1), 32 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (4, 1))
c7 = Conv((7, 1), 16 => 8, relu; pad = SamePad(), stride = (1, 1), dilation = (2,1))
c8 = Conv((5, 1), 8 => 1, relu; pad = SamePad(), stride = (1, 1), dilation = 1)

b6 = BatchNorm(16, relu)



CUDA.memory_status()  
GC.gc(true)
CUDA.reclaim()   
#CUDA.unsafe_free!(x1,y1,xhat,yhat,x,y)



model = Chain(Chain(b0, c0, c1, c2, c3, b3, c4, b4), Chain(c5, c6, b6, c7, c8)) |> gpu

function trainConvCleaner!(model, data; epochs, bs = 5, adamStep = .005, log_every = 2, reltol = 1e-5, decay = .99, maxiter = 500)
    ps = Flux.params(model)
    train_loader = Flux.DataLoader(data, batchsize=bs, shuffle=true)
    opt = Adam(adamStep)

    errors = zeros(Float32, div(epochs * length(train_loader), log_every), 4)
    errors[:, 1] .= (1:size(errors, 1)).- 1
    errors[1, 2:4] .= [1 ; 1.00001f0 ; 1.000001f0]
    
    counter = 0
    logged = 0
    for epoch in 1:epochs
        println("****** EPOCH ", epoch, " ******\n")
        println("Step number: ", "\t Absolute error: ", "\t Relative Error: ",  "\t Error rate of reduction: ")
               
        println(" ")
        println(" ")
        for (xbatch, ybatch) in train_loader
            counter += 1

            
            if (counter % log_every) == 0
                logged += 1
                xbatch, ybatch = xbatch |> gpu, ybatch |> gpu
                gr = Flux.withgradient(() -> Flux.mse(model(xbatch), ybatch), ps)
                Flux.update!(opt, ps, gr[2])
                errors[logged, 2] = (1 - decay) * gr[1]

                if logged == 1
                    errors[logged, 2] = gr[1]
                    errors[logged, 3:4] .= [1.0f0, 1.0f0]
                   
                elseif (logged == 2) 
                    errors[logged, 2] = decay * errors[logged - 1, 2] + (1 - decay) * gr[1]
                    errors[logged, 3] = gr[1] / errors[logged - 1, 2]
                    errors[logged, 4] =  errors[logged, 3] 
                    
                else
                    errors[logged, 2] = decay * errors[logged - 1, 2] + (1 - decay) * gr[1]
                    errors[logged, 3] = gr[1] / errors[logged - 1, 2]
                    errors[logged, 4] =  decay * errors[logged - 1, 4] + ((1 - decay) * errors[logged, 3])
                   
                end

                if (abs(errors[logged, 4] - 1 ) < reltol) && (counter > 50)
                  
                  #  errors[1:2, :] .= 0
                    return errors[3:logged, :]
                end

                

                errors .= round.(errors, digits = 7)
                println("  ", counter, "\t\t", gr[1], "\t\t ", errors[logged, 3], "\t\t\t", errors[logged, 4])
                

                #println("Step number: ", "\t Absolute error: ", "\t Relative Error: ",  errors[logged, 3], "\t Error rate of reduction: ")
                #println("  ", counter, "\t", gr[1], "\t", errors[logged, 3], "\t", errors[logged, 4])
                println(" ")


        
            else
                
                xbatch, ybatch = xbatch |> gpu, ybatch |> gpu
                gr = gradient(() -> Flux.mse(model(xbatch), ybatch), ps)
                Flux.update!(opt, ps, gr)
            end

            if counter > maxiter
               
                #errors[1:2, 3:4] .= 0
                return errors[3:logged, :]
            end

        end
    
        
    end
    
    errors[1:2, 3:4] .= 0
    return errors[1:logged, :]

end

       

errors = trainConvCleaner!(model, (xp, yp); epochs = 20,
    bs = 3, 
    adamStep = .002, 
    log_every = 10, 
    reltol = 1e-12, 
    decay = .99, 
    maxiter = 1050)


lines(errors[:, 1], errors[:, 2])
lines(errors[:, 1], errors[:, 3])
lines(errors[:, 1], errors[:, 4])

#mcpu = copy(model) |> cpu

l = 71
xt = xp[:, :, :, l]
yt = yp[:, :, 1, l]

xt = reshape(xt, size(xt)..., :)
yhat = model(cu(xt))  #, cu(yp)
m1 = model |> cpu

heatmap(cpu(sqrt.(xt[:, :, 1, 1] .^2 .+ xt[:, :, 2, 1] .^2)))
heatmap(cpu(yhat[:, :, 1, 1]))
heatmap(cpu(yt))




####### recurrent NN
rnn = Flux.RNNCell(2, 5)

x = rand(Float32, 2) # dummy data
h = rand(Float32, 5)  # initial hidden state

h, y = rnn(h, x)
rnn





blasdfl






















############### getting frequency channel onset kernels

### This segment only shows that the filters are spatially invariant and isn't necessary to run
#=
function get_channel_onset_bases(fb, fs; wl = 100, skip = 5)

    channels = div(size(fb.cfs, 1), skip)
    #return channels
    b = zeros(ComplexF32, 2 * wl, size(fb.cfs, 1), channels)
    s = zeros(Float64, (2 * 20000) + (2 * fs))
    s2 = zeros(Float64, (2 * fs))
    fr = zeros(Float64, size(s, 1))
    am = zeros(Float64, size(s, 1))

    signal = zeros(ComplexF32, size(s, 1), size(fb.cfs, 1))


    #return b
    for i in 1:channels
        
        fr = createEnvelope([fb.cfs[1 + (skip * (i - 1)), 2]],  [0], 0, 2, fs)
        am = createEnvelope([1.0],  [0], 0, 2, fs)
        s2 = createSignalFromInstFA(fr, am, fs).channels
        s[20001:(20000 + (2 * fs ))] .=  s2 
        signal = applyComplexFilterbank(s, fb; forwards_backwards = true);

        #if i == 5
         #   return signal
        #end
        b[:, :, i] .= signal[(20000 - wl):(20000 + wl - 1), :]
    end

    return b
end

fb = createComplexGTFilterBank(50, 500, fs, 150, .3, .016)
g = get_channel_onset_bases(fb, fs; wl = 9500, skip = 5)
#heatmap(mag.(g[18000:100:22000, :]))
bs1 = mag.(g[:, :, 7])
bs2 = imag.(g[:, :, 7])
heatmap(bs1[1:100:end, :])
heatmap(bs2[1:100:end, :])



f = lines(bs1[1, :])
for j in 1:100:(size(bs1, 1) - 1000)
    lines!(bs1[j, :])
end
f

f = lines(bs1[1500, :])
lines!(bs1[2000, :])
f
b1 = bs1[1500, :]
b2 = bs2[1500, :]

f = lines(b1)
lines!(b2)
f

b1 = b1[1:(end - 55)]
b2 = b2[56:end]


#Spatial invariance! Thus, the shape of the convolutional kernel for frequency channel onset detection is the same for each channel,
#with only its position varying
f = lines(b1)
lines!(b2)
f

=#

#####################################################



# Now we want to train a neural network that parameterizes a continuous kernel.
# It is a feedforward network with two inputs - the distance from the center of the kernel
# in the frequency and time domains. The input features aren't sample numbers, but true frequency
# and time values, which will hopefully make this approach completely robust to different choices 
# of sampling rate and center frequencies for each channel. 

function channel_onset_kernel(fb, fs; wl = 1500)

    channel = div(size(fb.cfs, 1), 2)
    s = zeros(Float64, (2 * 20000) + (2 * fs))


        
    fr = createEnvelope([fb.cfs[channel, 2]],  [0], 0, 2, fs)
    am = createEnvelope([1.0],  [0], 0, 2, fs)
    s2 = createSignalFromInstFA(fr, am, fs).channels
    s[20001:(20000 + (2 * fs ))] .=  s2 
    signal = applyComplexFilterbank(s, fb; forwards_backwards = true);
    signal ./= maximum(mag.(signal))
    
    onset = signal[(20000 - wl):(20000 + wl - 1), :]
    offset = signal[(20000 + (2 * fs) - wl):(20000 + (2 * fs) + wl - 1), :]


    return onset, offset, fb.cfs[channel, 2]
end

wl = 1000
fs = 30000
fb = createComplexGTFilterBank(50, 1000, fs, 500, .3, .016)

function get_onsetKernel_trainingSet(fb, fs; wl, stride = [1, 1])

    onset, offset, cf = channel_onset_kernel(fb, fs; wl = wl)
    onset

    channel_cfs = Float32.(real.(copy(onset)))
    window_times = permutedims(Float32.(real.(copy(onset))))
    Threads.@threads for j in axes(channel_cfs, 2)
        channel_cfs[:, j] .= (fb.cfs[j, 1] - cf)/cf
    end

    Threads.@threads for j in axes(window_times, 2)
        window_times[:, j] .= ((j - wl - .5) * 10 / fs) 
    end
    window_times = permutedims(window_times)

    #columns: frequency difference from center, time difference from center
    locations = zeros(Float32, length(onset), 2)
    valuesComplex = zeros(eltype(onset), length(onset))

    Threads.@threads for i in eachindex(onset)
        locations[i, 1] = channel_cfs[i]
        locations[i, 2] = window_times[i]
        valuesComplex[i] = onset[i]
    end

    values = [real.(valuesComplex) imag.(valuesComplex)]

    return permutedims(locations), permutedims(values)
end

wl = 1000
fs = 30000
stride = (50, 2)
fb = createComplexGTFilterBank(50, 1000, fs, 500, .3, .016)
@time locations, values = get_onsetKernel_trainingSet(fb, fs; wl = wl, stride = stride);
locations
values # = [real.(valuesComplex) imag.(valuesComplex)]


### define network structure
using Flux, CUDA

#partition data
function train_test_partition(X, Y, frac_train)
    n = Int(frac_train * size(X, 2))
    indices = collect(1:size(X, 2))
    trainIndices = sample(indices, n, replace = false)
    testIndices = indices[∉(Set(trainIndices)).(indices)]
    
    xTrain = X[:, trainIndices]
    xTest = X[:, testIndices]
    yTrain = Y[:, trainIndices]
    yTest = Y[:, testIndices]
   
    return xTrain, xTest, yTrain, yTest

end


kernelFunc = Chain(Dense(2 => 8, tanh), 
                   Dense(8 => 32, tanh), 
                   Dense(32 => 32, relu), 
                   Dense(32 => 2, tanh))

kernelFunc = fmap(cu, kernelFunc)
pred_error(x, y) = sum((x .- y).^2)

frac_train = .75
batchSize = 8000

epochs = 50
opt_state = Flux.setup(Adam(.05), kernelFunc)


@time xTrain, xTest, yTrain, yTest = train_test_partition(locations, values, frac_train);
numBatches = div(size(xTrain, 2), batchSize) 

xTrain = cu(xTrain);
yTrain = cu(yTrain);
xTest = cu(xTest);
yTest = cu(yTest);
perm = collect(1:size(xTrain, 2))


pred_error(kernelFunc(xTest), yTest)

testError = zeros(Float32, epochs)


for epoch in 1:epochs
    shuffle!(perm)
    xTrain .= @view(xTrain[:, perm]);
    yTrain .= @view(yTrain[:, perm]);
    batchSamples = (1):(batchSize)

    xBatch =  @view(xTrain[:, batchSamples])
    yBatch =  @view(yTrain[:, batchSamples])

    #pred_error(kernelFunc(xBatch), yBatch)


    for b in 1:(numBatches - 1)
    
        gr = gradient(m -> pred_error(m(xBatch), yBatch), kernelFunc)
        Flux.update!(opt_state, kernelFunc, gr[1])
        batchSamples = ((b * batchSize) + 1):((b + 1) * batchSize)

        xBatch .=  @view(xTrain[:, batchSamples])
        yBatch .=  @view(yTrain[:, batchSamples])
    end

    testError[epoch] = pred_error(kernelFunc(xBatch), yBatch)
    println(epoch)
end



lines(testError)

#B6DZLW8T

τ = .01
threshold = .05
iters = 20
ds=100

X = amp[100:ds:end, :]
G1 = mag.(freqresponses(fb, fs))



#G1 .*=1.2
@time A1 = lca_iir2(X, G1; τ = τ, threshold = threshold, iters = iters);
heatmap(X)
heatmap(A1)


τ = .01
threshold = .03
iters = 5
ds=10



Y = signal[3000:ds:12100, :]
G2 = freqresponses(fb, fs)
#G2 = permutedims(G2)
phaseShifts = cispi.(fb.cfs[:, 2] .* 2 .* (ds/fs))

@time A2 = lca_iir2(Y, G2, phaseShifts; τ = τ, threshold = threshold, iters = iters);
A2m = mag.(A2)
heatmap(mag.(Y))
heatmap(A2m)





τ = .01
threshold = .03
iters = 15
ds=10

Y = signal[4000:ds:15100, :]
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


aa2 = A2m[1000, :]
scatterlines(aa2)

aa3 = mag.(A3)[1000, :]
scatterlines(aa3)



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






