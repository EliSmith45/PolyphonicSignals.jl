using CairoMakie 
cd("PolyphonicSignals")
using Revise

using PolyphonicSignals

############Flux stuff ###########
using Flux
using CUDA

fs = 8000
fs_envelope = 500
duration = 2
fmin = 40
fmax = 1600

fb = createComplexGTFilterBank(fmin, fmax, fs, 500, .4, .016);

function trainNN!(model, data; epochs, bs = 5, adamStep = .005, log_every = 2, reltol = 1e-5, decay = .99, maxiter = 500)
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
                #errors[logged, 2] = (1 - decay) * gr[1]

                if logged == 1
                    errors[logged, 2] = gr[1]
                    errors[logged, 3:4] .= [1.0f0, 1.0f0]
                   
                elseif (logged == 2) 
                    errors[logged, 2] = decay * errors[logged - 1, 2] + (1 - decay) * gr[1]
                    errors[logged, 3] =  errors[logged, 2] / errors[logged - 1, 2]
                    errors[logged, 4] =  errors[logged, 3] 
                    
                else
                    errors[logged, 2] = decay * errors[logged - 1, 2] + (1 - decay) * gr[1]
                    errors[logged, 3] =  errors[logged, 2] / errors[logged - 1, 2]
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


function makeTrainingSetOnsetDetection(fb1, fb2; fs = 20000, 
                                        samples = 20000, 
                                        pad = 1000,
                                        numFreqs,
                                        ds = 4)
   

    
    paddedLength = samples + pad
    dsLength = div(paddedLength, ds)
    targets = zeros(Float32, dsLength, size(fb1.cfs, 1), 1, numFreqs)
    tfds = zeros(ComplexF32, dsLength, size(fb1.cfs, 1), 4, numFreqs)
    #tfd =  zeros(ComplexF32, dsLength, numFreqs)

 
    s = zeros(Float32, samples + pad, length(batches) - 1)
   
    Threads.@threads for j in 1:numFreqs
        frs = sample[fb1.cfs[5:(end - 5), :]]
       
      
                
        lrf = [(rand() - .5) * 20]
        mff = [(rand() + 3) * 2]
        maf = [rand() * .1]
        erf = rand() * .1

        amp = [(rand() + 1.5) * 2]
        lra = [(rand() - .5) * 20]
        mfa = [rand() * 5]
        maa = [rand() * .00000001]
        era = rand() * .1
        
        s[(pad + 1):end, j] .= createSignalFromInstFA(createEnvelope(frs; linModRate = lrf, modulationFreq = mff, modulationAmp = maf, errorRate = erf, duration = samples / fs, fs = fs),
                                    createEnvelope(amp; linModRate = lra, modulationFreq = mfa, modulationAmp = maa, errorRate = era, duration = samples / fs, fs = fs),
                                    fs)


        
        tfds[:, :, 1, j] .= applyComplexFilterbank(@view(s[:, j]), fb1; forwards_backwards = true, ds)
        tfds[:, :, 2, j] .= imag.(@view(tfds[:, :, 1, j]))
        tfds[:, :, 1, j] .= real.(@view(tfds[:, :, 1, j]))



        tfds[:, :, 3, j] .= applyComplexFilterbank(@view(s[:, j]), fb2; forwards_backwards = true, ds)
        tfds[:, :, 4, j] .= imag.(@view(tfds[:, :, 3, j]))
        tfds[:, :, 3, j] .= real.(@view(tfds[:, :, 3, j]))

         #  targets[div(pad, ds) + div(samples, ds) + 1, j, 1, generated] = 1.0f0
        #targets[:, :, 1, j] .= DSP.conv(@view(targets[:, :, 1, j]), kern)[(1 + boundary1):(end - boundary1), (1 + boundary2):(end - boundary2)]
                
    end
 
    return real.(tfds), targets
end

using StatsBase
function makeTrainingSetLinearizer(fb1; fs = 20000, 
        samples = 20000, 
        keeprange = (2000, samples - 2000),
        fbins = 40,
        observations = 50,
        qfProbs = [.95, .04, .01],
        ds = 2)




    fbn = createComplexGTFilterBank(fb1.cfs[1, 2], fb1.cfs[end, 2], fs, size(fb1.cfs, 1), 1, .016)
    fbn.cfs = fb1.cfs[1:fbins, :]
    fbn.filters = fb1.filters[1:fbins]
    #dsLength = div(samples, ds)
    #targets1 = zeros(Float32, samples, size(fb1.cfs, 1), 1, observations)
    tfds1 = zeros(ComplexF32, samples, fbins, 5, observations)
    tfds2 = zeros(ComplexF32, samples, fbins, 5, observations)
    
    #targets2 = zeros(Float32, samples, size(fb1.cfs, 1), 1, observations)
    #tfds2 = zeros(ComplexF32, samples, size(fb1.cfs, 1), 1, observations)
    
    fBinLow = zero(Int64)
    frs = zeros(Float32, 4)
    amps = zeros(Float32, 4)
    qWeights = weights(qfProbs)
    numFreq = 0
    #tfd =  zeros(ComplexF32, dsLength, numFreqs)


    s = zeros(Float32, samples, 5)
    #frEnv = copy(s)
    #ampEnv = copy(s)
    
    for j in 1:observations

        s .= 0
        frs .= 0
        amps .= 0
        numFreq = sample([2, 3, 4], qWeights)
        fBinLow = rand(1:(size(fb1.cfs, 1) - fbins))
       
        fbn.cfs .= @view(fb1.cfs[fBinLow:(fBinLow + fbins - 1), :])
        fbn.filters .= @view(fb1.filters[fBinLow:(fBinLow + fbins - 1)])
        for k in 1:numFreq
            frs[k] = rand((fb1.cfs[fBinLow + 10, 2]):(fb1.cfs[fBinLow + fbins - 9, 2])) + (rand() * 2)
            amps[k] = rand() + 0.2

            lrf = [(rand() - .5) * 20]
            mff = [(rand()) * 1.1]
            maf = [rand() * .05]
            erf = rand() * .1
        
            lra = [(rand() - .5) * 2]
            mfa = [rand() ]
            maa = [rand() * .005]
            era = rand() * .1
        
            s[:, k] .= createSignalFromInstFA(createEnvelope(frs[k]; linModRate = lrf, modulationFreq = mff, modulationAmp = maf, errorRate = erf, duration = samples / fs, fs = fs),
                createEnvelope(amps[k]; linModRate = lra, modulationFreq = mfa, modulationAmp = maa, errorRate = era, duration = samples / fs, fs = fs),
                fs)
        
            applyComplexFilterbank!(@view(tfds1[:, :, k, j]), @view(s[:, k]), fbn; forwards_backwards = true, tfd2 = @view(tfds2[:, :, k, j]))
        end

        for j in 1:4
            s[:, 5] .+= @view(s[:, j])
        end
        applyComplexFilterbank!(@view(tfds1[:, :, 5, j]), @view(s[:, 5]), fbn; forwards_backwards = true, tfd2 = @view(tfds2[:, :, 5, j]))
  


    end

    keepbins = keeprange[1]:ds:keeprange[2]
    inputs = zeros(eltype(frs), length(keepbins), size(tfds1, 2), 3, observations)
    inputs[:, :, 1, :] .= real.(@view(tfds1[keepbins, :, 5, :]))
    inputs[:, :, 2, :] .= imag.(@view(tfds1[keepbins, :, 5, :]))
    inputs[:, :, 3, :] .= mag.(@view(tfds1[keepbins, :, 5, :]))
    outputs = zeros(eltype(inputs), length(keepbins), size(tfds1, 2), 1, observations)
    for j in 1:observations
        for k in 1:4
            outputs[:, :, 1, j] .+= mag.(@view(tfds1[keepbins, :, k, j]))
        end
    end
   # mag.(@view(tfds1[:, :, 1:4, :]))
    return inputs, outputs #real.(tfds), targets
end

fs = 20000
fmin = 40
fmax = 1600
nFilt = 400

fb1 = createComplexGTFilterBank(fmin, fmax, fs, nFilt, .4, .016);

@time tsIn, tsOut = makeTrainingSetLinearizer(fb1; fs = fs, 
    samples = div(fs, 2), 
    keeprange = (2000, div(fs, 2) - 2000),
    fbins = 40,
    observations = 150,
    qfProbs = [.9, .07, .02],
    ds = 4);

N = 140
heatmap(tsIn[:, :, 3, N])
heatmap(tsOut[:, :, 1, N])




a1 = Conv((5, 3), 1 => 8, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
a2 = Conv((9, 5), 8 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (2, 1))
a3 = Conv((9, 5), 16 => 32, relu; pad = SamePad(), stride = (1, 1), dilation = (4, 1))
a4 = Conv((9, 5), 32 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (8, 1))
a5= Conv((5, 5), 16 => 4, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
a6 = Conv((3, 3), 4 => 1, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))

p1 = Conv((5, 3), 2 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
p2 = Conv((5, 5), 16 => 32, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
p3 = Conv((5, 5), 32 => 64, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
p4 = Conv((1, 1), 64 => 64, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
p5= Conv((3, 3), 64 => 1, tanh; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
#p6 = Conv((3, 3), 16 => 1, tanh; pad = SamePad(), stride = (1, 1), dilation = (1, 1))



CUDA.memory_status()  
GC.gc(true)
CUDA.reclaim()   
#CUDA.unsafe_free!(x1,y1,xhat,yhat,x,y)



ampEncoder = Chain(a1, a2, a3, a4, a5, a6) |> gpu
phaseMask = Chain(p1, p2, p3, p5) |> gpu
linearizer = Parallel(.*, ampEncoder, phaseMask) |> gpu

xa = cu(tsIn[:, :, 3:3, :])
xp = cu(tsIn[:, :, 1:2, :])
yTrain = cu(tsOut)
@time xy = ampEncoder(xa);
@time yp = phaseMask(xp);
@time y1 = linearizer((xa, xp));
y1 = y1 |> cpu
N = 4
heatmap(tsIn[:, :, 3, N])
heatmap(tsOut[:, :, 1, N])
heatmap(y1[:, :, 1, N])
Flux.mse(y1, tsOut)


errors = trainNN!(linearizer, ((xa, xp), yTrain); epochs = 50,
    bs = 25, 
    adamStep = .0015, 
    log_every = 1, 
    reltol = 1e-12, 
    decay = .95, 
    maxiter = 1000)

@time y1 = linearizer((xa[:, :, :, 1:20], xp[:, :, :, 1:20])...);
y1 = y1 |> cpu
N = 1
heatmap(tsIn[:, :, 3, N])
heatmap(tsOut[:, :, 1, N])
heatmap(y1[:, :, 1, N])
Flux.mse(y1, tsOut)




fs = 10000
fmin = 40
fmax = 1600
nFilt = 50
repeats = 3

fb1 = createComplexGTFilterBank(fmin, fmax, fs, nFilt, .3, .016);
fb2 = createComplexGTFilterBank(fmin, fmax, fs, nFilt, .95, .016);


@time xTrain, yTrain = makeTrainingSetOnsetDetection(fb1, fb2; fs = fs, samples = 1200, pad = 1000, repeats = repeats, diffusionKernel = (3, 3), ds = 10, batchSize = 20);
GC.gc(true)

s=170
heatmap(sqrt.((xTrain[:, :, 1, s].^2) .+ (xTrain[:, :, 2, s].^2)))
heatmap(sqrt.((xTrain[:, :, 3, s].^2) .+ (xTrain[:, :, 4, s].^2)))
heatmap(yTrain[:, :, 1, s])

heatmap(xTrain[:, :, 4, s] )
heatmap(xTrain[:, :, 3, s] )
heatmap(xTrain[:, :, 2, s] )
heatmap(xTrain[:, :, 1, s] )



####design the CNN


c1 = Conv((5, 5), 4 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
c2 = Conv((7, 5), 16 => 32, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
c3 = Conv((7, 5), 32 => 32, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
c4 = Conv((7, 5), 32 => 16, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
c5= Conv((7, 5), 16 => 4, relu; pad = SamePad(), stride = (1, 1), dilation = (1, 1))
c6 = Conv((7, 5), 4 => 1, sigmoid; pad = SamePad(), stride = (1, 1), dilation = (1, 1))

b1 = BatchNorm(4, relu)
b2 = BatchNorm(16, relu)
b3 = BatchNorm(32, relu)
b4 = BatchNorm(32, relu)
b5 = BatchNorm(16, relu)
b6 = BatchNorm(4, relu)



CUDA.memory_status()  
GC.gc(true)
CUDA.reclaim()   
#CUDA.unsafe_free!(x1,y1,xhat,yhat,x,y)



model = Chain(b1, c1, b2, c2, b3, c3, b4, c4, b5, c5, b6, c6) |> gpu

xx = cu(xTrain[:, :, :, 1:2])
xy = model(xx)
heatmap(xx[:, :, 4, 1])
heatmap(xy[:, :, 1, 1])

errors = trainNN!(model, (xTrain, yTrain); epochs = 50,
    bs = 3, 
    adamStep = .002, 
    log_every = 10, 
    reltol = 1e-12, 
    decay = .95, 
    maxiter = 1050)


xyTrained = model(xx)
heatmap(xx[:, :, 4, 1])
heatmap(yTrain[:, :, 1, 1])

heatmap(xyTrained[:, :, 1, 1])


lines(errors[:, 1], errors[:, 2])
lines(errors[:, 1], errors[:, 3])
lines(errors[:, 1], errors[:, 4])












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


       

errors = trainNN!(model, (xp, yp); epochs = 20,
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

