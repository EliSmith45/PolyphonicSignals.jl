


#https://perso.ens-lyon.fr/patrice.abry/ENSEIGNEMENTS/14M2SCExam/Selesnick.pdf
module MorphologicalComponents ###SALSA implementation of two-dictionary MCA
export salsa


using WAV, DSP, FFTW #fast fourier transform backend
using DataStructures, Memoize
using ..TQWT

function soft(x, lambda)
    return maximum([0, 1- (lambda/abs(x))])*x 
end

function similarLL(old)
    new = similar(old)
    Threads.@threads for j in eachindex(new)
        new[j] = zeros(Float32, length(old[j]))
    end
    return new
end

function update_b(y, w, b, d, mu, plan, high, low, coef)
    
    tqwt!(coef; highPass = high, lowPass = low, seq = y, p = plan)
    
    Threads.@threads for j in eachindex(b)
        b[j] .= coef[j] .+ mu.*(w[j] .+ d[j])
    end
    
end

function update_b!(b, tr, w, d, mu)
    
    Threads.@threads for j in eachindex(b)
        b[j] .= tr[j] .+ mu.*(w[j] .+ d[j])
    end
    
end
function update_u!(u, b, ct, mu)
    
    Threads.@threads for j in eachindex(u)
        u[j] .=  (1/mu).*(b[j]) .- (1/(mu*(mu+2))).*ct[j]
    end
    
end

function update_w!(w, u, d, lambda, l, mu)
    Threads.@threads for j in eachindex(w)
        T=lambda[j]*l/(2*mu)
        for i in eachindex(w[j])
            w[j][i] = u[j][i] - d[j][i]
            
            pen = abs(w[j][i]) < T ? w[j][i] : (T*sign(w[j][i]))
            w[j][i] -= pen
        end
    end
end

function update_d!(d, u, w)
    Threads.@threads for j in eachindex(d)
            d[j] .+= (w[j] .- u[j])
    end
end


function salsa(y; l1, l2, mu, plan1, plan2, targetError, maxIter, check_every, printIter)
    
    
    c = copy(y)
    transform1 = init_coefficients(plan1)
    high1 = init_highPassVec(plan1)
    low1 = init_lowPassVec(plan1)
    
    transform2 = init_coefficients(plan2)
    high2 = init_highPassVec(plan2)
    low2 = init_lowPassVec(plan2)

    

   # c = copy(y)
    tqwt!(transform1; highPass=high1, lowPass=low1, seq=y, p=plan1)
    u1 = similarLL(transform1)
    b1 = similarLL(transform1)
    d1 = similarLL(transform1)
    w1 = deepcopy(transform1)
    ct1 = similarLL(transform1)
    
    tqwt!(transform2; highPass=high2, lowPass=low2, seq=y, p=plan2)
    u2 = similarLL(transform2)
    b2 = similarLL(transform2)
    d2 = similarLL(transform2)
    w2 = deepcopy(transform2)
    ct2 = similarLL(transform2)

    res = ones(Float32, div(maxIter, check_every)) .* Inf
    checked = 1
    iter = 1

    lambda1 = zeros(Float32, length(d1))
    lambda2 = zeros(Float32, length(d2))

    Threads.@threads for i in eachindex(lambda1)
        lambda1[i] = maximum(getWavelets(length(y); p = plan1, whichLevel = i))
    end
    for i in eachindex(lambda1)
        if lambda1[i] == 0
            lambda1[i] = 2*lambda1[i-1]-lambda1[i-2]
        end
    end
    
    Threads.@threads for i in eachindex(lambda2)
        lambda2[i] = maximum(getWavelets(length(y); p = plan2, whichLevel = i))
    end
    for i in eachindex(lambda2)
        if lambda2[i] == 0
            lambda2[i] = 2*lambda2[i-1]-lambda2[i-2]
        end
    end

    #lambda1 = ones(Float32, length(w1))
    #lambda2 = ones(Float32, length(w2))
    
    while (iter <= maxIter)

        update_b!(b1, transform1, w1, d1, mu)
        update_b!(b2, transform2, w2, d2, mu)
        
        c .= itqwt!(b1; highPass=high1, lowPass=low1, p=plan1) .+ itqwt!(b2; highPass=high2, lowPass=low2, p=plan2)
       
        tqwt!(ct1; highPass=high1, lowPass=low1, seq=c, p=plan1)
        tqwt!(ct2; highPass=high2, lowPass=low2, seq=c, p=plan2)
        
        update_u!(u1, b1, ct1, mu)
        update_u!(u2, b2, ct2, mu)

        update_w!(w1, u1, d1, lambda1, l1, mu)
        update_w!(w2, u2, d2, lambda2, l2, mu)
        update_d!(d1, u1, w1)
        update_d!(d2, u2, w2)

        
      



        
        
        
        if mod(iter, check_every) == 0 && checked > 5
            y1 = itqwt!(w1; highPass = high1, lowPass = low1, p = plan1) 
            y2 = itqwt!(w2; highPass = high2, lowPass = low2, p = plan2)
            err = y .- y1 .- y2
            res[checked] = sum(abs.(err))/sum(abs.(y))
        

            if  res[checked] < targetError
                res = res[1:checked]
                return y1, y2, res
            elseif res[checked]/res[checked - 1] > .95 #&& res[checked] < 5*targetError
                res = res[1:checked]
                return y1, y2, res
            end

            checked += 1
        end
        
        if printIter
            println(iter)
        end
        iter += 1
    end

    
    y1 = itqwt!(w1; highPass = high1, lowPass = low1, p = plan1) 
    y2 = itqwt!(w2; highPass = high2, lowPass = low2, p = plan2)
    res = res[1:checked]
    return y1, y2, res
end

end #end of MCA module
