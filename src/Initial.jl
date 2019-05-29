module InitialTree
    using Random
    using Statistics
    using Distributions
export main_initial

    function feed_forward(currentspec,tcurrent,tend,tree,X,Y,h,σ,Xobs,Yobs)
        tsplit = minimum([t for t in tree[currentspec,:] if t>tcurrent])
        if tsplit == tend
            fixed_walk(currentspec,tsplit-tcurrent,X,Y,h,Xobs,Yobs)
            return
        end
        #perform random walk until tsplit
        random_walk(currentspec,tsplit-tcurrent,X,Y,h,σ)
        #split
        newspec = [i for (i,val) in enumerate(tree[currentspec,:]) if val == tsplit][1]
        X[newspec] = X[currentspec]
        Y[newspec] = Y[currentspec]

        feed_forward(currentspec,tsplit,tend,tree,X,Y,h,σ,Xobs,Yobs)
        feed_forward(newspec,tsplit,tend,tree,X,Y,h,σ,Xobs,Yobs)
    end

    function fixed_walk(cur,t,X,Y,h,Xobs,Yobs)
        # straight line to end observation
        n = Int(round(t/h))
        x0 = X[cur][end]
        y0 = Y[cur][end]
        xn = Xobs[cur]
        yn = Yobs[cur]
        x = range(x0,stop = xn, length = n+1)
        y = range(y0,stop = yn, length = n+1)
        X[cur] = [X[cur];x[2:end]]
        Y[cur] = [Y[cur];y[2:end]]
    end

    function random_walk(cur,t,X,Y,h,σ)
        #move up to tsplit
        n = Int(round(t/h))
        x0 = X[cur][end]
        y0 = Y[cur][end]
        x = [x0 for dummy in 1:n+1]
        y = [y0 for dummy in 1:n+1]
        Z1 = rand(Normal(),n)
        Z2 = rand(Normal(),n)
        for i in 1:n
            x[i+1] = x[i] + sqrt(h)*σ/100*Z1[i]
            y[i+1] = y[i] + sqrt(h)*σ/100*Z2[i]
        end
        X[cur] = [X[cur];x[2:end]]
        Y[cur] = [Y[cur];y[2:end]]
    end

    function main_initial(Xobs,Yobs,numspecies,t0,tend,tree_matrix,h,σ)
        xstart,ystart = 0.0,0.0#mean(Xobs),mean(Yobs)
        X = [[xstart] for dummy in 1:numspecies]
        Y = [[ystart] for dummy in 1:numspecies]
        feed_forward(1,t0,tend,tree_matrix,X,Y,h,σ,Xobs,Yobs)
        return X,Y
    end
end
