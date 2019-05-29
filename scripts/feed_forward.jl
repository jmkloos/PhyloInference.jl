include("../src/Bayesian.jl")
using Plots
using Random
using Statistics
using Distributions
using LaTeXStrings
using Colors

Random.seed!(3)

#VARS
tstart = -3.0
tnow = 0.0
xstart = 3.0
ystart = 3.0
σ = 0.1
θ_x = 0.0
θ_y = 0.5
h = 0.1
#ρ = -0.9
numspecies = 100

function main()
    t0 = 0.0
    tend = tnow - tstart
    tree_struc = create_tree_struc(numspecies)
    split_times = create_split_vec_short(tend,numspecies,tree_struc)
    #split_times = create_split_vec(tend,numspecies)
    # tree_struc = [0,1,2,3]
    # split_times = [0.0, 1.0, 2.0, 3.0]
    tree_matrix = create_tree_matrix(numspecies,tree_struc,split_times,tend)
    global X = [[xstart] for dummy in 1:numspecies]
    global Y = [[ystart] for dummy in 1:numspecies]
    tvec = range(0.0,stop = tend,length = Int(tend/h) + 1)
    #RUN
    currentspecies = 1
    feed_forward(currentspecies,t0,tend,tree_matrix,X,Y)
    Xobs = [x[end] for x in X]
    Yobs = [y[end] for y in Y]
    display(scatter(Xobs,Yobs,size = (300,200),dpi = 200, markersize = 7,legend = false))
    println("Xobs = $Xobs")
    println("Yobs = $Yobs")
    make_graphs(tvec,X,Y)
    #θ_x,θ_y = main_likelihood(X,Y,tree_struc,split_times,h,θ_x,θ_y)
    TX,order = get_tree(X,tree_struc,split_times, h)
    TY,order = get_tree(Y,tree_struc,split_times, h)
    θ_x = Likelihood4.main_likelihood(TX,TY,h,σ)
    θ_y = Likelihood4.main_likelihood(TY,TX,h,σ)
    return X, Y, Xobs, Yobs,TY
end

function create_tree_struc(n)
    struc = Array{Any,1}(1:n)
    struc[1] = nothing
    for i in 2:n
        struc[i] = rand(1:i-1)
    end
    return struc
end

function create_split_vec_old(T,n)
    split_vec = zeros(n)
    randlist = rand(Uniform(0.1,T-1),n-1)#change 0.1 to h
    rounded = [round(i; digits = 2) for i in randlist]
    if length(rounded) != length(Set(rounded)) #duplicates
        println("error")
        return create_split_vec(T,n)
    else
        return [0; sort(rounded)]
    end
end
function create_split_vec(T,n)
    split_vec = zeros(n)
    randlist = sort(sample(h:h:T-h, n-1, replace=false))
    return [0.0; randlist]
end


function create_split_vec_short(T,n,tree_struc)
    m = Int(round(1.1*n))
    sample_list = [0.0; sample(h:h:T-h,m-1)]
    println(sort(sample_list))
    split_vec = zeros(n)
    for i in 2:n
        split_vec[i] = minimum(sample_list[j] for j in 2:m if sample_list[j] >
        maximum([split_vec[tree_struc[i]] ; [split_vec[k] for k in 1:i-1 if tree_struc[k] == tree_struc[i]]]))
        sample_list[findall(x -> x == split_vec[i],sample_list)[1]] = 0.0
        a,b = split_vec[i], tree_struc[i]
        println("($i,$a,$b)")
    end
    return split_vec
end


function create_tree_matrix(n,struc,split,tend)
    matrix = zeros(n,n)
    for i in 2:n
        anc = struc[i]
        matrix[i,anc] = split[i]
        matrix[anc,i] = split[i]
    end
    for i in 1:n
        matrix[i,i] = tend
    end
    return matrix
end

function feed_forward(currentspec,tcurrent,tend,tree,X,Y)
    tsplit = minimum([t for t in tree[currentspec,:] if t>tcurrent])
    #perform random walk until tsplit
    random_walk(currentspec,tsplit-tcurrent,X,Y)
    if tsplit == tend
        return
    end
    #split
    newspec = [i for (i,val) in enumerate(tree[currentspec,:]) if val == tsplit][1]
    X[newspec] = X[currentspec]
    Y[newspec] = Y[currentspec]

    feed_forward(currentspec,tsplit,tend,tree,X,Y)
    feed_forward(newspec,tsplit,tend,tree,X,Y)
end

function random_walk(cur,t,X,Y)
    #move up to tsplit
    n = Int(round(t/h))
    x0 = X[cur][end]
    y0 = Y[cur][end]
    x = [x0 for dummy in 1:n+1]
    y = [y0 for dummy in 1:n+1]
    Z1 = rand(Normal(),n)
    Z2 = rand(Normal(),n)
    for i in 1:n
        x[i+1] = x[i] + θ_x*(y[i] - x[i])*h + √(h)*σ *Z1[i]
        y[i+1] = y[i] + θ_y*(x[i] - y[i])*h + √(h)*σ *Z2[i]
    end
    X[cur] = [X[cur];x[2:end]]
    Y[cur] = [Y[cur];y[2:end]]
end

function make_graphs(t,X,Y)
    # P0 = plot(legend =:right,dpi = 1000, size= (300,200),fontfamily = "Computer modern",legendfont = (7,"Computer modern"),fg_legend =:transparent,grid=false)
    # #plot!(legend = false)
    # for s in X
    #     plot!(t, s, label = "Trait X")
    # end
    # for s in Y
    #     plot!(t, s, label = "Trait Y")
    # end
    # display(P0)
    # savefig("forward_tree_branch_simple.pdf")
    # cur_colors = get_color_palette(:phase, plot_color(:white), 4)
    # counter = 1
    # P1 = plot(legend =:bottomleft, dpi = 1000, size= (300,200),fontfamily = "Computer modern",legendfont = (5,"Computer modern"),fg_legend =:transparent,grid=false)
    # #plot!(legend = false)
    # for s in X
    #     plot!(t, s, label = "species $counter", color = cur_colors[counter])
    #     counter += 1
    # end
    # display(P1)
    # savefig("forward_tree_drift_X")
    counter = 1
    P1 = plot(legend =:topleft,dpi = 1000, size= (300,200),fontfamily = "Computer modern",legendfont = (6,"Computer modern"),fg_legend =:transparent,grid=false)
    #plot!(legend = false)
    for s in X
        plot!(t, s, label = "sp. $counter")
        counter += 1
    end
    display(P1)
    savefig("forward_tree_drift_X.pdf")

    counter = 1
    P2 = plot(legend =:topleft, dpi = 1000, size= (300,200),fontfamily = "Computer modern",legendfont = (6,"Computer modern"),fg_legend =:transparent,grid=false)
    #plot!(legend = false)
    for s in Y
        plot!(t, s, label = "sp. $counter")
        counter += 1
    end
    display(P2)
    savefig("forward_tree_drift_Y.pdf")

    col = 1/h
    step = Int(round((length(X[1])-1)/col))
    P3 = plot(dpi = 1000,size = (300,200),fontfamily = "Computer modern",legend =:right,fg_legend =:transparent,grid=false,legendfont = (6,"Computer modern"))
    for i in 1:length(X)
        j=1
        for k in 1:col
            plot!(X[i][j:j+step],Y[i][j:j+step],linealpha = 0.05 + 0.9*((j+step)/(length(X[i])))^2,linewidth = 1.0,color = i,label= "")
            j+=step
        end
    end
    for i in 1:length(X)
        plot!([3],[3], color = i, label = "sp. $i")
    end
    Xobs = [X[i][end] for i in 1:length(X)]
    Yobs = [Y[i][end] for i in 1:length(X)]
    for i in 1:length(X)
        scatter!([Xobs[i]],[Yobs[i]],markersize = 3,markercolor = i,markerstrokewidth = 1, label = "")
    end
    # scatter!(Xobs,Yobs, markersize = 3, markercolor = [i for i in 1:length(X)],markerstrokewidth = 1.5, labels = ["sp.1","sp.2","sp.1","sp.2"])
    savefig("forward_scatter.pdf")
    display(P3)

    col = 1#10
    step = Int(round((length(X[1])-1)/col))
    P4 = plot3d(xlabel = "time", ylabel = "trait X", zlabel = "Trait Y               ",legend = false,dpi=200)
    for i in 1:length(X)
        j=1
        for k in 1:col
            plot3d!(t[j:j+step],X[i][j:j+step],Y[i][j:j+step], gridalpha = 1.0, gridwidth = 1.0, #camera = (40,20),
            linealpha = ((j+step)/(length(X[i]))),linewidth = 1.0,color = i)

            j+=step
        end
    end
    Xobs = [X[i][end] for i in 1:length(X)]
    Yobs = [Y[i][end] for i in 1:length(X)]
    t_end = [t[end] for i in 1:length(X)]
    scatter3d!(t_end,Xobs,Yobs, markersize =3, markercolor = [i for i in 1:length(X)],dpi=200)
    savefig("3dplot.png")
    display(P4)
end

function get_tree(X,tree_struc,split_times, h)
    println(tree_struc)
    println(split_times)
    numspecies = length(tree_struc)
    T = []
    order = []
    function get_branch(T,order,i=1,j=1,t=0.0)
        first = Int(round(t/h)) + 1
        if i in tree_struc[j+1:end]
            k = minimum([n for n in j+1:numspecies if tree_struc[n] == i])
            tnew = split_times[k]
            last = Int(round(tnew/h)) + 1
            branch = X[i][first:last]
            push!(T,branch)
            get_branch(T,order,i,k,tnew)
            get_branch(T,order,k,k,tnew)
        else
            last = length(X[i])
            branch = X[i][first:last]
            push!(order,i)
            push!(T,branch)
        end
    end
    get_branch(T,order)
    return T,order
end

 X, Y, Xobs, Yobs = main()
