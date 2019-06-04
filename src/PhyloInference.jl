module PhyloInference

include("./Gibbs.jl")
include("./Bayesian.jl")
include("./Initial.jl")
include("./Graphs.jl")

using Random
using Statistics
using Distributions
using Plots
using Base.Iterators #flatten()

struct Tree{T}
    tlist::T
end

function main(tstart,tnow,xstart,ystart,σ,h,θ_x,θ_y,Xobs,Yobs,iter_amount,X,Y; _folder="output")
    global folder = _folder
    t0 = 0.0
    tend = tnow - tstart
    numspecies = length(Xobs)
    tree_struc = create_tree_struc(numspecies)
    split_times = create_split_vec(tend,numspecies,tree_struc,h)
    tree_matrix = create_tree_matrix(numspecies,tree_struc,split_times,tend)

    #Real trees
    XR = deepcopy(X)
    YR = deepcopy(Y)
    TXR,order = get_tree(XR, tree_struc,split_times, h)
    TYR,order = get_tree(YR, tree_struc,split_times, h)

    #RUN
    X,Y = InitialTree.main_initial(Xobs,Yobs,numspecies,t0,tend,tree_matrix,h,σ)
    global tvec = range(0.0,stop = tend,length = Int(tend/h)+1)

    #Initial trees
    Graphs.make_graphs(1,tvec,X, "Trait X")
    Graphs.make_graphs(2,tvec,Y, "Trait Y")

    TX,order = get_tree(X,tree_struc,split_times, h)
    TY,order = get_tree(Y,tree_struc,split_times, h)
    err_x = get_error(TX,TXR)
    err_y = get_error(TY,TYR)
    println("error Xinitial: $err_x")
    println("error Yinitial: $err_y")
    global θ_xs, θ_ys = Float64[], Float64[]
    global TX_list,TY_list = Any[],Any[]
    global err_xs, err_ys = Float64[], Float64[]
    push!(θ_xs,0.0)
    push!(θ_ys,0.0)

    #MAINALG
    c0 = 100  #amount of tree samples before repredicting θ
    c1 = Int(round(iter_amount/c0)) #amount of repredictions of θ
    for q in 1:c1
        TX,TY = GibbsAlg.main_alg(TX,TY,h,σ,θ_x,θ_y,order,tree_struc,c0)
        push!(TX_list,Tree(deepcopy(TX)))
        push!(TY_list,Tree(deepcopy(TY)))
        err_x = get_error(TX,TXR)
        err_y = get_error(TY,TYR)
        push!(err_xs,err_x)
        push!(err_ys,err_y)
        println("error X: $err_x")
        println("error Y: $err_y")
        Z = rand(Bool)
        if Z
            θ_x = Likelihood.main_likelihood(TX,TY,h,σ)
        else
            θ_y =  Likelihood.main_likelihood(TY,TX,h,σ)
        end
        println("$q:\nθ_x = $θ_x\nθ_y = $θ_y")
        push!(θ_xs, θ_x)
        push!(θ_ys, θ_y)
    end
    global X_list = [get_lists(TX.tlist,deepcopy(X),order,tree_struc,split_times,h) for TX in TX_list[max(length(TX_list)-1000,1):end]]
    global Y_list = [get_lists(TY.tlist,deepcopy(Y),order,tree_struc,split_times,h) for TY in TY_list[max(length(TX_list)-1000,1):end]]
    Xave = get_ave_tree(X_list)
    Yave = get_ave_tree(Y_list)
    err_x = get_error(get_tree(Xave,tree_struc,split_times, h)[1],TXR)
    err_y = get_error(get_tree(Yave,tree_struc,split_times, h)[1],TYR)
    println("error Xave: $err_x")
    println("error Yave: $err_y")
    Graphs.make_theta_plot(θ_xs,θ_ys)
    Graphs.make_ribbon_plot(tvec,X_list, "Trait X")
    Graphs.make_ribbon_plot(tvec,Y_list, "Trait Y")
end

function create_tree_struc(n)
    struc = Array{Any,1}(1:n)
    struc[1] = nothing
    for i in 2:n
        struc[i] = rand(1:i-1)
    end
    return struc
end

function create_split_vec(T,n,tree_struc,h)
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

function get_tree(X,tree_struc,split_times, h)
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

function get_lists(TX,X,order,tree_struc,split_times, h)
    numspecies = length(tree_struc)
    cur_index = 1
    amount = length([j for j in tree_struc if j == 1])
    X[1][1] = TX[1][1]
    X[1][2:end] = collect(flatten(vcat([TX[i][2:end] for i in cur_index : cur_index + amount])))
    cur_index += amount + 1
    for s in order[2:end]
        amount = length([j for j in tree_struc if j == s])
        first = Int(round(split_times[s]/h)) + 1
        X[s][1:first] = X[tree_struc[s]][1:first]
        X[s][first+1:end] = collect(flatten(vcat([TX[i][2:end] for i in cur_index : cur_index + amount])))
        cur_index += amount + 1
    end
    return X
end

# function make_graphs(fignum,t,X,title = "")
#     counter = 1
#     P1 = plot(xlabel = "time", legend =:topleft, dpi = 200,title = title)
#     plot!(legend = false)
#     for s in X
#         plot!(t, s, label = "species $counter")
#         counter += 1
#     end
#     display(P1)
#     savefig(joinpath(folder,"gibbs_tree$fignum"))
# end
#
# function make_ribbon_plot(t,Xs,title="")
#     P2 = plot(xlabel = "time", legend =false, dpi = 300,title=title)
#     n = length(Xs[1][1]) #total steps
#     m = length(Xs) #amount of trees
#     for s in 1:length(Xs[1]) #numspecies
#         W = [Xs[x][s][y] for x in 1:m, y in 1:n]
#         W = W'
#         p = 0.1
#         upper = vec(mapslices(w -> quantile(w, 1-p/2), W, dims=2))
#         ave = vec(mean(W, dims=2))
#         lower = vec(mapslices(w -> quantile(w, p/2), W, dims=2))
#         plot!(t,ave, ribbon = (ave-lower, upper-ave), label = "species $s", fillalpha = 0.05)
#     end
#     display(P2)
#     savefig(joinpath(folder,"ribbon_tree$title.png"))
# end
#
# function make_theta_plot(θ_xs,θ_ys)
#     P5 = plot(xlabel = "theta x",ylabel = "theta y", legend = false,dpi=200)
#     plot!(θ_xs,θ_ys, linestyle =:dash)
#     for i in 1:length(θ_xs)
#         scatter!([θ_xs[i]],[θ_ys[i]],markersize =  4,markercolor =:blue,markeralpha = (i/length(θ_xs))^2)
#     end
#     plot!([0,1],[0,1]) #line y = x for reference
#     display(P5)
#     savefig(joinpath(folder,"theta.png"))
#
#     # P6 = violin(["theta_x" "theta_y"],[θ_xs[3000:end] θ_ys[3000:end]],title = "Violin plot",legend= false,dpi=200)
#     # scatter!(["theta_x","theta_y"],[0.9,0.1])
#     # display(P6)
#     # savefig(joinpath(folder,"violintheta.png")
#
#     plot([θ_xs θ_ys],label=["theta_x","theta_y"],xlabel = "iteration",dpi=200)
#     savefig(joinpath(folder,"theta_change.png"))
# end

function get_error(TX,TXR)
    err = (TXR[1][1] - TX[1][1])^2
    for i in 1:length(TX)
        for j in 2:length(TX[i])
            err += (TXR[i][j] - TX[i][j])^2
        end
    end
    return √(err)
end

function get_ave_tree(Xs)
    n = length(Xs[1][1]) #total steps
    m = length(Xs) #amount of trees
    Xave = deepcopy(Xs[1])
    for s in 1:length(Xs[1]) #numspecies
        W = [Xs[x][s][y] for x in 1:m, y in 1:n]
        Xave[s] = vec(mean(W,dims = 1))
    end
    return Xave
end

end # module
