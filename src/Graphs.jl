module Graphs
    using Plots
    using Statistics
    using LaTeXStrings

    global folder = "output"

    function make_graphs(fignum,t,X,title = "")
        counter = 1
        P1 = plot(legend =false,dpi = 1000, size= (300,200),fontfamily = "Computer modern",grid=false)
        plot!(legend = false)
        for s in X
            plot!(t, s, label = "species $counter")
            counter += 1
        end
        display(P1)
        savefig(joinpath(folder,"gibbs_tree$fignum"))
    end

    function make_ribbon_plot(t,Xs,title="")
        P2 = plot(legend =false,dpi = 1000, size= (300,200),fontfamily = "Computer modern",grid=false)
        n = length(Xs[1][1]) #total steps
        m = length(Xs) #amount of trees
        for s in 1:length(Xs[1]) #numspecies
            W = [Xs[x][s][y] for x in 1:m, y in 1:n]
            W = W'
            p = 0.1
            upper = vec(mapslices(w -> quantile(w, 1-p/2), W, dims=2))
            ave = vec(mean(W, dims=2))
            lower = vec(mapslices(w -> quantile(w, p/2), W, dims=2))
            plot!(t,ave, ribbon = (ave-lower, upper-ave), label = "species $s", fillalpha = 0.05)
        end
        display(P2)
        savefig(joinpath(folder,"ribbon_tree$title.png"))
    end

    function make_theta_plot(θ_xs,θ_ys)
        P5 = plot(legend =false,dpi = 1000, size= (300,200),fontfamily = "Computer modern",grid=false)
        # plot!(θ_xs,θ_ys, linestyle =:dash)
        # for i in 1:length(θ_xs)
        #     scatter!([θ_xs[i]],[θ_ys[i]],markersize =  2,markercolor =1,markeralpha = (i/length(θ_xs))^2)
        # end
        for i in 1:length(θ_xs)-1
            plot!(θ_xs[i:i+1],θ_ys[i:i+1], color = 1, marker = 1,markercolor = "black",seriestype =:steppre,  markeralpha = 0.1+(0.9*i/length(θ_xs))^2,linealpha = 0.1+(0.9*i/length(θ_xs))^2)
            #scatter!([θ_xs[i]],[θ_ys[i]],markersize =  2,markercolor =1,markeralpha = (i/length(θ_xs))^2)
        end
        #scatter!([θ_xs[end]],[θ_ys[end]],markersize =  2,markercolor =1,markeralpha = ((length(θ_xs)-1)/length(θ_xs))^2)
        plot!([0,1],[0,1],color = 2,linestyle =:dash,lw=1) #line y = x for reference
        display(P5)
        savefig(joinpath(folder,"theta.png"))

        P7 = plot(legend =:topright, dpi = 1000,ylim =(0,1), size= (300,200),fontfamily = "Computer modern",legendfont = (8,"Computer modern"),fg_legend =:transparent,grid=false)
        plot!([runmean(θ_xs) runmean(θ_ys)],label=[L"$\theta_X$",L"$\theta_Y$"])
        display(P7)
        savefig(joinpath(folder,"theta_change.png"))
    end

    runmean(x, cx=cumsum(x)) = [cx[n]/n for n in 1:length(x)]
end
