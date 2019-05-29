module GibbsAlg
    using Random
    using Statistics
    using Distributions
    using Base.Iterators #flatten()
export main_alg

    function new_root(x2,y1,h,σ,θ_x)
        σ0 = 100000.0 #std prior X_0
        θ = θ_x
        μ = y1
        m = ((1-θ*h)*(x2 - μ*θ*h)*σ0^2)/(σ0^2*(1-θ*h)^2 + h*σ^2)
        v = σ0^2*h*σ^2/(σ0^2*(1-θ*h)^2 + h*σ^2)
        Z = randn()
        return m + √(v)*Z
    end

    function new_branch_point(x0,x2,y0,y1,h,σ,θ_x)
        θ = θ_x
        m = ((1-(θ*h))*(x2 + x0) + θ*h*(y0 - (1-θ*h)*y1)) /((1-(θ*h))^2 + 1)
        v = σ^2*h/((1-(θ*h))^2 + 1)
        Z = randn()
        return m + √(v)*Z
    end

    function new_split_point(x0,x2a,x2b,y0,y1,h,σ,θ_x)
        θ = θ_x
        m = ((1-(θ*h))*(x0 + x2a + x2b) + θ*h*(y0 - 2*(1-θ*h)*y1)) / (2*(1-θ*h)^2 + 1)
        v = σ^2*h/(2*(1-(θ*h))^2 + 1)
        Z = randn()
        return m + √(v)*Z
    end

    function sampling(TX,TY,h,σ,θ_x,order,tree_struc)
        TX[1][1] = new_root(TX[1][2],TY[1][1],h,σ,θ_x)
        cur_index = 1
        for s in order
            amount =  length([j for j in tree_struc if j == s])
            for i in cur_index : cur_index + amount
                for j in 2:length(TX[i])-1
                    TX[i][j] = new_branch_point(TX[i][j-1],TX[i][j+1],TY[i][j-1],TY[i][j],h,σ,θ_x)
                end
                if i != cur_index + amount #split_point
                    i2 = [m for m in i+2:length(TX) if TX[m][1] == TX[i][end]][1]
                    TX[i][end] = new_split_point(TX[i][end-1],TX[i+1][2],TX[i2][2],TY[i][end-1],TY[i][end],h,σ,θ_x)
                    TX[i+1][1] = TX[i2][1] = TX[i][end]
                end
            end
            cur_index += amount + 1
        end
        return TX
    end

    function main_alg(TX,TY,h,σ,θ_x,θ_y,order,tree_struc,amount)
        for i in 1:amount
            TX = sampling(TX,TY,h,σ,θ_x,order,tree_struc)
            TY = sampling(TY,TX,h,σ,θ_y,order,tree_struc)
        end
        return TX,TY
    end
end
