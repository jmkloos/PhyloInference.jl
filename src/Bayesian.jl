module Likelihood
    using Distributions
    using Polynomials
    using Plots
export main_likelihood

    function calc_girsanov(x1,x0,y0,h,σ)
        b = (y0-x0)*(x1-x0)/(σ^2)
        a = -(0.5 * (y0-x0)^2 * h)/σ^2
        return b, a
    end
    function get_girsanov(TX,TY,h,σ)
        bx = ax = 0.0
        for i in 1:length(TX)
            for j in 1:length(TX[i])-1
                    b, a = calc_girsanov(TX[i][j+1],TX[i][j],TY[i][j],h,σ)
                    bx += b
                    ax += a
            end
        end
        return bx, ax
    end

    function main_likelihood(TX,TY,h,σ)
        bx, ax = get_girsanov(TX,TY,h,σ)
        σ2x = -1/(2*ax)
        μx = bx*σ2x
        Zx = μx + √(σ2x)*randn()
        return Zx
    end
end
