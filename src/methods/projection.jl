

function box_projection(y::AbstractVector)
    return clamp.(y, 0.0, 1.0)
end

function hyperplane_projection(y::AbstractVector, k::Integer)
    n = length(y)
    return y .- (sum(y) - k) / n
end

function dykstra_projection(x0::AbstractVector, k::Integer, eps::Float64)
    n = length(x0)

    x2 = copy(x0)
    y1 = zeros(n)
    y2 = zeros(n)

    while true
        y1_old = copy(y1)
        y2_old = copy(y2)

        z1 = x2 - y1
        x1 = box_projection(z1)
        y1 = x1 - z1

        z2 = x1 - y2
        x2 = hyperplane_projection(z2, k)
        y2 = x2 - z2

        res = norm(y1 - y1_old)^2 + norm(y2 - y2_old)^2

        if res <= eps
            return x2
        end
    end
end