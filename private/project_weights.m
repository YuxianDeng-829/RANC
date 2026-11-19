function w = project_weights(v, r, delta)
    mass = r / sum(r);
    atZero = mass' * max(v, delta);
    if atZero == 1
        w = max(v, delta);
        return;
    end

    if atZero > 1
        low = 0;
        high = 1;
        while mass' * max(v - high * mass, delta) > 1
            high = 2 * high;
        end
    else
        low = -1;
        high = 0;
        while mass' * max(v - low * mass, delta) < 1
            low = 2 * low;
        end
    end

    for iter = 1:80
        tau = low + (high - low) / 2;
        w = max(v - tau * mass, delta);
        total = mass' * w;
        if abs(total - 1) <= 8 * eps
            return;
        end
        if total > 1
            low = tau;
        else
            high = tau;
        end
    end
end
