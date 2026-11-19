function [labels, w] = RANC_D(B, c, lambda, maxIter, tol)
    if nargin < 3 || isempty(lambda)
        lambda = 1;
    end
    if nargin < 4 || isempty(maxIter)
        maxIter = 100;
    end
    if nargin < 5 || isempty(tol)
        tol = 1e-6;
    end
    [B, S0, r, active] = prepare_graph(B, c, lambda, maxIter, tol);
    c = double(c);
    lambda = double(lambda);
    maxIter = double(maxIter);

    m = numel(r);
    delta = 1e-6;
    anchorLabels = n2hi_init(S0, c);
    H = full(sparse((1:m)', anchorLabels, 1, m, c));
    weights = ones(m, 1);
    objective = discrete_objective(S0, r, H, weights, lambda);

    for iter = 2:maxIter
        previous = objective;
        [H, anchorLabels] = update_indicator(S0, r, weights, H, anchorLabels);
        objective = discrete_objective(S0, r, H, weights, lambda);

        for weightIter = 1:5
            [~, gradient] = discrete_objective(S0, r, H, weights, lambda);
            direction = mass_tangent(gradient, r);
            direction = direction / max(1, norm(direction, inf));
            step = 1;
            accepted = false;

            for trial = 1:20
                candidate = project_weights(weights + step * direction, r, delta);
                candidateValue = discrete_objective(S0, r, H, candidate, lambda);
                if candidateValue > objective
                    accepted = true;
                    break;
                end
                step = step / 2;
            end

            if ~accepted
                break;
            end
            gain = (candidateValue - objective) / max(1, abs(objective));
            weights = candidate;
            objective = candidateValue;
            if gain < 1e-8
                break;
            end
        end

        if abs(objective - previous) / max(1, abs(previous)) < tol
            break;
        end
    end

    [labels, w] = clustering_output(B, anchorLabels, c, weights, active);
end

function [objective, gradient] = discrete_objective(S0, r, H, w, lambda)
    relativeMass = r / sum(r);
    objective = -lambda * sum(relativeMass .* (w - 1).^2);
    if nargout > 1
        gradient = -lambda * (2 * relativeMass .* (w - 1));
    end

    for q = 1:size(H, 2)
        members = H(:, q) ~= 0;
        memberWeights = w(members);
        memberMass = r(members);
        within = S0(members, members) * memberWeights;
        numerator = memberWeights' * within;
        volume = memberMass' * memberWeights;
        association = numerator / volume;
        objective = objective + association;
        if nargout > 1
            gradient(members) = gradient(members) + ...
                (2 * within - association * memberMass) / volume;
        end
    end
end

function [H, labels] = update_indicator(S0, r, w, H, labels)
    m = size(H, 1);
    c = size(H, 2);
    Sw = (w * w') .* S0;
    mass = w .* r;
    SH = Sw * H;
    numerator = sum(H .* SH, 1);
    volume = mass' * H;
    count = sum(H, 1);
    order = randperm(m);

    for i = order
        from = labels(i);
        if count(from) <= 1
            continue;
        end

        remainingVolume = volume(from) - mass(i);
        if remainingVolume <= 0
            continue;
        end
        remainingNumerator = max(0, numerator(from) - 2 * SH(i, from));
        newNumerator = numerator + 2 * SH(i, :);
        newVolume = volume + mass(i);
        gain = remainingNumerator / remainingVolume - numerator(from) / volume(from) ...
            + newNumerator ./ newVolume - numerator ./ volume;
        gain(from) = 0;
        [bestGain, to] = max(gain);
        if bestGain <= 1e-12
            continue;
        end

        labels(i) = to;
        H(i, from) = 0;
        H(i, to) = 1;
        numerator(from) = remainingNumerator;
        numerator(to) = newNumerator(to);
        volume(from) = remainingVolume;
        volume(to) = newVolume(to);
        count(from) = count(from) - 1;
        count(to) = count(to) + 1;
        SH(:, from) = SH(:, from) - Sw(:, i);
        SH(:, to) = SH(:, to) + Sw(:, i);
    end
end
