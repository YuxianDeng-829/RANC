function [labels, w] = RANC_R(B, c, lambda, maxIter, tol)
    if nargin < 3 || isempty(lambda)
        lambda = 1;
    end
    if nargin < 4 || isempty(maxIter)
        maxIter = 30;
    end
    if nargin < 5 || isempty(tol)
        tol = 1e-6;
    end
    [B, S0, r, active] = prepare_graph(B, c, lambda, maxIter, tol);
    c = double(c);
    lambda = double(lambda);
    maxIter = double(maxIter);

    delta = 1e-6;
    relativeMass = r / sum(r);
    weights = ones(numel(r), 1);
    [G, objective] = relaxed_profile(S0, r, weights, c, lambda);

    for iter = 2:maxIter
        M = S0 .* (G * G');
        gradient = 2 * M * weights - lambda * (2 * relativeMass .* (weights - 1));
        direction = mass_tangent(gradient, r);
        step = 1;
        accepted = false;

        while step >= 1e-8
            candidate = project_weights(weights + step * direction, r, delta);
            if isequal(candidate, weights)
                break;
            end
            [candidateG, candidateValue] = relaxed_profile(S0, r, candidate, c, lambda);
            if candidateValue >= objective
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
        G = candidateG;
        objective = candidateValue;
        if gain < tol
            break;
        end
    end

    rowNorm = sqrt(sum(G.^2, 2));
    rowNorm(rowNorm == 0) = 1;
    G = G ./ rowNorm;
    if c == 1
        anchorLabels = ones(numel(r), 1);
    elseif c == numel(r)
        anchorLabels = (1:c)';
    else
        anchorLabels = kmeans(G, c, 'Replicates', 20, 'MaxIter', 1000, ...
            'EmptyAction', 'singleton', 'Display', 'off');
    end

    [labels, w] = clustering_output(B, anchorLabels, c, weights, active);
end

function [G, objective] = relaxed_profile(S0, r, w, c, lambda)
    m = numel(r);
    scale = sqrt(w) ./ sqrt(r);
    C = (scale .* S0) .* scale';
    C = C / 2 + C' / 2;

    useDense = m <= 64 || c >= m - 1 || ~any(C(:));
    if ~useDense
        try
            [P, eigenvalues, flag] = eigs(C, c, 'largestreal');
            useDense = flag ~= 0 || any(~isfinite(eigenvalues(:)));
        catch
            useDense = true;
        end
    end
    if useDense
        [vectors, values] = eig(C);
        [sortedValues, order] = sort(diag(values), 'descend');
        P = vectors(:, order(1:c));
        eigenvalues = diag(sortedValues(1:c));
    end

    G = P ./ (sqrt(r) .* sqrt(w));
    objective = sum(diag(eigenvalues)) - lambda * sum((r / sum(r)) .* (w - 1).^2);
end
