function [B, anchors] = build_anchor_graph(X, m, k)
    validateattributes(X, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, ...
        'build_anchor_graph', 'X');
    validateattributes(m, {'numeric'}, {'scalar', 'real', 'finite', 'integer', '>=', 2, '<=', size(X, 1)}, ...
        'build_anchor_graph', 'm');
    validateattributes(k, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive', '<', m}, ...
        'build_anchor_graph', 'k');

    depth = log2(double(m));
    if depth ~= floor(depth)
        error('RANC:InvalidAnchorCount', 'BKHK requires m to be a power of two.');
    end
    X = full(double(X));
    m = double(m);
    k = double(k);
    n = size(X, 1);
    scale = std(X, 0, 1);
    scale(scale == 0) = 1;
    X = (X - mean(X, 1)) ./ scale;
    if any(~isfinite(X(:)))
        error('RANC:InvalidFeatures', 'Feature standardization produced nonfinite values.');
    end

    anchors = hierarchical_centers(X, depth);
    B = sparse(n, m);

    for first = 1:1024:n
        rows = first:min(n, first + 1023);
        Xblock = X(rows, :);
        distances = squared_distances(Xblock, anchors);
        [nearestDistances, nearest] = sort(distances, 2, 'ascend');
        gains = nearestDistances(:, k + 1) - nearestDistances(:, 1:k);
        total = sum(gains, 2);
        ties = total == 0;
        total(ties) = 1;
        gains = gains ./ total;
        gains(ties, :) = 1 / k;

        sampleIndices = repmat((1:numel(rows))', 1, k);
        anchorIndices = nearest(:, 1:k);
        B(rows, :) = sparse(sampleIndices(:), anchorIndices(:), gains(:), numel(rows), m);
    end
end

function centers = hierarchical_centers(X, depth)
    n = size(X, 1);
    order = randperm(n);
    left = false(n, 1);
    left(order(1:floor(n / 2))) = true;

    for iter = 1:10
        centers = [mean(X(left, :), 1); mean(X(~left, :), 1)];
        distances = squared_distances(X, centers);
        margin = distances(:, 1) - distances(:, 2);
        [~, order] = sort(margin, 'ascend');
        if depth == 1
            split = min(n - 1, max(1, nnz(margin < 0)));
        else
            split = floor(n / 2);
        end
        left = false(n, 1);
        left(order(1:split)) = true;
    end

    if depth == 1
        centers = [mean(X(left, :), 1); mean(X(~left, :), 1)];
    else
        centers = [hierarchical_centers(X(left, :), depth - 1); ...
                   hierarchical_centers(X(~left, :), depth - 1)];
    end
end

function distances = squared_distances(X, centers)
    distances = sum(X.^2, 2) + sum(centers.^2, 2)' - 2 * (X * centers');
    distances = max(distances, 0);
end
