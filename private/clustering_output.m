function [labels, w] = clustering_output(B, anchorLabels, c, weights, active)
    m = numel(weights);
    H = sparse((1:m)', anchorLabels, 1, m, c);
    scores = B * spdiags(weights, 0, m, m) * H;
    [~, labels] = max(scores, [], 2);
    labels = full(labels(:));

    w = ones(numel(active), 1);
    w(active) = weights;
end
