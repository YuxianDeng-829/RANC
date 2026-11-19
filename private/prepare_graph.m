function [B, S0, r, active] = prepare_graph(B, c, lambda, maxIter, tol)
    validateattributes(B, {'numeric'}, {'2d', 'real', 'nonempty'}, ...
        'RANC', 'B');
    validateattributes(c, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
        'RANC', 'c');
    validateattributes(lambda, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, ...
        'RANC', 'lambda');
    validateattributes(maxIter, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
        'RANC', 'maxIter');
    validateattributes(tol, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, ...
        'RANC', 'tol');

    B = sparse(double(B));
    edges = nonzeros(B);
    if any(~isfinite(edges) | edges < 0)
        error('RANC:InvalidGraph', 'B must contain finite nonnegative edge weights.');
    end
    rowSum = full(sum(B, 2));
    if any(rowSum <= 0 | ~isfinite(rowSum))
        error('RANC:InvalidGraph', 'Each sample must have a finite positive row sum in B.');
    end
    [row, col, value] = find(B);
    B = sparse(row, col, value ./ rowSum(row), size(B, 1), size(B, 2));

    r = full(sum(B, 1))';
    active = r > 0;
    if nnz(active) < c
        error('RANC:TooFewAnchors', 'B must contain at least c active anchor columns.');
    end
    B = B(:, active);
    r = r(active);
    S0 = full(B' * B);
    m = numel(r);
    S0(1:m+1:end) = 0;
end
