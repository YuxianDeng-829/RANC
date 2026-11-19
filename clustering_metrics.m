function scores = clustering_metrics(truth, labels)
    validateattributes(truth, {'numeric', 'logical'}, ...
        {'real', 'finite', 'nonempty', 'vector'}, 'clustering_metrics', 'truth');
    n = numel(truth);
    validateattributes(labels, {'numeric', 'logical'}, ...
        {'real', 'finite', 'nonempty', 'vector', 'numel', n}, 'clustering_metrics', 'labels');

    [trueClasses, ~, trueIndex] = unique(truth(:));
    [predClasses, ~, predIndex] = unique(labels(:));
    counts = accumarray([trueIndex, predIndex], 1, [numel(trueClasses), numel(predClasses)]);
    pairs = matchpairs(-counts, n + 1);
    matchedCounts = counts(sub2ind(size(counts), pairs(:, 1), pairs(:, 2)));
    scores.ACC = sum(matchedCounts) / n;

    trueCounts = sum(counts, 2);
    predCounts = sum(counts, 1);
    trueProb = trueCounts / n;
    predProb = predCounts / n;
    jointProb = counts / n;
    independentProb = trueProb * predProb;
    observed = counts > 0;
    mutualInfo = sum(jointProb(observed) .* log2(jointProb(observed) ./ independentProb(observed)));
    trueEntropy = -sum(trueProb .* log2(trueProb));
    predEntropy = -sum(predProb .* log2(predProb));
    normalizer = max(trueEntropy, predEntropy);
    if normalizer > 0
        scores.NMI = min(1, max(0, mutualInfo / normalizer));
    else
        scores.NMI = 1;
    end

    truePairs = sum(trueCounts .* (trueCounts - 1)) / 2;
    predPairs = sum(predCounts .* (predCounts - 1)) / 2;
    sharedPairs = sum(counts(:) .* (counts(:) - 1)) / 2;
    pairSum = truePairs + predPairs;
    if pairSum > 0
        scores.F1 = 2 * sharedPairs / pairSum;
    else
        scores.F1 = 1;
    end

    allPairs = n * (n - 1) / 2;
    scores.ARI = 0;
    if allPairs > 0
        expected = (truePairs / allPairs) * predPairs;
        denominator = pairSum / 2 - expected;
        if denominator > 0
            scores.ARI = (sharedPairs - expected) / denominator;
        end
    end
end
