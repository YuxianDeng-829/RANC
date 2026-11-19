function label = n2hi_init(A, c)

    A = full(double(A));
    A(~isfinite(A)) = 0;
    A = max(A, 0);
    A = (A + A') / 2;

    m = size(A, 1);
    A(1:m+1:end) = 0;

    if c > m
        error('RANC:TooFewAnchors', 'N2HI requires at least c anchors.');
    end

    if c == m
        label = (1:m)';
        return;
    end

    if nnz(A) == 0
        label = deterministic_greedy_init(A, c);
        return;
    end

    groups = cell(m, 1);
    for i = 1:m
        groups{i} = i;
    end

    Acur = A;

    hierarchy_labels = {};
    hierarchy_graphs = {};
    hierarchy_counts = [];

    maxLayer = m;

    for layer = 1:maxLayer

        curN = length(groups);

        if curN <= 1
            break;
        end

        local_label = first_neighbor_components(Acur);
        K = max(local_label);

        if K == curN
            local_label = force_merge_one_pair(Acur);
            K = max(local_label);
        end

        newGroups = cell(K, 1);
        for kk = 1:K
            idx = find(local_label == kk);
            temp = [];
            for t = 1:length(idx)
                temp = [temp, groups{idx(t)}];
            end
            newGroups{kk} = temp;
        end

        label_orig = zeros(m, 1);
        for kk = 1:K
            label_orig(newGroups{kk}) = kk;
        end

        Y = full(sparse(1:curN, local_label, 1, curN, K));
        sizes = sum(Y, 1)';
        denom = sizes * sizes';
        A_next = Y' * Acur * Y;
        A_next = A_next ./ max(denom, eps);
        A_next = (A_next + A_next') / 2;
        A_next(1:K+1:end) = 0;
        A_next(~isfinite(A_next)) = 0;
        A_next = max(A_next, 0);

        hierarchy_labels{end+1} = label_orig;
        hierarchy_graphs{end+1} = A_next;
        hierarchy_counts(end+1) = K;

        if K == c
            label = label_orig(:);
            label = relabel_consecutive(label);
            return;
        end

        groups = newGroups;
        Acur = A_next;

        if K < c
            break;
        end
    end

    valid_layers = find(hierarchy_counts > c);

    if isempty(valid_layers)
        label = deterministic_greedy_init(A, c);
        return;
    end

    [~, pos] = min(hierarchy_counts(valid_layers) - c);
    chosen_layer = valid_layers(pos);

    label = hierarchy_labels{chosen_layer};
    Aref = hierarchy_graphs{chosen_layer};
    K = hierarchy_counts(chosen_layer);

    while K > c

        Atemp = Aref;
        Atemp(1:K+1:end) = -inf;

        [maxVal, linearIdx] = max(Atemp(:));

        if ~isfinite(maxVal)
            u = 1;
            v = 2;
        else
            [u, v] = ind2sub([K, K], linearIdx);
        end

        if u == v
            v = mod(u, K) + 1;
        end

        if u > v
            temp = u;
            u = v;
            v = temp;
        end

        label(label == v) = u;
        label(label > v) = label(label > v) - 1;

        for ii = 1:K
            if ii ~= u && ii ~= v
                Aref(ii, u) = (Aref(ii, u) + Aref(ii, v)) / 2;
                Aref(u, ii) = Aref(ii, u);
            end
        end

        Aref(v, :) = [];
        Aref(:, v) = [];

        K = K - 1;
        Aref(1:K+1:end) = 0;
    end

    label = relabel_consecutive(label(:));

    if length(unique(label)) < c
        label = deterministic_greedy_init(A, c);
    end

end

function label = first_neighbor_components(A)

    A = full(double(A));
    A(~isfinite(A)) = 0;
    A = max(A, 0);
    A = (A + A') / 2;

    n = size(A, 1);
    A(1:n+1:end) = -inf;

    parent = 1:n;

    for i = 1:n
        [val, j] = max(A(i, :));

        if ~isfinite(val) || val <= 0
            continue;
        end

        parent = uf_union(parent, i, j);
    end

    root = zeros(n, 1);
    for i = 1:n
        [parent, root(i)] = uf_find(parent, i);
    end

    label = relabel_consecutive(root);

end

function label = force_merge_one_pair(A)

    A = full(double(A));
    A(~isfinite(A)) = 0;
    A = max(A, 0);

    n = size(A, 1);
    A(1:n+1:end) = -inf;

    [maxVal, idx] = max(A(:));

    label = (1:n)';

    if ~isfinite(maxVal)
        if n >= 2
            label(2) = 1;
        end
    else
        [u, v] = ind2sub([n, n], idx);
        if u ~= v
            label(v) = u;
        end
    end

    label = relabel_consecutive(label);

end

function [parent, root] = uf_find(parent, x)

    root = x;

    while parent(root) ~= root
        root = parent(root);
    end

    while parent(x) ~= x
        next = parent(x);
        parent(x) = root;
        x = next;
    end

end

function parent = uf_union(parent, a, b)

    [parent, ra] = uf_find(parent, a);
    [parent, rb] = uf_find(parent, b);

    if ra ~= rb
        if ra < rb
            parent(rb) = ra;
        else
            parent(ra) = rb;
        end
    end

end

function label = deterministic_greedy_init(A, c)

    A = full(double(A));
    A(~isfinite(A)) = 0;
    A = max(A, 0);
    A = (A + A') / 2;

    m = size(A, 1);
    A(1:m+1:end) = 0;

    if c > m
        error('RANC:TooFewAnchors', 'N2HI requires at least c anchors.');
    end

    deg = sum(A, 2);

    [~, first] = max(deg);
    centers = zeros(c, 1);
    centers(1) = first;

    selected = false(m, 1);
    selected(first) = true;

    for t = 2:c

        score = inf(m, 1);

        for i = 1:m
            if selected(i)
                score(i) = -inf;
            else
                score(i) = -max(A(i, centers(1:t-1)));
            end
        end

        [~, idx] = max(score);
        centers(t) = idx;
        selected(idx) = true;
    end

    Sim = A(:, centers);
    [~, label] = max(Sim, [], 2);

    for q = 1:c
        label(centers(q)) = q;
    end

    label = relabel_consecutive(label(:));

end

function label_new = relabel_consecutive(label)

    label = label(:);
    [~, first, groups] = unique(label, 'first');
    [~, order] = sort(first);
    ids = zeros(numel(order), 1);
    ids(order) = (1:numel(order))';
    label_new = ids(groups);

end
