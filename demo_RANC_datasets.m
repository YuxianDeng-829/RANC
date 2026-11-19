datasetName = 'jaffe50_uni';
method = 'both';
m = [64];
k = 10;
lambda = 10;
numRuns = 20;

datasetNames = {'Seeds', 'Coil20', 'MSRA25', 'ORL64', 'PalmData25', ...
                'Pendigits', 'dig1-10_uni', 'jaffe50_uni'};
datasetName = regexprep(datasetName, '(?i)\.mat$', '');
datasetName = validatestring(datasetName, datasetNames, 'demo_RANC_datasets', 'datasetName');
method = validatestring(method, {'D', 'R', 'both'}, 'demo_RANC_datasets', 'method');
validateattributes(k, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    'demo_RANC_datasets', 'k');
validateattributes(lambda, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, ...
    'demo_RANC_datasets', 'lambda');
validateattributes(numRuns, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, 'demo_RANC_datasets', 'numRuns');
numRuns = double(numRuns);

repoRoot = fileparts(mfilename('fullpath'));
fileName = [datasetName, '.mat'];
dataFile = fullfile(repoRoot, 'datasets', fileName);
if exist(dataFile, 'file') ~= 2
    error('RANC:DatasetNotFound', 'Place %s in the datasets folder beside this demo.', fileName);
end
data = load(dataFile, 'X', 'Y');
if ~isfield(data, 'X') || ~isfield(data, 'Y')
    error('RANC:InvalidDataset', '%s must contain the variables X and Y.', fileName);
end
n = size(data.X, 1);
validateattributes(data.Y, {'numeric'}, {'vector', 'real', 'finite', 'nonempty', 'numel', n}, ...
    'demo_RANC_datasets', 'Y');
c = numel(unique(data.Y(:)));
if isempty(m)
    m = max(32, 2^nextpow2(max(c, double(k)) + 1));
end
validateattributes(m, {'numeric'}, {'scalar', 'real', 'finite', 'integer', '>', c, '<', n}, ...
    'demo_RANC_datasets', 'm');

methods = {method};
if strcmp(method, 'both')
    methods = {'D', 'R'};
end
methodCount = numel(methods);
rowCount = numRuns * methodCount;
methodNames = cell(rowCount, 1);
runNumbers = zeros(rowCount, 1);
labels = cell(rowCount, 1);
weights = cell(rowCount, 1);
acc = zeros(rowCount, 1);
nmi = zeros(rowCount, 1);
ari = zeros(rowCount, 1);
f1 = zeros(rowCount, 1);
elapsedSeconds = zeros(rowCount, 1);

previousState = rng;
try
    for runIdx = 1:numRuns
        rng(42 + runIdx - 1, 'twister');
        graphClock = tic;
        B = build_anchor_graph(data.X, m, k);
        graphSeconds = toc(graphClock);
        solverState = rng;

        for methodIdx = 1:methodCount
            row = (runIdx - 1) * methodCount + methodIdx;
            model = methods{methodIdx};
            rng(solverState);
            solverClock = tic;
            switch model
                case 'D'
                    [labels{row}, weights{row}] = RANC_D(B, c, lambda);
                case 'R'
                    [labels{row}, weights{row}] = RANC_R(B, c, lambda);
            end
            elapsedSeconds(row) = graphSeconds + toc(solverClock);

            scores = clustering_metrics(data.Y, labels{row});
            methodNames{row} = ['RANC-', model];
            runNumbers(row) = runIdx;
            acc(row) = scores.ACC;
            nmi(row) = scores.NMI;
            ari(row) = scores.ARI;
            f1(row) = scores.F1;
        end
    end
catch runError
    rng(previousState);
    rethrow(runError);
end
rng(previousState);

results = table(repmat({datasetName}, rowCount, 1), methodNames, runNumbers, ...
    repmat(c, rowCount, 1), repmat(m, rowCount, 1), repmat(k, rowCount, 1), ...
    repmat(lambda, rowCount, 1), acc, nmi, ari, f1, elapsedSeconds, ...
    'VariableNames', {'Dataset', 'Method', 'Run', 'Clusters', 'Anchors', ...
                      'Neighbors', 'Lambda', 'ACC', 'NMI', 'ARI', 'F1', 'TimeSeconds'});

if rowCount == 1
    labels = labels{1};
    weights = weights{1};
end
disp(results);
