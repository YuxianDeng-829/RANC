# Reliability-Aware Anchor Normalized Clustering (RANC)

This repository contains the MATLAB implementations of **RANC-D** and **RANC-R** described by Deng, Zhu, Nie, and Yang in [Reliability-aware anchor normalized clustering via bipartite graph column reweighting](https://doi.org/10.1016/j.ins.2026.124405), *Information Sciences*, 766, article 124405 (2027).

RANC-D directly updates anchor cluster assignments with coordinate ascent and uses N2HI initialization. RANC-R learns a spectral embedding and runs K-means on the row-normalized embedding. Both learn anchor reliability weights and use those weights to propagate cluster labels to samples.

## Requirements

- MATLAB R2019a or newer for the benchmark demo and evaluation, which use [`matchpairs`](https://www.mathworks.com/help/matlab/ref/matchpairs.html) to compute clustering accuracy.
- Statistics and Machine Learning Toolbox for RANC-R's K-means step. RANC-D uses base MATLAB.

## Run the benchmark demo

Set the MATLAB current folder to the repository root, or add that folder to the MATLAB path. The demo reads the requested MAT file from [`datasets/`](datasets/README.md). Every bundled file contains a feature matrix `X` with one sample per row and a corresponding reference label vector `Y`. The demo gets the number of clusters from `Y` and uses `Y` to evaluate the final labels.

```matlab
demo_RANC_datasets;                            % Seeds, RANC-D
demo_RANC_datasets('Seeds', 'both');            % RANC-D and RANC-R on the same graph
demo_RANC_datasets('jaffe50_uni', 'R');          % Jaffe50, RANC-R

[labels, weights, results] = demo_RANC_datasets('Seeds', 'both', 32, 10, 10, 20);
disp(results);
```

The call signature is `demo_RANC_datasets(datasetName, method, m, k, lambda, numRuns)`. Use the file name with or without `.mat`. Use `'D'`, `'R'`, or `'both'` for the method. Omitted or empty arguments use the defaults.

- The default dataset is Seeds, the default method is RANC-D, and the default number of runs is 1.
- The demo defaults to `k = 10` and `lambda = 10`. It uses the smallest power of two that is at least 32 and greater than both the class count and `k` when `m` is empty.
- Each run uses a reproducible seed. In `'both'` mode, both methods receive the same graph within that run.

The results table contains one row per run and method, including ACC, NMI, ARI, F1, and the elapsed time in seconds. Time covers graph construction and the solver. ACC uses optimal one-to-one label matching. NMI divides mutual information by the larger of the two label entropies. F1 is computed over pairs of samples. A single result row returns `labels` and `weights` as vectors. Multiple result rows return cell arrays aligned with the table.

For mean and standard deviation by method, run:

```matlab
groupsummary(results, 'Method', {'mean', 'std'}, {'ACC', 'NMI', 'ARI', 'F1', 'TimeSeconds'})
```

## Parameter values in the paper

| Parameter | Values |
| --- | --- |
| Number of anchors, `m` | 16, 32, 64, 128, 256, 512, 1024, 2048, subject to `c < m < n` |
| Nearest anchors per sample, `k` | 10 |
| Reliability regularization, `lambda` | 1, 5, 10, 50, 100, 500, 1000, 5000 |

## Call the solvers directly

Use `build_anchor_graph` with your own feature matrix, or supply an existing sample–anchor matrix `B`. Graph construction standardizes the features, builds the anchors with balanced binary hierarchical K-means (BKHK), and uses the local distance rule in Eq. (2) of the paper. The rows of the optional `anchors` output are in standardized feature coordinates.

```matlab
[B, anchors] = build_anchor_graph(X, m, k);
[labelsD, weightsD] = RANC_D(B, c, lambda, maxIter, tol);
[labelsR, weightsR] = RANC_R(B, c, lambda, maxIter, tol);
```

`B` has one row per sample and one column per anchor. It must be finite and nonnegative, with a positive sum in every row and at least `c` active columns. The solvers row-normalize `B` and use active anchors in the optimization. Both return an `n`-by-1 vector of one-based sample labels and an `m`-by-1 vector of weights in the original anchor order. An inactive anchor has weight one in the returned vector.

The optional solver defaults are `lambda = 1`, `tol = 1e-6`, and `maxIter = 100` for RANC-D or `30` for RANC-R, including initialization.

## Repository layout

| Path | Purpose |
| --- | --- |
| `RANC_D.m`, `RANC_R.m` | Discrete and relaxed solvers |
| `build_anchor_graph.m` | Feature standardization and BKHK anchor graph construction |
| `demo_RANC_datasets.m` | Benchmark demo with per-run clustering scores |
| `clustering_metrics.m` | Evaluation from reference and predicted labels |
| `private/` | Internal graph, projection, initialization, and label propagation routines |
| `datasets/` | Eight real benchmark MAT files and a data dictionary |
| `CITATION.cff` | GitHub citation metadata |

The solvers follow Algorithms 1 and 2 in the paper. The repository provides an independent BKHK implementation, so its initial graphs and resulting scores can differ from earlier experiment scripts.

## Citation

```bibtex
@article{Deng2027RANC,
  author  = {Deng, Yuxian and Zhu, Jianyong and Nie, Feiping and Yang, Hui},
  title   = {Reliability-aware anchor normalized clustering via bipartite graph column reweighting},
  journal = {Information Sciences},
  volume  = {766},
  pages   = {124405},
  year    = {2027},
  doi     = {10.1016/j.ins.2026.124405}
}
```

The anchor builder follows W. Zhu, F. Nie, and X. Li, *Fast Spectral Clustering with Efficient Large Graph Construction*, ICASSP 2017, pp. 2492–2496. RANC-D uses the N2HI procedure cited as reference [33] in the RANC article.

## License

The software license is pending selection by the rights holders. For redistribution terms of the benchmark data, consult the original dataset providers.

