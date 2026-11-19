# Reliability-Aware Anchor Normalized Clustering (RANC)

This repository contains the MATLAB implementations of **RANC-D** and **RANC-R** described by Deng, Zhu, Nie, and Yang in [Reliability-aware anchor normalized clustering via bipartite graph column reweighting](https://doi.org/10.1016/j.ins.2026.124405), *Information Sciences*, 766, article 124405 (2027).

RANC-D directly updates anchor cluster assignments with coordinate ascent and uses N2HI initialization. RANC-R learns a spectral embedding and runs K-means on the row-normalized embedding. Both learn anchor reliability weights and use those weights to propagate cluster labels to samples.

## Requirements

- MATLAB R2019a or newer for the benchmark demo and evaluation, which use [`matchpairs`](https://www.mathworks.com/help/matlab/ref/matchpairs.html) to compute clustering accuracy.
- Statistics and Machine Learning Toolbox for RANC-R's K-means step. RANC-D uses base MATLAB.

## Run the benchmark demo

Set the MATLAB current folder to the repository root, open `demo_RANC_datasets.m`, and click **Run**. With the default settings, the script runs both models once on the Seeds dataset and displays their clustering scores. You can also run the script from the Command Window.

```matlab
demo_RANC_datasets
```

Edit the settings at the top of the script to change the experiment.

| Setting | Default | Usage |
| --- | --- | --- |
| `datasetName` | `'Seeds'` | Choose a [bundled dataset](datasets/README.md) by filename, with or without `.mat`. |
| `method` | `'both'` | Use `'D'`, `'R'`, or `'both'`. Both methods use the same graph within a run. |
| `m` | `[]` | Enter a power of two satisfying `max(c,k) < m < n`. Leave it empty to select the smallest eligible power of two at least 32. |
| `k` | `10` | Set the number of nearest anchors per sample. |
| `lambda` | `10` | Set the reliability regularization coefficient. |
| `numRuns` | `1` | Set the number of runs. Each run uses a reproducible seed. |

Each MAT file contains a feature matrix `X` with one sample per row and a corresponding reference label vector `Y`. The script uses `Y` to get the number of clusters and evaluate the final labels.

After the run, the workspace contains `results`, `labels`, and `weights`. The script displays one row per run and method, including ACC, NMI, ARI, F1, and the elapsed time in seconds. Time covers graph construction and the solver. ACC uses optimal one-to-one label matching. NMI divides mutual information by the larger of the two label entropies. F1 is computed over pairs of samples. For one result row, `labels` and `weights` are vectors. With multiple rows, they are cell arrays aligned with `results`.

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
