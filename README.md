# kmeans-c

K-means clustering library in C99. Single header, single source file, no
dependencies beyond `libm`. Comes with a 2-D demo and a Python script for
plots and silhouette analysis.

## Features

- Lloyd's algorithm with configurable iteration cap and convergence tolerance
- K-means++ seeding (default) or uniform random init
- Multi-restart: runs `n_init` times, keeps the lowest-inertia result
- Inertia (WCSS) reported on the final result
- Demo computes inertia across `k = 2..8` for an elbow plot
- `visualize.py` renders ground-truth vs predicted clusters, the elbow curve,
  misclassified points, and a manually-computed silhouette plot

## Build

```bash
make            # build the demo binary `kmeans_demo`
make run        # build, then run; writes CSVs into data/
make valgrind   # run under valgrind with leak check
make clean      # remove build/ and the binary
```

Requires GCC or Clang with C99 support and `libm`. Visualization needs
Python 3, NumPy, and matplotlib.

## Usage

The library is two files: `include/kmeans.h` and `src/kmeans.c`. Drop them
into a project, include the header, link `-lm`.

```c
#include "kmeans.h"

KMeansParams params = kmeans_default_params();
params.init   = KMEANS_INIT_PLUSPLUS;
params.n_init = 10;
params.seed   = 123;  // 0 means seed from time(NULL)

// data is row-major, shape [n_samples, n_features]
KMeansResult r = kmeans(data, n_samples, n_features, k, &params);

// r.labels    : int[n_samples]
// r.centroids : double[k * n_features]
// r.inertia   : double (WCSS)
// r.n_iter    : iterations of the best run
// r.converged : 1 if it converged within max_iter

kmeans_free(&r);
```

End-to-end demo:

```bash
make run
python3 visualize.py
# output/kmeans_results.png
# output/silhouette.png
```

## Layout

```
include/kmeans.h    public API
src/kmeans.c        implementation
src/main.c          demo: synthetic blobs, clustering, elbow sweep
visualize.py        reads data/*.csv, writes output/*.png
Makefile
```

## License

MIT. See [LICENSE](LICENSE).
