/*
 * K-means implementation in C99.
 *
 * Standard Lloyd's algorithm: pick initial centroids, assign points to the
 * nearest centroid, recompute centroids as cluster means, repeat until
 * centroids stop moving (or the iteration cap is hit). The public entry point
 * runs this whole loop n_init times and keeps the run with the lowest WCSS.
 */

#include "kmeans.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Squared Euclidean distance between two feature vectors. */
static double sq_dist(const double *a, const double *b, size_t n_features) {
  double dist = 0.0;
  for (size_t f = 0; f < n_features; f++) {
    double diff = a[f] - b[f];
    dist += diff * diff;
  }
  return dist;
}

/* Assign each point to its nearest centroid. Returns 1 if any label changed. */
static int assign_labels(const double *data, size_t n_samples,
                         size_t n_features, const double *centroids, int k,
                         int *labels) {
  int changed = 0;

  for (size_t i = 0; i < n_samples; i++) {
    const double *point = data + i * n_features;
    int best_cluster = 0;
    double best_dist = HUGE_VAL;

    for (int c = 0; c < k; c++) {
      double d = sq_dist(point, centroids + c * n_features, n_features);
      if (d < best_dist) {
        best_dist = d;
        best_cluster = c;
      }
    }

    if (labels[i] != best_cluster) {
      labels[i] = best_cluster;
      changed = 1;
    }
  }
  return changed;
}

/*
 * Recompute each centroid as the mean of its assigned points.
 * Empty clusters keep their previous centroid. Returns the largest
 * Euclidean shift across all centroids.
 */
static double update_centroids(const double *data, size_t n_samples,
                               size_t n_features, int k, const int *labels,
                               double *centroids) {
  double *sums = (double *)calloc((size_t)k * n_features, sizeof(double));
  int *counts = (int *)calloc((size_t)k, sizeof(int));

  if (!sums || !counts) {
    free(sums);
    free(counts);
    return 0.0;
  }

  for (size_t i = 0; i < n_samples; i++) {
    int c = labels[i];
    counts[c]++;
    for (size_t f = 0; f < n_features; f++) {
      sums[c * n_features + f] += data[i * n_features + f];
    }
  }

  double max_shift = 0.0;
  for (int c = 0; c < k; c++) {
    if (counts[c] == 0)
      continue;

    double shift = 0.0;
    for (size_t f = 0; f < n_features; f++) {
      double new_val = sums[c * n_features + f] / counts[c];
      double diff = new_val - centroids[c * n_features + f];
      shift += diff * diff;
      centroids[c * n_features + f] = new_val;
    }
    if (shift > max_shift)
      max_shift = shift;
  }

  free(sums);
  free(counts);
  return sqrt(max_shift);
}

/* Pick k distinct random points as starting centroids. */
static void init_random(const double *data, size_t n_samples, size_t n_features,
                        int k, double *centroids) {
  int *indices = (int *)malloc(n_samples * sizeof(int));
  if (!indices)
    return;

  for (size_t i = 0; i < n_samples; i++)
    indices[i] = (int)i;

  /* Partial Fisher-Yates shuffle: take the first k. */
  for (int i = 0; i < k; i++) {
    int j = i + rand() % (int)(n_samples - (size_t)i);
    int tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
    memcpy(centroids + i * n_features, data + (size_t)indices[i] * n_features,
           n_features * sizeof(double));
  }
  free(indices);
}

/*
 * K-means++ seeding. Each new centroid is sampled with probability
 * proportional to the squared distance to the closest already-chosen centroid.
 * Yields more stable results than uniform random init.
 */
static void init_plusplus(const double *data, size_t n_samples,
                          size_t n_features, int k, double *centroids) {
  double *dist2 = (double *)malloc(n_samples * sizeof(double));
  if (!dist2)
    return;

  /* First centroid is sampled uniformly. */
  int first = rand() % (int)n_samples;
  memcpy(centroids, data + (size_t)first * n_features,
         n_features * sizeof(double));

  for (int c = 1; c < k; c++) {
    double total = 0.0;

    for (size_t i = 0; i < n_samples; i++) {
      double min_d = HUGE_VAL;
      for (int j = 0; j < c; j++) {
        double d = sq_dist(data + i * n_features,
                           centroids + (size_t)j * n_features, n_features);
        if (d < min_d)
          min_d = d;
      }
      dist2[i] = min_d;
      total += min_d;
    }

    /* Weighted sample: farther points are more likely to be picked. */
    double target = ((double)rand() / RAND_MAX) * total;
    double cumsum = 0.0;
    size_t chosen = 0;
    for (size_t i = 0; i < n_samples; i++) {
      cumsum += dist2[i];
      if (cumsum >= target) {
        chosen = i;
        break;
      }
    }
    memcpy(centroids + (size_t)c * n_features, data + chosen * n_features,
           n_features * sizeof(double));
  }
  free(dist2);
}

typedef struct {
  double *centroids;
  int *labels;
  double inertia;
  int n_iter;
  int converged;
} RunResult;

/* Single Lloyd run from one seeding. */
static RunResult run_once(const double *data, size_t n_samples,
                          size_t n_features, int k,
                          const KMeansParams *params) {
  RunResult res = {NULL, NULL, HUGE_VAL, 0, 0};

  res.centroids = (double *)malloc((size_t)k * n_features * sizeof(double));
  res.labels = (int *)malloc(n_samples * sizeof(int));
  if (!res.centroids || !res.labels) {
    free(res.centroids);
    free(res.labels);
    res.centroids = NULL;
    res.labels = NULL;
    return res;
  }

  for (size_t i = 0; i < n_samples; i++)
    res.labels[i] = -1;

  if (params->init == KMEANS_INIT_PLUSPLUS)
    init_plusplus(data, n_samples, n_features, k, res.centroids);
  else
    init_random(data, n_samples, n_features, k, res.centroids);

  for (int iter = 0; iter < params->max_iter; iter++) {
    res.n_iter = iter + 1;

    assign_labels(data, n_samples, n_features, res.centroids, k, res.labels);

    double shift = update_centroids(data, n_samples, n_features, k, res.labels,
                                    res.centroids);

    if (shift < params->tolerance) {
      res.converged = 1;
      break;
    }
  }

  /* Final assignment + inertia after the last centroid update. */
  assign_labels(data, n_samples, n_features, res.centroids, k, res.labels);
  res.inertia =
      kmeans_inertia(data, n_samples, n_features, res.labels, res.centroids, k);
  return res;
}

KMeansParams kmeans_default_params(void) {
  KMeansParams p;
  p.max_iter = KMEANS_MAX_ITER;
  p.tolerance = KMEANS_TOLERANCE;
  p.n_init = KMEANS_N_INIT;
  p.init = KMEANS_INIT_PLUSPLUS;
  p.seed = 0;
  return p;
}

KMeansResult kmeans(const double *data, size_t n_samples, size_t n_features,
                    int k, const KMeansParams *params) {
  KMeansResult result = {NULL, NULL, HUGE_VAL, 0, 0};

  if (!data || n_samples == 0 || n_features == 0 || k <= 0) {
    fprintf(stderr, "[kmeans] error: invalid input.\n");
    return result;
  }
  if ((size_t)k > n_samples) {
    fprintf(stderr, "[kmeans] error: k (%d) > n_samples (%zu).\n", k,
            n_samples);
    return result;
  }

  KMeansParams p = params ? *params : kmeans_default_params();

  unsigned int seed = (p.seed != 0) ? p.seed : (unsigned int)time(NULL);
  srand(seed);

  /* Multi-restart: keep the run with the lowest inertia. */
  for (int run = 0; run < p.n_init; run++) {
    RunResult r = run_once(data, n_samples, n_features, k, &p);

    if (!r.labels || !r.centroids) {
      free(r.centroids);
      free(r.labels);
      continue;
    }

    if (r.inertia < result.inertia) {
      kmeans_free(&result);

      result.labels = r.labels;
      result.centroids = r.centroids;
      result.inertia = r.inertia;
      result.n_iter = r.n_iter;
      result.converged = r.converged;
    } else {
      free(r.centroids);
      free(r.labels);
    }
  }

  if (!result.labels) {
    fprintf(stderr, "[kmeans] error: every restart failed.\n");
  }
  return result;
}

void kmeans_free(KMeansResult *result) {
  if (!result)
    return;
  free(result->labels);
  result->labels = NULL;
  free(result->centroids);
  result->centroids = NULL;
  result->inertia = 0.0;
  result->n_iter = 0;
  result->converged = 0;
}

double kmeans_inertia(const double *data, size_t n_samples, size_t n_features,
                      const int *labels, const double *centroids, int k) {
  (void)k;
  double total = 0.0;
  for (size_t i = 0; i < n_samples; i++) {
    total += sq_dist(data + i * n_features,
                     centroids + (size_t)labels[i] * n_features, n_features);
  }
  return total;
}
