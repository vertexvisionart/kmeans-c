/*
 * Demo program: generates a 2-D synthetic dataset with five Gaussian blobs,
 * runs k-means on it, and writes CSVs that the visualize.py script picks up.
 * Also computes inertia across k=2..8 for an elbow plot.
 */

#include "kmeans.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_CLUSTERS_TRUE 5
#define N_SAMPLES 500
#define N_FEATURES 2
#define CLUSTER_STD 0.7
#define DATASET_SEED 42

static const double TRUE_CENTERS[N_CLUSTERS_TRUE][N_FEATURES] = {
    {0.0, 0.0}, {6.0, 2.0}, {-4.0, 5.0}, {3.0, -5.0}, {-2.0, -3.0}};

/* Standard normal sample via Marsaglia polar method. */
static double randn(void) {
  static int has_spare = 0;
  static double spare;

  if (has_spare) {
    has_spare = 0;
    return spare;
  }
  has_spare = 1;

  double u, v, s;
  do {
    u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);

  s = sqrt(-2.0 * log(s) / s);
  spare = v * s;
  return u * s;
}

static void generate_dataset(double *data, int *true_labels, size_t n_samples,
                             size_t n_features, int n_clusters,
                             double std_dev) {
  size_t per_cluster = n_samples / (size_t)n_clusters;

  for (size_t i = 0; i < n_samples; i++) {
    int c = (int)(i / per_cluster);
    if (c >= n_clusters)
      c = n_clusters - 1;
    true_labels[i] = c;

    for (size_t f = 0; f < n_features; f++) {
      data[i * n_features + f] = TRUE_CENTERS[c][f] + randn() * std_dev;
    }
  }
}

static void save_dataset_csv(const char *path, const double *data,
                             const int *true_labels, const int *pred_labels,
                             size_t n_samples, size_t n_features) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "cannot open file: %s\n", path);
    return;
  }

  for (size_t j = 0; j < n_features; j++)
    fprintf(f, "x%zu,", j);
  fprintf(f, "true_label,pred_label\n");

  for (size_t i = 0; i < n_samples; i++) {
    for (size_t j = 0; j < n_features; j++)
      fprintf(f, "%.6f,", data[i * n_features + j]);
    fprintf(f, "%d,%d\n", true_labels[i], pred_labels[i]);
  }
  fclose(f);
  printf("  dataset written: %s\n", path);
}

static void save_centroids_csv(const char *path, const double *centroids, int k,
                               size_t n_features) {
  FILE *f = fopen(path, "w");
  if (!f)
    return;

  for (size_t j = 0; j < n_features; j++)
    fprintf(f, "x%zu,", j);
  fprintf(f, "cluster\n");

  for (int c = 0; c < k; c++) {
    for (size_t j = 0; j < n_features; j++)
      fprintf(f, "%.6f,", centroids[c * n_features + j]);
    fprintf(f, "%d\n", c);
  }
  fclose(f);
  printf("  centroids written: %s\n", path);
}

static void save_elbow_csv(const char *path, const int *k_vals,
                           const double *inertias, int n_k) {
  FILE *f = fopen(path, "w");
  if (!f)
    return;
  fprintf(f, "k,inertia\n");
  for (int i = 0; i < n_k; i++)
    fprintf(f, "%d,%.4f\n", k_vals[i], inertias[i]);
  fclose(f);
  printf("  elbow data written: %s\n", path);
}

/* matrix[p * k + t] = how often predicted label p coincided with true t. */
static void confusion_matrix(const int *true_labels, const int *pred_labels,
                             size_t n_samples, int k, int *matrix) {
  memset(matrix, 0, (size_t)k * (size_t)k * sizeof(int));
  for (size_t i = 0; i < n_samples; i++) {
    int t = true_labels[i];
    int p = pred_labels[i];
    if (t >= 0 && t < k && p >= 0 && p < k)
      matrix[p * k + t]++;
  }
}

int main(void) {
  printf("k-means clustering demo (C99)\n\n");

  /* 1. Synthesize the dataset. */
  srand(DATASET_SEED);

  size_t n_samples = N_SAMPLES;
  size_t n_features = N_FEATURES;

  double *data = (double *)malloc(n_samples * n_features * sizeof(double));
  int *true_labels = (int *)malloc(n_samples * sizeof(int));

  if (!data || !true_labels) {
    fprintf(stderr, "out of memory.\n");
    free(data);
    free(true_labels);
    return 1;
  }

  generate_dataset(data, true_labels, n_samples, n_features, N_CLUSTERS_TRUE,
                   CLUSTER_STD);

  printf("dataset:\n");
  printf("  samples:       %zu\n", n_samples);
  printf("  features:      %zu\n", n_features);
  printf("  true clusters: %d\n", N_CLUSTERS_TRUE);
  printf("  std dev:       %.2f\n\n", CLUSTER_STD);

  /* 2. Cluster. */
  int k = N_CLUSTERS_TRUE;

  KMeansParams params = kmeans_default_params();
  params.init = KMEANS_INIT_PLUSPLUS;
  params.n_init = 10;
  params.seed = 123;

  printf("k-means parameters:\n");
  printf("  k:        %d\n", k);
  printf("  init:     k-means++\n");
  printf("  restarts: %d\n", params.n_init);
  printf("  max iter: %d\n\n", params.max_iter);

  printf("clustering...\n");
  KMeansResult result = kmeans(data, n_samples, n_features, k, &params);

  if (!result.labels) {
    fprintf(stderr, "clustering failed.\n");
    free(data);
    free(true_labels);
    return 1;
  }

  /* 3. Report. */
  printf("\nresults:\n");
  printf("  iterations: %d\n", result.n_iter);
  printf("  converged:  %s\n",
         result.converged ? "yes" : "no (hit iteration cap)");
  printf("  inertia:    %.2f\n\n", result.inertia);

  printf("found centroids:\n");
  for (int c = 0; c < k; c++) {
    printf("  cluster %d: (", c);
    for (size_t f = 0; f < n_features; f++) {
      printf("%.3f%s", result.centroids[c * n_features + f],
             f < n_features - 1 ? ", " : "");
    }
    printf(")\n");
  }

  printf("\ntrue centers:\n");
  for (int c = 0; c < N_CLUSTERS_TRUE; c++) {
    printf("  cluster %d: (", c);
    for (size_t f = 0; f < n_features; f++) {
      printf("%.3f%s", TRUE_CENTERS[c][f],
             f < (size_t)N_FEATURES - 1 ? ", " : "");
    }
    printf(")\n");
  }

  /* 4. Cluster sizes + confusion matrix. */
  printf("\ncluster sizes:\n");
  int *cluster_sizes = (int *)calloc((size_t)k, sizeof(int));
  for (size_t i = 0; i < n_samples; i++)
    cluster_sizes[result.labels[i]]++;
  for (int c = 0; c < k; c++) {
    int bar_len = cluster_sizes[c] / 5;
    printf("  [%d] %3d |", c, cluster_sizes[c]);
    for (int b = 0; b < bar_len; b++)
      printf("#");
    printf("\n");
  }
  free(cluster_sizes);

  printf("\nconfusion matrix (pred x true):\n");
  int *cm = (int *)calloc((size_t)k * (size_t)k, sizeof(int));
  confusion_matrix(true_labels, result.labels, n_samples, k, cm);
  printf("       ");
  for (int j = 0; j < k; j++)
    printf("T%-3d ", j);
  printf("\n");
  for (int i = 0; i < k; i++) {
    printf("  P%d  |", i);
    for (int j = 0; j < k; j++)
      printf("%-4d ", cm[i * k + j]);
    printf("\n");
  }
  free(cm);

  /* 5. Persist results. */
  printf("\nwriting CSVs...\n");
  save_dataset_csv("data/dataset.csv", data, true_labels, result.labels,
                   n_samples, n_features);
  save_centroids_csv("data/centroids.csv", result.centroids, k, n_features);

  /* 6. Elbow sweep over k = 2..8. */
  printf("\nelbow sweep (k = 2..8)...\n");
  int k_vals[7] = {2, 3, 4, 5, 6, 7, 8};
  double inertias[7];
  KMeansParams ep = kmeans_default_params();
  ep.n_init = 5;
  ep.seed = 42;

  for (int i = 0; i < 7; i++) {
    KMeansResult r = kmeans(data, n_samples, n_features, k_vals[i], &ep);
    inertias[i] = r.inertia;
    printf("  k=%d  inertia=%.1f\n", k_vals[i], inertias[i]);
    kmeans_free(&r);
  }
  save_elbow_csv("data/elbow.csv", k_vals, inertias, 7);

  kmeans_free(&result);
  free(data);
  free(true_labels);

  printf("\ndone. Run `python3 visualize.py` to render the plots.\n");
  return 0;
}
