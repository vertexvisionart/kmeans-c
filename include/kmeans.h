#ifndef KMEANS_H
#define KMEANS_H

#include <stddef.h>

#define KMEANS_MAX_ITER 300   /* max iterations per run */
#define KMEANS_TOLERANCE 1e-6 /* convergence threshold (centroid shift) */
#define KMEANS_N_INIT 10      /* number of independent restarts */

/* Initialization strategy for the starting centroids. */
typedef enum {
  KMEANS_INIT_RANDOM = 0,   /* k random data points */
  KMEANS_INIT_PLUSPLUS = 1  /* k-means++ seeding */
} KMeansInitMethod;

typedef struct {
  int max_iter;          /* iteration cap */
  double tolerance;      /* stop when max centroid shift drops below this */
  int n_init;            /* how many independent runs to perform */
  KMeansInitMethod init; /* initialization strategy */
  unsigned int seed;     /* RNG seed; 0 = seed from time(NULL) */
} KMeansParams;

typedef struct {
  int *labels;       /* cluster index for every sample, length n_samples */
  double *centroids; /* k * n_features, row-major */
  double inertia;    /* WCSS: sum of squared distances to assigned centroid */
  int n_iter;        /* iterations actually performed in the best run */
  int converged;     /* 1 if converged, 0 if iteration cap was hit */
} KMeansResult;

/* Default parameter set: k-means++, 10 restarts, 300 iters, tol 1e-6. */
KMeansParams kmeans_default_params(void);

/*
 * Run k-means on a row-major dataset of shape [n_samples, n_features].
 * If params is NULL, defaults are used. Caller must release the returned
 * result with kmeans_free().
 */
KMeansResult kmeans(const double *data, size_t n_samples, size_t n_features,
                    int k, const KMeansParams *params);

/* Release memory owned by a KMeansResult. Safe on zero-initialized structs. */
void kmeans_free(KMeansResult *result);

/* Compute WCSS for an arbitrary labeling and centroid set. */
double kmeans_inertia(const double *data, size_t n_samples, size_t n_features,
                      const int *labels, const double *centroids, int k);

#endif
