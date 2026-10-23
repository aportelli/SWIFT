/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Antonin Portelli (antonin.portelli@ed.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/*
 * Measure the ceilings of the roofline model of a machine: single-precision
 * FMA throughput (scalar and SIMD, one thread and all threads) and memory
 * bandwidth (STREAM triad, one thread and all threads). Results are printed
 * as JSON for analyse_task_perf.py.
 *
 * Build (auto-vectorisation must be off for the scalar measurement; the SIMD
 * one uses explicit vector types):
 *   clang -O2 -ffp-contract=fast -fno-vectorize -fno-slp-vectorize
 *         -mcpu=native -pthread machine_peak.c -o machine_peak
 *   gcc -O2 -ffp-contract=fast -fno-tree-vectorize -march=native
 *       -pthread machine_peak.c -o machine_peak
 * Usage: ./machine_peak <threads>
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef VEC_BYTES
#if defined(__AVX512F__)
#define VEC_BYTES 64
#elif defined(__AVX__)
#define VEC_BYTES 32
#else
#define VEC_BYTES 16
#endif
#endif

typedef float vec __attribute__((vector_size(VEC_BYTES)));
#define LANES (VEC_BYTES / (int)sizeof(float))

/* Independent FMA chains to cover the latency of the FMA pipelines (more
 * than about 24 spills the 32 AArch64 vector registers) */
#ifndef CHAINS
#define CHAINS 24
#endif
#define FMA_ITERS 50000000L

/* STREAM arrays: well beyond the size of the caches */
#define STREAM_N (64L * 1024 * 1024)
#define STREAM_REPS 10

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static volatile float sink;

static void *fma_vector(void *arg) {
  (void)arg;
  vec acc[CHAINS], a, b;
  for (int k = 0; k < LANES; k++) {
    a[k] = 0.999999f;
    b[k] = 1e-7f;
  }
  for (int c = 0; c < CHAINS; c++) acc[c] = a * (float)c;
  for (long i = 0; i < FMA_ITERS; i++)
    for (int c = 0; c < CHAINS; c++) acc[c] = acc[c] * a + b;
  float s = 0.f;
  for (int c = 0; c < CHAINS; c++)
    for (int k = 0; k < LANES; k++) s += acc[c][k];
  sink = s;
  return NULL;
}

static void *fma_scalar(void *arg) {
  (void)arg;
  float acc[CHAINS];
  const float a = 0.999999f, b = 1e-7f;
  for (int c = 0; c < CHAINS; c++) acc[c] = (float)c;
  for (long i = 0; i < FMA_ITERS; i++)
    for (int c = 0; c < CHAINS; c++) acc[c] = acc[c] * a + b;
  float s = 0.f;
  for (int c = 0; c < CHAINS; c++) s += acc[c];
  sink = s;
  return NULL;
}

struct stream_arg {
  float *a, *b, *c;
  long n;
};

static void *stream_triad(void *p) {
  struct stream_arg *s = (struct stream_arg *)p;
  for (int r = 0; r < STREAM_REPS; r++) {
    for (long i = 0; i < s->n; i++) s->a[i] = s->b[i] + 0.5f * s->c[i];
    __asm__ volatile("" ::: "memory");
  }
  return NULL;
}

/* Run fn on nthreads threads and return the wall-clock time */
static double run(void *(*fn)(void *), int nthreads, void *args, size_t size) {
  pthread_t *t = malloc(nthreads * sizeof(pthread_t));
  const double tic = now();
  for (int i = 0; i < nthreads; i++)
    pthread_create(&t[i], NULL, fn, (char *)args + i * size);
  for (int i = 0; i < nthreads; i++) pthread_join(t[i], NULL);
  const double toc = now();
  free(t);
  return toc - tic;
}

static double stream(int nthreads) {
  struct stream_arg *args = malloc(nthreads * sizeof(struct stream_arg));
  const long n = STREAM_N / nthreads;
  for (int i = 0; i < nthreads; i++) {
    args[i].n = n;
    args[i].a = malloc(n * sizeof(float));
    args[i].b = malloc(n * sizeof(float));
    args[i].c = malloc(n * sizeof(float));
  }
  /* First touch from the threads themselves would be better on NUMA
   * machines; here we only need the pages to exist. */
  for (int i = 0; i < nthreads; i++)
    for (long k = 0; k < n; k++) {
      args[i].a[k] = 0.f;
      args[i].b[k] = 1.f;
      args[i].c[k] = 2.f;
    }
  const double t = run(stream_triad, nthreads, args, sizeof(*args));
  for (int i = 0; i < nthreads; i++) {
    free(args[i].a);
    free(args[i].b);
    free(args[i].c);
  }
  free(args);
  /* 2 loads + 1 store per element (no write-allocate counted) */
  return 3.0 * sizeof(float) * n * nthreads * STREAM_REPS / t / 1e9;
}

int main(int argc, char *argv[]) {
  const int nthreads = argc > 1 ? atoi(argv[1]) : 1;
  char dummy[64];
  const double fma_flops = 2.0 * CHAINS * FMA_ITERS;

  const double t_vec1 = run(fma_vector, 1, dummy, 0);
  const double t_sca1 = run(fma_scalar, 1, dummy, 0);
  const double t_vecn = run(fma_vector, nthreads, dummy, 0);
  const double bw1 = stream(1);
  const double bwn = stream(nthreads);

  printf("{\n");
  printf("  \"threads\": %d,\n", nthreads);
  printf("  \"simd_lanes_fp32\": %d,\n", LANES);
  printf("  \"gflops_per_core\": %.2f,\n", fma_flops * LANES / t_vec1 / 1e9);
  printf("  \"gflops_per_core_scalar\": %.2f,\n", fma_flops / t_sca1 / 1e9);
  printf("  \"gflops_node\": %.2f,\n",
         fma_flops * LANES * nthreads / t_vecn / 1e9);
  printf("  \"gbytes_per_s_per_core\": %.2f,\n", bw1);
  printf("  \"gbytes_per_s_node\": %.2f\n", bwn);
  printf("}\n");
  return 0;
}
