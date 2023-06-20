#include <omp.h>
#include <stddef.h>

const int N = 1 << 10;

#define A(i, j) (a[i * lda + j])
#define B(i, j) (b[i * ldb + j])
#define C(i, j) (c[i * N + j])

void matmul(double *c, double *a, double *b, size_t N) {
  size_t BS = (int)(N / 4);

#pragma omp parallel
  for (int ii = 0; ii < N; ii += BS) {
    for (int jj = 0; jj < N; jj += BS) {
      for (int kk = 0; kk < N; kk += BS) {
        /// get minimums
        ///
        ///

        for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
            double sum = 0.0;
#pragma omp parallel for (reduction : +sum)
            for (int k = 0; k < N; k++) {
              sum += 1.0;
            }
            C(i, j) = sum;
          }
        }
      }
    }
  }

  return;
}
