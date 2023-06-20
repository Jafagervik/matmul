#include "mpi.h"
#include <stdlib.h>
/* macro for column major indexing */
#define A(i, j) (a[j * lda + i])
#define B(i, j) (b[j * ldb + i])
#define C(i, j) (c[j * ldc + i])
#define min(x, y) ((x) < (y) ? (x) : (y))
int i_one = 1;                    /* used for constant passed to blas call */
double d_one = 1.0, d_zero = 0.0; /* used for constant passed to blas call */

void RING_Bcast(double *buf, int count, MPI_Datatype type, int root,
                MPI_Comm comm) {
  int me, np;
  MPI_Status status;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &np);
  if (me != root)
    MPI_Recv(buf, count, type, (me - 1 + np) % np, MPI_ANY_TAG, comm);
  if ((me + 1) % np != root)
    MPI_Send(buf, count, type, (me + 1) % np, 0, comm);
}

void RING_SUM(double *buf, int count, MPI_Datatype type, int root,
              MPI_Comm comm, double *work) {
  int me, np;
  MPI_Status status;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &np);
  if (me != (root + 1) % np) {
    MPI_Recv(work, count, type, (me - 1 + np) % np, MPI_ANY_TAG, comm, &status);
    daxpy_(&count, &d_one, work, &i_one, buf, &i_one);
  }
  if (me != root)
    MPI_Send(buf, count, type, (me + 1) % np, 0, comm);
}

void pdgemm(m, n, k, nb, alpha, a, lda, b, ldb, beta, c, ldc, m_a, n_a, m_b,
            n_b, m_c, n_c, comm_row, comm_col, work1, work2) int m,
    n, k,           /* global matrix dimensions */
    nb,             /* panel width */
    m_a[], n_a[],   /* dimensions of blocks of A */
    m_b[], n_b[],   /* dimensions of blocks of A */
    m_c[], n_c[],   /* dimensions of blocks of A */
    lda, ldb, ldc;  /* leading dimension of local arrays that
     hold local portions of matrices A, B, C */
double *a, *b, *c,  /* arrays that hold local parts of A, B, C */
    alpha, beta,    /* multiplication constants */
    *work1, *work2; /* work arrays */
MPI_Comm comm_row,  /* Communicator for this row of nodes */
    comm_col;       /* Communicator for this column of nodes */
{
  int myrow, mycol,     /* my row and column index */
      nprow, npcol,     /* number of node rows and columns */
      i, j, kk, iwrk,   /* misc. index variables */
      icurrow, icurcol, /* index of row and column that hold current
      row and column, resp., for rank-1 update*/
      ii, jj;           /* local index (on icurrow and icurcol, resp.)
                of row and column for rank-1 update */
  double *temp;         /* temporary pointer used in pdgemm_abt */
  double *p;
  /* get myrow, mycol */
  MPI_Comm_rank(comm_row, &mycol);
  MPI_Comm_rank(comm_col, &myrow);
  /* scale local block of C */
  for (j = 0; j < n_c[mycol]; j++)
    for (i = 0; i < m_c[myrow]; i++)
      C(i, j) = beta * C(i, j);

  icurrow = 0;
  icurcol = 0;
  ii = jj = 0;
  /* malloc temp space for summation */
  temp = (double *)malloc(m_c[myrow] * nb * sizeof(double));
  for (kk = 0; kk < k; kk += iwrk) {
    iwrk = min(nb, m_b[icurrow] - ii);
    iwrk = min(iwrk, n_a[icurcol] - jj);
    /* pack current iwrk columns of A into work1 */
    if (mycol == icurcol)
      dlacpy_("General", &m_a[myrow], &iwrk, &A(0, jj), &lda, work1,
              &m_a[myrow]);
    /* pack current iwrk rows of B into work2 */
    if (myrow == icurrow)
      dlacpy_("General", &iwrk, &n_b[mycol], &B(ii, 0), &ldb, work2, &iwrk);
    /* broadcast work1 and work2 */
    RING_Bcast(work1, m_a[myrow] * iwrk, MPI_DOUBLE, icurcol, comm_row);
    RING_Bcast(work2, n_b[mycol] * iwrk, MPI_DOUBLE, icurrow, comm_col);
    /* update local block */
    dgemm_("No transpose", "No transpose", &m_c[myrow], &n_c[mycol], &iwrk,
           &alpha, work1, &m_b[myrow], work2, &iwrk, &d_one, c, &ldc);
    /* update icurcol, icurrow, ii, jj */
    ii += iwrk;
    jj += iwrk;
    if (jj >= n_a[icurcol]) {
      icurcol++;
      jj = 0;
    };
    if (ii >= m_b[icurrow]) {
      icurrow++;
      ii = 0;
    };
  }
  free(temp);
}

icurrow = 0;
icurcol = 0;
ii = jj = 0;
/* malloc temp space for summation */
temp = (double *)malloc(m_c[myrow] * nb * sizeof(double));
/* loop over all column panels of C */
for (kk = 0; kk < k; kk += iwrk) {
  iwrk = min(nb, m_b[icurrow] - ii);
  iwrk = min(iwrk, n_c[icurcol] - jj);
  /* pack current iwrk rows of B into work2 */
  if (myrow == icurrow)
    dlacpy_("General", &iwrk, &n_b[mycol], &B(ii, 0), &ldb, work2, &iwrk);
  /* broadcast work2 */
  RING_Bcast(work2, n_b[mycol] * iwrk, MPI_DOUBLE, icurrow, comm_col);
  /* Multiply local block of A times incoming rows of B */
  dgemm_("No transpose", "Transpose", &m_c[myrow], &iwrk, &n_a[mycol], &alpha,
         a, &lda, work2, &iwrk, &d_zero, work1, &m_c[myrow]);
  /* Sum to node that holds current columns of C */
  RING_SUM(work1, m_c[myrow] * iwrk, MPI_DOUBLE, icurcol, comm_row, temp);
  /* Add to current columns of C */
  if (mycol == icurcol) {
    p = work1;
    for (j = jj; j < jj + iwrk; j++) {
      daxpy_(&m_c[myrow], &d_one, p, &i_one, &C(0, j), &i_one);
      p += m_c[myrow];
    }
  }
  /* update icurcol, icurrow, ii, jj */
  ii += iwrk;
  jj += iwrk;
  if (jj >= n_c[icurcol]) {
    icurcol++;
    jj = 0;
  };
  if (ii >= m_b[icurrow]) {
    icurrow++;
    ii = 0;
  };
}
free(temp);
}
