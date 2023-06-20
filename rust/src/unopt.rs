use crate::MatrixType;

pub fn naive_tranpose<T: MatrixType, const M: usize, const N: usize, const P: usize>(
    c: &mut [T; M * N],
    a: &[T; M * P],
    b: &[T; N * P],
) {
    for i in 0..M {
        for j in 0..N {
            c[i * P + j] = (0..P).map(|k| a[i * N + k] * b[k * N + j]).sum();
        }
    }
}
