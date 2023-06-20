use std::time::Instant;

use crate::{Matrix, MatrixType};
use num_traits::Bounded;
use rand::prelude::*;

// Times the operation for a transposed matmul
pub fn matmul_timer<T, const M: usize, const P: usize, const N: usize>(
    C: &mut Matrix<M, N>,
    A: &Matrix<M, N>,
    B: &Matrix<P, N>,
) where
    T: MatrixType,
    [T; M * N]:,
    [T; P * N]:,
{
    let start = Instant::now();

    //C.naive_matmul(A, B);

    let duration = start.elapsed();

    println!("Elapsed time: {:.5}ms", duration.as_millis());
}

pub fn randomize_range<T: MatrixType, const M: usize, const N: usize>(
    mat: &mut [T; M * N],
    start: T,
    stop: T,
) where
    [T; M * N]:,
{
    let mut rng = rand::thread_rng();

    for element in mat.iter_mut() {
        *element = rng.gen_range(start..=stop);
    }
}

pub fn randomize_all<T: MatrixType, const M: usize, const N: usize>() -> Matrix<M, N>
where
    [i32; M * N]:,
{
    let mut rng = rand::thread_rng();

    let minv: i32 = Bounded::min_value();
    let maxv: i32 = Bounded::max_value();

    [0; M * N].map(|_| rng.gen_range(minv..=maxv))
}

pub fn print_matrix<T: MatrixType, const M: usize, const N: usize>(
    mat: &[T; M * N],
    decimals: usize,
) where
    [T; M * N]:,
{
    for i in 0..N {
        for j in 0..N {
            print!("{:.decimals$} ", mat[i * N + j]);
        }
        println!();
    }
    println!();
}

pub fn transpose<T: MatrixType, const M: usize, const N: usize>(matrix: &[T; M * N]) -> [T; N * M]
where
    T: MatrixType,
    [T; M * N]:,
    [T; N * M]:,
{
    let mut data: [T; N * M] = [T::zero(); N * M];

    for i in 0..N {
        for j in 0..N {
            data[i * N + j] = matrix[j * N + i];
        }
    }

    data
}

pub fn check_f32<const M: usize, const N: usize>(lhs: &[f32; M * N], rhs: &[f32; M * N]) -> bool {
    let eps: f32 = 1e-5;

    for i in 0..M {
        for j in 0..N {
            if (lhs[i * N + j] - rhs[i * N + j]).abs() >= eps {
                return false;
            }
        }
    }

    true
}

pub fn check_i32<const M: usize, const N: usize>(lhs: &[i32; M * N], rhs: &[i32; M * N]) -> bool {
    for i in 0..M {
        for j in 0..N {
            if lhs[i * N + j] != rhs[i * N + j] {
                return false;
            }
        }
    }

    true
}
