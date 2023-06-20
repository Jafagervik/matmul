//! File for defining matrix with helpers
use std::{
    fmt::{Debug, Display},
    iter::Sum,
};

use num_traits::{Bounded, NumAssign, NumOps, NumRef};
use rand::distributions::uniform::SampleUniform;

pub trait MatrixType:
    Debug
    + Display
    + Default
    + NumOps
    + NumRef
    + NumAssign
    + Copy
    + Clone
    + Send
    + Sync
    + SampleUniform
    + PartialEq
    + PartialOrd
    + Bounded
    + Sum
{
}

// pub type Matrix<T: MatrixType, const M: usize, const N: usize> = [T; M * N];
pub type Matrix<const M: usize, const N: usize> = [i32; M * N];

impl MatrixType for i8 {}
impl MatrixType for i16 {}
impl MatrixType for i32 {}
impl MatrixType for i64 {}
impl MatrixType for i128 {}
impl MatrixType for f32 {}
impl MatrixType for f64 {}
//
// #[derive(Clone, Debug, PartialEq, PartialOrd)]
// pub struct Matrix<T>
// where
//     T: MatrixType,
// {
//     data: *mut T,
//     nrows: usize,
//     ncols: usize,
// }
//
// impl<T> Matrix<T>
// where
//     T: MatrixType,
// {
//     pub fn new(data: Arc<[T]>, rows: usize, cols: usize) -> Self {
//         Self {
//             data,
//             nrows: rows,
//             ncols: cols,
//         }
//     }
//
//     pub fn as_slice(&self) -> &[T] {
//         unsafe { std::slice::from_raw_parts(self.data, self.nrows * self.ncols) }
//     }
//
//     pub fn transpose(&self) -> Self {
//         let mut data: [T; M * N] = [T::zero(); N * M];
//
//         for i in 0..N {
//             for j in 0..N {
//                 data[i * N + j] = self.at(j, i);
//             }
//         }
//
//         let data = Arc::new(data);
//
//         Self::<T>::new(data)
//     }
//
//     pub fn transpose_inplace(&mut self) {
//         for i in 0..self.nrows {
//             for j in (i + 1)..ncols {
//                 self.data.swap(i * N + j, j * N + i);
//             }
//         }
//         std::mem::swap(&mut self.nrows, &mut self.ncols);
//     }
//
//     pub fn gen_rand_matrix<const M: usize, const N: usize>(start: T, stop: T) -> Self {
//         let mut rng = rand::thread_rng();
//
//         let data = [(); M * N].map(|_| rng.gen_range(start..=stop)).collect();
//         let data = Arc::new(data);
//
//         Self::new(data, M, N)
//     }
//
//     pub fn print(&self) {
//         for i in 0..self.nrows {
//             for j in 0..self.ncols {
//                 print!("{} ", &self.data[i * self.ncols + j]);
//             }
//             println!();
//         }
//         println!();
//     }
//
//     /// Gets a value at a speciic postion
//     pub fn at(&self, i: usize, j: usize) -> T {
//         self.data[i * self.ncols + j]
//     }
//
//     /// BAD
//     pub fn naive_matmul(&mut self, a: &Self, b: &Self) {
//         for i in 0..a.nrows {
//             for j in 0..b.ncols {
//                 for k in 0..a.ncols {
//                     self.data[i * b.nrows + j] += a.at(i, k) * b.at(k, j);
//                 }
//             }
//         }
//     }
//
//     /// Not tooo bad, using iterators and a transposed b matrix
//     pub fn naive_transpose_matmul<const P: usize>(&mut self, a: &Matrix<T>, b: &Matrix<T>) {
//         for i in 0..a.nrows {
//             for j in 0..b.ncols {
//                 self.data[i * b.nrows + j] = (0..N)
//                     .into_iter()
//                     .map(|k| a.data[i * N + k] * b.data[j * N + k])
//                     .sum();
//             }
//         }
//     }
// }
