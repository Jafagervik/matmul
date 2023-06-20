// // #![rustversion::attr(nightly, feature(portable_simd))]
// //! Here we do a little bit of trolling
// use rand::prelude::*;
use std::mem;

fn main() {
    unimplemented!();
}
//
// const N: usize = 1 << 2;
// type Matrix = [i32; N * N];
//
// const DEBUG: bool = true;
//
// macro_rules! at {
//     ($matrix:expr, $i:expr, $j:expr, $n:expr) => {
//         $matrix[$i * $n + $j]
//     };
// }
//
// /// B is transposed, matrices are 4x4 and we're using std::simd
// // fn simd_matmul(c: &mut Matrix, a: &Matrix, b: &Matrix) {
// //     use std::simd::i32x4;
// //
// //     for i in 0..N {
// //         for j in 0..N {
// //             // Idea: load all values needed, multiply them and reduce simd register into single
// //             // value
// //             let a_load = i32x4(a[i * N + 0], a[i * N + 1], a[i * N + 2], a[i * N + 3]);
// //             let b_load = i32x4(b[j * N + 0], b[j * N + 1], b[j * N + 2], b[j * N + 3]);
// //
// //             let c_load = a_load * b_load;
// //
// //             let c_val = c_load.to_i32();
// //
// //             c[i * N + j] = c_val;
// //         }
// //     }
// // }
//
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// #[target_feature(enable = "avx2")]
// /// Here we exploit SIMD to speed up N = 4 NNxNN matmul on i32s
// unsafe fn matmul_44x44(c: &mut Matrix, a: &Matrix, b: &Matrix) {
//     #[cfg(target_arch = "x86")]
//     use std::arch::x86::*;
//     #[cfg(target_arch = "x86_64")]
//     use std::arch::x86_64::{_mm_mullo_epi32, _mm_set_epi32};
//
//     for i in 0..N {
//         for j in 0..N {
//             // Load a row of as
//             let a_load = _mm_set_epi32(a[i * N + 0], a[i * N + 1], a[i * N + 2], a[i * N + 3]);
//             // load a row of b_transpose, or a col of b
//             let b_transpose_load =
//                 _mm_set_epi32(b[j * N + 0], b[j * N + 1], b[j * N + 2], b[j * N + 3]);
//
//             // Multiply these two __m128i registers
//             let c_load = _mm_mullo_epi32(a_load, b_transpose_load);
//
//             // Now store these value into an array and sum it up
//             let result: [i32; 4] = mem::transmute(c_load);
//             c[i * N + j] = result.iter().sum();
//         }
//     }
// }
//
// fn main() {
//     let a = gen_rand_matrix(1, 2);
//     let b = gen_rand_matrix(1, 2);
//
//     let b_transposed = transpose(&b);
//
//     let mut c: Matrix = [0; N * N];
//
//     unsafe { matmul_44x44(&mut c, &a, &b_transposed) };
//
//     if DEBUG {
//         print_matrix(&a);
//         println!();
//         println!();
//         print_matrix(&b_transposed);
//         println!();
//         println!();
//         print_matrix(&c);
//     }
// }
//
// fn transpose(mat: &Matrix) -> Matrix {
//     let mut result: Matrix = [0; N * N];
//
//     for i in 0..N {
//         for j in 0..N {
//             result[i * N + j] = mat[j * N + i];
//         }
//     }
//
//     result
// }
//
// fn transpose_inplace(matrix: &mut Matrix) {
//     for i in 0..N {
//         for j in (i + 1)..N {
//             matrix.swap(i * N + j, j * N + i);
//         }
//     }
// }
//
// fn gen_rand_matrix(start: i32, stop: i32) -> Matrix {
//     let mut rng = rand::thread_rng();
//
//     [(); N * N].map(|_| rng.gen_range(start..=stop))
// }
//
// fn print_matrix(vec: &Matrix) {
//     for i in 0..N {
//         for j in 0..N {
//             print!("{} ", vec[i * N + j]);
//         }
//         println!();
//     }
//     println!();
// }
//
// /// BAD
// fn naive_matmul(c: &mut Matrix, a: &Matrix, b: &Matrix) {
//     for i in 0..N {
//         for j in 0..N {
//             for k in 0..N {
//                 c[i * N + j] += a[i * N + k] * b[k * N + j];
//             }
//         }
//     }
// }
//
// /// Not tooo bad, using iterators and a transposed b matrix
// fn naive_transpose_matmul(c: &mut Matrix, a: &Matrix, b: &Matrix) {
//     for i in 0..N {
//         for j in 0..N {
//             c[i * N + j] = (0..N).map(|k| a[i * N + k] * b[j * N + k]).sum();
//         }
//     }
// }
