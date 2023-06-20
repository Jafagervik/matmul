//! Contains all X86 AVX matmuls for i32s

// ====================================================================================
//
//                       NxN matmul NxN
//
// ====================================================================================
use rayon::prelude::*;
use std::mem;

type D = i32;

use crate::{Matrix, MatrixType};

pub unsafe fn mm_4<const M: usize, const P: usize, const N: usize>(
    c: &mut [D; M * N],
    a: &[D; M * P],
    b: &[D; N * P],
) where
    [(); N * N]:,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_mullo_epi32, _mm_set_epi32};

    for i in 0..N {
        for j in 0..N {
            // Load a row of as
            let a_load = _mm_set_epi32(a[i * N + 0], a[i * N + 1], a[i * N + 2], a[i * N + 3]);
            // load a row of b_transpose, or a col of b
            let b_transpose_load =
                _mm_set_epi32(b[j * N + 0], b[j * N + 1], b[j * N + 2], b[j * N + 3]);

            // Multiply these two __m128i registers
            let c_load = _mm_mullo_epi32(a_load, b_transpose_load);

            // Now store these value into an array and sum it up
            let result: [D; 4] = mem::transmute(c_load);
            c[i * N + j] = result.iter().sum();
        }
    }
}

// Square 8 by 8
pub unsafe fn mm_8<const M: usize, const P: usize, const N: usize>(
    c: &mut [D; M * N],
    a: &[D; M * P],
    b: &[D; N * P],
) where
    [(); N * N]:,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_mullo_epi32, _mm256_set_epi32};

    const N: usize = 8;

    for i in 0..N {
        for j in 0..N {
            // Load a row of as
            let a_load = _mm256_set_epi32(
                a[i * N + 0],
                a[i * N + 1],
                a[i * N + 2],
                a[i * N + 3],
                a[i * N + 4],
                a[i * N + 5],
                a[i * N + 6],
                a[i * N + 7],
            );
            // load a row of b_transpose, or a col of b
            let b_transpose_load = _mm256_set_epi32(
                b[j * N + 0],
                b[j * N + 1],
                b[j * N + 2],
                b[j * N + 3],
                b[j * N + 4],
                b[j * N + 5],
                b[j * N + 6],
                b[j * N + 7],
            );

            // Multiply these two __m128i registers
            let c_load = _mm256_mullo_epi32(a_load, b_transpose_load);

            // Now store these value into an array and sum it up
            let result: [D; N] = mem::transmute(c_load);
            c[i * N + j] = result.iter().sum();
        }
    }
}

// ====================================================================================
//
//                       MxN matmul NxK
//
// ====================================================================================

unsafe fn mm_48x84<const M: usize, const P: usize, const N: usize>() {
    unimplemented!()
}
