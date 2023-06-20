//! Contains all X86 AVX matmuls for i32s

// ====================================================================================
//
//                       NxN matmul NxN
//
// ====================================================================================
use rayon::prelude::*;
use std::mem;

type D = i32;

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

// Square 16 by 16
pub unsafe fn mm_16<const M: usize, const P: usize, const N: usize>(
    c: &mut [D; M * N],
    a: &[D; M * P],
    b: &[D; N * P],
) where
    [(); N * N]:,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_mullo_epi32, _mm256_set_epi32, _mm256_storeu_si256,
    };

    const N: usize = 16;

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
            let c_load1: __m256i = _mm256_mullo_epi32(a_load, b_transpose_load);

            // Load second load from a
            let a_load = _mm256_set_epi32(
                a[i * N + 8],
                a[i * N + 9],
                a[i * N + 10],
                a[i * N + 11],
                a[i * N + 12],
                a[i * N + 13],
                a[i * N + 14],
                a[i * N + 15],
            );
            // load second row of b_transpose, or a col of b
            let b_transpose_load = _mm256_set_epi32(
                b[j * N + 8],
                b[j * N + 9],
                b[j * N + 10],
                b[j * N + 11],
                b[j * N + 12],
                b[j * N + 13],
                b[j * N + 14],
                b[j * N + 15],
            );

            // Multiply these two __m128i registers
            let c_load2: __m256i = _mm256_mullo_epi32(a_load, b_transpose_load);

            // Now add two rounds of multiplying
            let res_sum: __m256i = _mm256_add_epi32(c_load1, c_load2);

            let mut result: [i32; N] = [0; N];
            let result_ptr = result.as_mut_ptr() as *mut __m256i;
            _mm256_storeu_si256(result_ptr, res_sum);

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
