use std::mem;

type D = f32;

pub unsafe fn mm_4<const M: usize, const P: usize, const N: usize>(
    c: &mut [D; M * N],
    a: &[D; M * P],
    b: &[D; N * P],
) where
    [(); M * P]:,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_mul_ps, _mm_set_ps};

    for i in 0..N {
        for j in 0..N {
            // Load a row of as
            let a_load = _mm_set_ps(a[i * N + 0], a[i * N + 1], a[i * N + 2], a[i * N + 3]);
            // load a row of b_transpose, or a col of b
            let b_transpose_load =
                _mm_set_ps(b[j * N + 0], b[j * N + 1], b[j * N + 2], b[j * N + 3]);

            // Multiply these two __m128i registers
            let c_load = _mm_mul_ps(a_load, b_transpose_load);

            // Now store these value into an array and sum it up
            let result: [D; 4] = mem::transmute(c_load);
            c[i * N + j] = result.iter().sum();
        }
    }
}

pub unsafe fn mm_BIG<const M: usize, const P: usize, const N: usize>(
    c: &mut [D; M * N],
    a: &[D; M * P],
    b: &[D; N * P],
) where
    [(); M * P]:,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_mul_ps, _mm_set_ps};

    for i in 0..N {
        for j in 0..N {
            // Load a row of as
            let a_load = _mm_set_ps(a[i * N + 0], a[i * N + 1], a[i * N + 2], a[i * N + 3]);
            // load a row of b_transpose, or a col of b
            let b_transpose_load =
                _mm_set_ps(b[j * N + 0], b[j * N + 1], b[j * N + 2], b[j * N + 3]);

            // Multiply these two __m128i registers
            let c_load = _mm_mul_ps(a_load, b_transpose_load);

            // Now store these value into an array and sum it up
            let result: [D; 4] = mem::transmute(c_load);
            c[i * N + j] = result.iter().sum();
        }
    }
}
