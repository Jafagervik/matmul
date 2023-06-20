use rust::matmul::x86::avx::f32;
use rust::{helpers::*, unopt};

type D = f32;

#[test]
fn squared_4() {
    const N: usize = 4;
    let mut a = [0f32; N * N];
    let mut b = [0f32; N * N];

    randomize_range::<D, N, N>(&mut a, 1.0, 5.0);
    randomize_range::<D, N, N>(&mut b, 1.0, 5.0);

    let b_transposed = transpose::<D, N, N>(&b);

    let mut c = [0f32; N * N];

    unsafe {
        f32::mm_4::<N, N, N>(&mut c, &a, &b_transposed);
    };

    let mut c_unopt = [0f32; N * N];

    unopt::naive_tranpose::<D, N, N, N>(&mut c_unopt, &a, &b);

    assert_eq!(check_f32::<N, N>(&c, &c_unopt), true);
}
