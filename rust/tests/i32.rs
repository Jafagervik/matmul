use rust::matmul::x86::avx::i32;
use rust::{helpers::*, unopt};

type D = i32;

#[test]
fn squared_4() {
    const N: usize = 4;
    let mut a = [0i32; N * N];
    let mut b = [0i32; N * N];

    randomize_range::<D, N, N>(&mut a, 1, 2);
    randomize_range::<D, N, N>(&mut b, 1, 2);

    let b_transposed = transpose::<D, N, N>(&b);

    let mut c = [0; N * N];

    unsafe {
        i32::mm_4::<N, N, N>(&mut c, &a, &b_transposed);
    };

    let mut c_unopt = [0; N * N];

    unopt::naive_tranpose::<D, N, N, N>(&mut c_unopt, &a, &b);

    assert_eq!(check_i32::<N, N>(&c, &c_unopt), true);
}

#[test]
fn squared_8() {
    const N: usize = 8;
    let mut a = [0i32; N * N];
    let mut b = [0i32; N * N];

    randomize_range::<D, N, N>(&mut a, 1, 2);
    randomize_range::<D, N, N>(&mut b, 1, 2);

    let b_transposed = transpose::<D, N, N>(&b);

    let mut c = [0i32; N * N];

    unsafe {
        i32::mm_8::<N, N, N>(&mut c, &a, &b_transposed);
    };
    let mut c_unopt = [0; N * N];

    unopt::naive_tranpose::<D, N, N, N>(&mut c_unopt, &a, &b);

    assert_eq!(check_i32::<N, N>(&c, &c_unopt), true);
}
