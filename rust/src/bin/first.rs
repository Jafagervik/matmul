use std::time::Instant;

use rust::matmul::x86::avx::i32;
use rust::{helpers::*, unopt};

type D = i32;
const N: usize = 16;
const M: usize = 8;
const K: usize = 8;

fn main() {
    let mut a = [0i32; N * N];
    let mut b = [0i32; N * N];

    randomize_range::<D, N, N>(&mut a, 1, 2);
    randomize_range::<D, N, N>(&mut b, 1, 2);

    let b_transposed = transpose::<D, N, N>(&b);

    let mut c = [0i32; N * N];

    let start = Instant::now();

    unsafe {
        i32::mm_16::<N, N, N>(&mut c, &a, &b_transposed);
    };

    let elapsed = start.elapsed();

    print_matrix::<D, N, N>(&a, 2);
    print_matrix::<D, N, N>(&b_transposed, 2);
    print_matrix::<D, N, N>(&c, 2);

    println!("Optimal: {:.1}", elapsed.as_nanos());
    println!();

    let mut c_unopt = [0i32; N * N];

    let start = Instant::now();
    unopt::naive_tranpose::<D, N, N, N>(&mut c_unopt, &a, &b);

    let elapsed = start.elapsed();

    print_matrix::<D, N, N>(&c_unopt, 2);
    println!("Naive: {:.1}", elapsed.as_nanos());

    assert_eq!(check_i32::<N, N>(&c, &c_unopt), true);
}
