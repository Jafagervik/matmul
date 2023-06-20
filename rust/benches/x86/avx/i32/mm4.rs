use rust::helpers::*;
use rust::matmul::x86::avx::i32;

use criterion::{criterion_group, criterion_main, Criterion};
type D = i32;
const N: usize = 4;

// Benchmark for matrix multiplication
fn optim(c: &mut Criterion) {
    let mut a = [0i32; N * N];
    let mut b = [0i32; N * N];

    randomize_range::<D, N, N>(&mut a, 1, 10);
    randomize_range::<D, N, N>(&mut b, 1, 10);

    let b_transposed = transpose::<D, N, N>(&b);

    let mut z = [0i32; N * N];

    c.bench_function("X86-64 AVX 4x4 Dense MatMul", |b| {
        b.iter(|| unsafe { i32::mm_4::<N, N, N>(&mut z, &a, &b_transposed) })
    });
}

fn naive(c: &mut Criterion) {
    let mut x = [0i32; N * N];
    let mut b = [0i32; N * N];

    randomize_range::<D, N, N>(&mut x, 1, 10);
    randomize_range::<D, N, N>(&mut b, 1, 10);

    let y = transpose::<D, N, N>(&b);

    let mut z = [0i32; N * N];

    c.bench_function("Unopt 4x4 Dense MatMul", |b| {
        b.iter(|| rust::unopt::naive_tranpose::<D, N, N, N>(&mut z, &x, &y))
    });
}

criterion_group!(benches, optim, naive);
criterion_main!(benches);
