use criterion::{criterion_group, criterion_main, Criterion};
use rust::helpers::*;
use rust::matmul::x86::avx::f32;

type D = f32;
const N: usize = 4;
const MIN: f32 = -1_000_000.0;
const MAX: f32 = -1_000_000.0;

// Benchmark for matrix multiplication
fn optim(c: &mut Criterion) {
    let mut a = [0f32; N * N];
    let mut b = [0f32; N * N];

    randomize_range::<D, N, N>(&mut a, MIN, MAX);
    randomize_range::<D, N, N>(&mut b, MIN, MAX);

    let b_transposed = transpose::<D, N, N>(&b);

    let mut z = [0f32; N * N];

    c.bench_function("F32 X86-64 AVX 4x4 Dense MatMul", |b| {
        b.iter(|| unsafe { f32::mm_4::<N, N, N>(&mut z, &a, &b_transposed) })
    });
}

fn naive(c: &mut Criterion) {
    let mut x = [0f32; N * N];
    let mut b = [0f32; N * N];

    randomize_range::<D, N, N>(&mut x, MIN, MAX);
    randomize_range::<D, N, N>(&mut b, MIN, MAX);

    let y = transpose::<D, N, N>(&b);

    let mut z = [0f32; N * N];

    c.bench_function("F32 Unopt 4x4 Dense MatMul", |b| {
        b.iter(|| rust::unopt::naive_tranpose::<D, N, N, N>(&mut z, &x, &y))
    });
}

criterion_group!(benches, optim, naive);
criterion_main!(benches);
