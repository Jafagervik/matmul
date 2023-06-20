#![feature(test)]
use rayon::prelude::*;

fn get_target_arch() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        Some("x86_64".to_owned())
    }

    #[cfg(target_arch = "x86")]
    {
        Some("x86".to_owned())
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Target architecture: AArch64");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        None
    }
}

fn get_simd_feature() -> Option<String> {
    unimplemented!()
}

fn main() {
    if let Some(arch) = get_target_arch() {
        match arch.as_str() {
            "x86_64" => println!("Target architecture: x86_64"),
            "x86" => println!("Target architecture: x86"),
            "aarch64" => println!("Target architecture: AArch64"),
            _ => println!("Unknown target architecture: {}", arch),
        }
    } else {
        println!("Target architecture information not available");
    }
    use std::arch::x86_64::*;
    unsafe {
        let a = _mm_set_epi32(1, 2, 3, 4);
        let b = _mm_set_epi32(2, 3, 4, 5);

        let result = _mm_mullo_epi32(a, b);

        let mut result_arr: [i32; 4] = [0; 4];
        _mm_storeu_si128(result_arr.as_mut_ptr() as *mut __m128i, result);
        let result_arr: [i32; 4] = std::mem::transmute(result);

        println!("{:?}", result_arr);

        let sum: i32 = result_arr.par_iter().sum();
        println!("{:?}", sum);
    }
}
