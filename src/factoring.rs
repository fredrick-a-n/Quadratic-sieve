use num::{Integer, One, ToPrimitive, Zero};
use num_bigint::BigUint;
use std::{cmp, sync::atomic};

pub fn factoring(number: [u8; 24]) {
    let now = std::time::Instant::now();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let num: BigUint = BigUint::parse_bytes(&number, 10).unwrap();

    if let Some((f1, f2)) = find_factors(num.clone()) {
        log::info!(
            "Two factors of {num} are {f1} and {f2}",
            num = num,
            f1 = f1,
            f2 = f2
        );
    } else {
        log::error!("No factors found for {num}", num = num);
    }

    println!("Took {}ms", now.elapsed().as_millis());
}

fn find_factors(n: BigUint) -> Option<(BigUint, BigUint)> {
    // Increase b if needed
    let b = cmp::min(
        10000,
        cmp::max(2 * n.to_f64().unwrap().sqrt().floor() as usize, 50),
    );
    log::info!("Let's find some factors of N={n}");
    log::info!("Got bound B = {b}");
    // Find all prime numbers up to B
    let primes: Vec<BigUint> = gen_primes(b)
        .into_iter()
        // Filter all primes that aren't quadratic residue of n, makes the program a lot faster in general but makes it break for some small numbers.
        .filter(|p| legendre(&n, &BigUint::from(*p)) == 1)
        .map(BigUint::from)
        .collect();
    log::debug!("Filtered prime list is {primes:?}");
    log::debug!("Found {p_len} primes", p_len = primes.len());

    // Find enough B-smooth numbers
    let r_vec = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

    let tcount = 8;
    let prog: atomic::AtomicUsize = Default::default();
    let max = primes.len() + 1;
    log::info!("Generating B-smooth numbers, please wait");
    std::thread::scope(|t| {
        for i in 0..tcount {
            let mut j = 1 + i as u32;
            let prog = &prog;
            let n = n.clone();
            let s = n.sqrt();
            let p = &primes;
            let x = r_vec.clone();
            t.spawn(move || {
                while prog.load(atomic::Ordering::SeqCst) < max {
                    // r = sqrt(N) + j
                    // r^2 = N + 2*j*sqrt(N) + j^2
                    // r^2 mod N = 2*j*sqrt(N) + j^2 (for small j)
                    // works for j < sqrt(2N) - sqrt(N) = (sqrt(2) - 1)sqrt(N) = 0.414*sqrt(N)

                    // let r: BigUint = s.clone() * BigUint::from((j as f64).sqrt().floor() as u32) + BigUint::from(j);
                    // let r2 = r.pow(2) % &n;
                    // 26034ms vs 317643ms
                    let r = &s + j;
                    let r2: BigUint = r.pow(2) - n.clone();
                    if let Some(r2_vec) = check_smoothness(r2.clone(), p) {
                        x.lock().unwrap().push((r, r2_vec));
                        prog.fetch_add(1, atomic::Ordering::SeqCst);
                    }

                    j += tcount;
                }
            });
        }
    });
    let r_vec = r_vec.lock().unwrap().clone();
    log::debug!("Found B-smooth numbers: {:?}", r_vec);

    let flat: Vec<i8> = r_vec
        .iter()
        .flat_map(|x| x.1.clone())
        .map(|x| (x % 2) as i8)
        .collect();
    let matrix = ndarray::Array2::from_shape_vec((r_vec.len(), primes.len()), flat).unwrap();
    for solution_vec in find_solutions(matrix) {
        log::debug!("Trying solutions: {:?}", solution_vec);
        let a = r_vec
            .iter()
            .zip(solution_vec.iter())
            .filter(|(_, solution)| **solution == 1)
            .map(|((r, _), _)| r.clone())
            .product::<BigUint>();
        let b = r_vec
            .iter()
            .zip(solution_vec.iter())
            .map(|((_, p_vec), solution)| {
                p_vec
                    .iter()
                    .map(|x| *solution as u32 * (*x) as u32)
                    .collect::<Vec<u32>>()
            })
            .reduce(|x: Vec<u32>, y: Vec<u32>| x.into_iter().zip(y).map(|(a, b)| a + b).collect())
            .unwrap()
            .iter()
            .zip(primes.iter())
            .map(|(c, p)| p.pow(c / 2))
            .product::<BigUint>();

        // a^2 - b^2 mod n = 0
        let diff: BigUint = if a > b { &a - &b } else { &b - &a };

        let f1 = diff.gcd(&n);
        let f2 = &n / f1.clone();

        log::info!("Found some factors: {f1}, {f2} => n = {f1}*{f2} = {n}");
        if f1 > One::one() && f1 < n {
            return Some((f1, f2));
        } else {
            log::error!("Those factors are irrelevant, trying next solution");
        }
    }
    return None;
}

// Checks if B-smooth, returns Some<Vec> of exponants if so, None otherwise
fn check_smoothness(n: BigUint, p: &Vec<BigUint>) -> Option<Vec<u8>> {
    let mut r = vec![0_u8; p.len()];
    let mut n = n.clone();

    for (i, p) in p.iter().enumerate() {
        let mut j = 0;
        loop {
            let (d, r) = n.div_rem(p);
            if r == BigUint::zero() {
                n = d;
            } else {
                break;
            }
            j += 1;
        }
        r[i] = j;
    }
    if n == BigUint::one() {
        return Some(r);
    } else {
        return None;
    }
}

fn gen_primes(n: usize) -> Vec<u32> {
    let mut primes: Vec<u32> = Vec::new();
    let mut sieve = vec![true; n as usize];
    let mut i = 2;
    while i * i <= n {
        if sieve[i as usize] {
            let mut j = i * i;
            while j < n {
                sieve[j as usize] = false;
                j += i;
            }
        }
        i += 1;
    }
    for i in 2..n {
        if sieve[i as usize] {
            primes.push(i as u32);
        }
    }
    return primes;
}

// Check if quadratic residue
fn legendre(n: &BigUint, p: &BigUint) -> u32 {
    if *p == 2u8.into() {
        return 1;
    }
    let (f, r) = (p - 1u8).div_rem(&BigUint::from(2u8));
    assert!(r == Zero::zero());
    return n.modpow(&f, p).to_u32().unwrap();
}

fn find_solutions(mut m: ndarray::Array2<i8>) -> Vec<Vec<i8>> {
    let mut right = ndarray::Array2::eye(m.nrows());

    recursive_gauss(m.view_mut(), right.view_mut());

    log::debug!("Gaussian reduction finished\n {:?}", m);
    log::debug!("Gaussian reduction finished \n {:?}", right);

    return m
        .axis_iter(ndarray::Axis(0))
        .zip(right.axis_iter(ndarray::Axis(0)))
        .filter(|(l, _)| l.sum() == 0)
        .map(|(_, r)| r.to_vec())
        .collect::<Vec<Vec<i8>>>();
}

fn recursive_gauss(mut left: ndarray::ArrayViewMut2<i8>, mut right: ndarray::ArrayViewMut2<i8>) {
    if left.is_empty() {
        return;
    }

    log::debug!("Gaussian reduction in progress\n {:?}", left);
    log::debug!("Gaussian reduction in progress \n {:?}", right);
    log::debug!("Gaussian reduction on matrix of size {:?}", left.shape());

    let pos = left
        .axis_iter_mut(ndarray::Axis(0))
        .enumerate()
        .find(|(_, row)| row[0] == 1);

    if let Some((pos, _)) = pos {
        let mut l_col_it = left.axis_iter_mut(ndarray::Axis(0));
        let mut r_col_it = right.axis_iter_mut(ndarray::Axis(0));
        if pos != 0 {
            ndarray::Zip::from(l_col_it.nth(0).unwrap())
                .and(l_col_it.nth(pos - 1).unwrap())
                .for_each(std::mem::swap);
            ndarray::Zip::from(r_col_it.nth(0).unwrap())
                .and(r_col_it.nth(pos - 1).unwrap())
                .for_each(std::mem::swap);
            l_col_it = left.axis_iter_mut(ndarray::Axis(0));
            r_col_it = right.axis_iter_mut(ndarray::Axis(0));
        }
        let l_piv = l_col_it.next().unwrap();
        let r_piv = r_col_it.next().unwrap();
        for (mut l, mut r) in l_col_it.zip(r_col_it) {
            if l[0] == 1 {
                l -= &l_piv;
                l.mapv_inplace(|x| i8::abs(x % 2));
                r -= &r_piv;
                r.mapv_inplace(|x| i8::abs(x % 2));
            }
        }

        let (_, left) = left.split_at(ndarray::Axis(0), 1);
        let (_, right) = right.split_at(ndarray::Axis(0), 1);
        let (_, left) = left.split_at(ndarray::Axis(1), 1);
        recursive_gauss(left, right)
    } else {
        let (_, left) = left.split_at(ndarray::Axis(1), 1);
        recursive_gauss(left, right)
    }
}
