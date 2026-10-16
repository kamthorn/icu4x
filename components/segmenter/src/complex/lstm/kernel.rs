// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

//! The LSTM recurrence, with the weights rearranged so that every inner loop
//! runs over contiguous `f32` slices of length `4 * hunits` and can be
//! vectorized by the compiler.
//!
//! The data stores `W` as `(4, hunits, embedd_dim)` and `U` as `(4, hunits, hunits)`,
//! so `W·x` and `U·h` are many short dot products. Here both are transposed once per
//! segmentation call, which turns them into `axpy` operations over the `4 * hunits`
//! gate values. `W·x` only depends on the input id, so it is computed once for every
//! distinct id of the input instead of once per position.

use super::matrix::MatrixZero;
use alloc::vec;
use alloc::vec::Vec;
#[allow(unused_imports)]
use core_maths::*;
use zerovec::ZeroSlice;
use zerovec::ule::AsULE;

/// Gate values are processed in blocks of this many lanes, which stay in registers
/// while the block accumulates `U·h`.
const LANES: usize = 16;

/// The weights of one direction of the LSTM, prepared for one input.
///
/// Rows of `xw` and `ut` hold the `4 * hunits` gate values padded with zeros to
/// `stride`, a multiple of [`LANES`].
#[derive(Debug)]
pub(super) struct Direction {
    hunits: usize,
    stride: usize,
    /// `b + W·x` for each distinct input id (in the order of `ids`).
    xw: Vec<f32>,
    /// `U` transposed: row `k` holds the gate weights that multiply `h[k]`.
    ut: Vec<f32>,
}

/// Transposes a row-major `rows × cols` matrix into `cols` rows of `stride >= rows` values.
fn transpose(m: &ZeroSlice<f32>, rows: usize, cols: usize, stride: usize) -> Vec<f32> {
    let mut out = vec![0.0; stride * cols];
    if cols == 0 {
        return out;
    }
    for (r, row) in m.as_ule_slice().chunks_exact(cols).take(rows).enumerate() {
        for (c, v) in row.iter().enumerate() {
            if let Some(dest) = out.get_mut(c * stride + r) {
                *dest = f32::from_unaligned(*v);
            }
        }
    }
    out
}

impl Direction {
    /// `ids` are the sorted, distinct embedding ids of the input.
    pub(super) fn new(
        embedding: MatrixZero<'_, 2>,
        w: MatrixZero<'_, 3>,
        u: MatrixZero<'_, 3>,
        b: MatrixZero<'_, 2>,
        ids: &[u16],
    ) -> Self {
        let (_, hunits, embedd_dim) = w.dim();
        let gates = 4 * hunits;
        let stride = gates.div_ceil(LANES) * LANES;
        let wt = transpose(w.as_slice(), gates, embedd_dim, stride);
        let ut = transpose(u.as_slice(), gates, hunits, stride);
        let mut bias: Vec<f32> = b.as_slice().iter().collect();
        bias.resize(stride, 0.0);

        let mut xw = vec![0.0; ids.len() * stride];
        let mut x = Vec::with_capacity(embedd_dim);
        for (&id, out) in ids.iter().zip(xw.chunks_exact_mut(stride)) {
            let Some(e) = embedding.submatrix::<1>(id as usize) else {
                debug_assert!(false, "id is at most dict.len()");
                continue;
            };
            x.clear();
            x.extend(e.as_slice().iter());
            add_mul_transposed(&bias, &x, &wt, out);
        }

        Self {
            hunits,
            stride,
            xw,
            ut,
        }
    }

    /// The length of the scratch space that [`Self::step`] needs.
    pub(super) fn scratch_len(&self) -> usize {
        self.stride
    }

    /// One LSTM step for the input id at index `slot` of `ids`. Updates `h` and `c`;
    /// `s` is scratch space of length [`Self::scratch_len`].
    pub(super) fn step(&self, slot: usize, h: &mut [f32], c: &mut [f32], s: &mut [f32]) {
        let hunits = self.hunits;
        let stride = self.stride;
        let Some(xw) = self.xw.get(slot * stride..(slot + 1) * stride) else {
            debug_assert!(false, "slot is an index of ids");
            return;
        };
        // s = xw + U·h
        add_mul_transposed(xw, h, &self.ut, s);

        // Gates in the order input, forget, cell, output. Each activation runs over a
        // contiguous slice so that the loop can be vectorized.
        let Some((ifg, o)) = s.get_mut(..4 * hunits).map(|s| s.split_at_mut(3 * hunits)) else {
            debug_assert!(false, "s has 4 * hunits values");
            return;
        };
        let (i_f, g) = ifg.split_at_mut(2 * hunits);
        i_f.iter_mut().for_each(|x| *x = sigmoid(*x));
        g.iter_mut().for_each(|x| *x = tanh(*x));
        o.iter_mut().for_each(|x| *x = sigmoid(*x));
        let (i, f) = i_f.split_at(hunits);
        for ((((c, i), f), g), o) in c.iter_mut().zip(i).zip(f).zip(&*g).zip(o.iter_mut()) {
            *c = i * g + *c * f;
            *o *= tanh(*c);
        }
        h.copy_from_slice(o);
    }
}

/// `out = init + Mᵀ·x`, where `mt` holds `x.len()` rows of `out.len()` values.
///
/// `out` is computed one block of [`LANES`] values at a time, so the block stays in
/// registers while the rows of `mt` are accumulated into it. All lengths are multiples
/// of [`LANES`] except `x.len()`.
#[inline]
fn add_mul_transposed(init: &[f32], x: &[f32], mt: &[f32], out: &mut [f32]) {
    let stride = out.len();
    for (bi, (out_block, init_block)) in out
        .as_chunks_mut::<LANES>()
        .0
        .iter_mut()
        .zip(init.as_chunks::<LANES>().0)
        .enumerate()
    {
        let mut acc = *init_block;
        for (xk, row) in x.iter().zip(mt.chunks_exact(stride)) {
            if let Some(m) = row.as_chunks::<LANES>().0.get(bi) {
                for (a, m) in acc.iter_mut().zip(m) {
                    *a += xk * m;
                }
            }
        }
        *out_block = acc;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline]
fn tanh(x: f32) -> f32 {
    // tanh(x) = 1 - 2 / (e^2x + 1); the absolute error near 0 is about 1e-7,
    // which is below what the model can resolve.
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

/// `e^x` without branches or table lookups, so that loops calling it can be vectorized.
///
/// Splits `x = n ln 2 + r` with `|r| <= ln 2 / 2` and evaluates `e^r` with a degree-6
/// polynomial (Cephes `expf`), relative error about 2e-7. The input is clamped so that
/// `2^n` stays a normal `f32`.
#[inline]
fn exp(x: f32) -> f32 {
    const LOG2_E: f32 = core::f32::consts::LOG2_E;
    // ln 2 split into a high part with few mantissa bits and a correction
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 88.0);
    // Round to nearest by adding and subtracting 1.5 * 2^23 (|x * LOG2_E| < 2^22);
    // `f32::round` is a libm call in no_std builds.
    const ROUND: f32 = 12_582_912.0;
    let t = x * LOG2_E + ROUND;
    let n = t - ROUND;
    let r = x - n * LN2_HI - n * LN2_LO;
    let p = 1.987_569_1e-4_f32;
    let p = p * r + 1.398_199_9e-3;
    let p = p * r + 8.333_452e-3;
    let p = p * r + 4.166_579_6e-2;
    let p = p * r + 1.666_666_5e-1;
    let p = p * r + 0.5;
    let er = p * r * r + r + 1.0;
    // 2^n, built from the exponent bits; n is in [-126, 127] after the clamp. The low
    // mantissa bits of `t` hold n, which avoids a float-to-int cast that would stop
    // the loop from being vectorized.
    let n_bits = t.to_bits().wrapping_sub(ROUND.to_bits());
    let scale = f32::from_bits(n_bits.wrapping_add(127) << 23);
    er * scale
}
