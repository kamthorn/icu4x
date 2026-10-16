// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

use icu_provider::prelude::*;
use potential_utf::PotentialUtf8;
use zerovec::{ZeroMap, ZeroVec};

// We do this instead of const generics because ZeroFrom and Yokeable derives, as well as serde
// don't support them
macro_rules! lstm_matrix {
    ($name:ident, $generic:literal) => {
        /// The struct that stores a LSTM's matrix.
        ///
        /// <div class="stab unstable">
        /// 🚧 This code is considered unstable; it may change at any time, in breaking or non-breaking ways,
        /// including in SemVer minor releases. While the serde representation of data structs is guaranteed
        /// to be stable, their Rust representation might not be. Use with caution.
        /// </div>
        #[derive(PartialEq, Debug, Clone, zerofrom::ZeroFrom, yoke::Yokeable)]
        #[cfg_attr(feature = "datagen", derive(serde::Serialize))]
        pub struct $name<'data> {
            // Invariant: dims.product() == data.len()
            #[allow(missing_docs)]
            pub(crate) dims: [u16; $generic],
            #[allow(missing_docs)]
            pub(crate) data: ZeroVec<'data, f32>,
        }

        impl<'data> $name<'data> {
            #[cfg(any(feature = "serde", feature = "datagen"))]
            #[doc = concat!("Creates a [`", stringify!($name), "`] with the given dimensions. Fails if the dimensions don't match the data.")]
            pub fn from_parts(
                dims: [u16; $generic],
                data: ZeroVec<'data, f32>,
            ) -> Result<Self, DataError> {
                if dims.iter().map(|&i| i as usize).product::<usize>() != data.len() {
                    Err(DataError::custom("Dimension mismatch"))
                } else {
                    Ok(Self { dims, data })
                }
            }

            #[doc(hidden)] // databake
            pub const fn from_parts_unchecked(
                dims: [u16; $generic],
                data: ZeroVec<'data, f32>,
            ) -> Self {
                Self { dims, data }
            }
        }

        #[cfg(feature = "serde")]
        impl<'de: 'data, 'data> serde::Deserialize<'de> for $name<'data> {
            fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
            where
                S: serde::de::Deserializer<'de>,
            {
                #[derive(serde::Deserialize)]
                struct Raw<'data> {
                    dims: [u16; $generic],
                    #[serde(borrow)]
                    data: ZeroVec<'data, f32>,
                }

                let raw = Raw::deserialize(deserializer)?;

                use serde::de::Error;
                Self::from_parts(raw.dims, raw.data)
                    .map_err(|_| S::Error::custom("Dimension mismatch"))
            }
        }

        #[cfg(feature = "datagen")]
        impl databake::Bake for $name<'_> {
            fn bake(&self, env: &databake::CrateEnv) -> databake::TokenStream {
                let dims = self.dims.bake(env);
                let data = self.data.bake(env);
                databake::quote! {
                    icu_segmenter::provider::$name::from_parts_unchecked(#dims, #data)
                }
            }
        }

        #[cfg(feature = "datagen")]
        impl databake::BakeSize for $name<'_> {
            fn borrows_size(&self) -> usize {
                self.data.borrows_size()
            }
        }
    };
}

lstm_matrix!(LstmMatrix1, 1);
lstm_matrix!(LstmMatrix2, 2);
lstm_matrix!(LstmMatrix3, 3);

#[derive(PartialEq, Debug, Clone, Copy)]
#[cfg_attr(feature = "datagen", derive(serde::Serialize, databake::Bake))]
#[cfg_attr(feature = "datagen", databake(path = icu_segmenter::provider))]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
/// The type of LSTM model
///
/// <div class="stab unstable">
/// 🚧 This code is considered unstable; it may change at any time, in breaking or non-breaking ways,
/// including in SemVer minor releases. While the serde representation of data structs is guaranteed
/// to be stable, their Rust representation might not be. Use with caution.
/// </div>
pub enum ModelType {
    /// A model working on code points
    Codepoints,
    /// A model working on grapheme clusters
    GraphemeClusters,
}

/// The struct that stores a LSTM model.
///
/// <div class="stab unstable">
/// 🚧 This code is considered unstable; it may change at any time, in breaking or non-breaking ways,
/// including in SemVer minor releases. While the serde representation of data structs is guaranteed
/// to be stable, their Rust representation might not be. Use with caution.
/// </div>
#[derive(PartialEq, Debug, Clone, yoke::Yokeable, zerofrom::ZeroFrom)]
#[cfg_attr(feature = "datagen", derive(serde::Serialize))]
#[yoke(prove_covariance_manually)]
pub struct LstmDataFloat32<'data> {
    /// Type of the model
    pub(crate) model: ModelType,
    /// The grapheme cluster dictionary used to train the model
    pub(crate) dic: ZeroMap<'data, PotentialUtf8, u16>,
    /// The embedding layer. Shape (dic.len + 1, e)
    pub(crate) embedding: LstmMatrix2<'data>,
    /// The forward layer's first matrix. Shape (h, 4, e)
    pub(crate) fw_w: LstmMatrix3<'data>,
    /// The forward layer's second matrix. Shape (h, 4, h)
    pub(crate) fw_u: LstmMatrix3<'data>,
    /// The forward layer's bias. Shape (h, 4)
    pub(crate) fw_b: LstmMatrix2<'data>,
    /// The backward layer's first matrix. Shape (h, 4, e)
    pub(crate) bw_w: LstmMatrix3<'data>,
    /// The backward layer's second matrix. Shape (h, 4, h)
    pub(crate) bw_u: LstmMatrix3<'data>,
    /// The backward layer's bias. Shape (h, 4)
    pub(crate) bw_b: LstmMatrix2<'data>,
    /// The output layer's weights. Shape (2, 4, h)
    pub(crate) time_w: LstmMatrix3<'data>,
    /// The output layer's bias. Shape (4)
    pub(crate) time_b: LstmMatrix1<'data>,
}

impl<'data> LstmDataFloat32<'data> {
    #[doc(hidden)] // databake
    #[expect(clippy::too_many_arguments)] // constructor
    pub const fn from_parts_unchecked(
        model: ModelType,
        dic: ZeroMap<'data, PotentialUtf8, u16>,
        embedding: LstmMatrix2<'data>,
        fw_w: LstmMatrix3<'data>,
        fw_u: LstmMatrix3<'data>,
        fw_b: LstmMatrix2<'data>,
        bw_w: LstmMatrix3<'data>,
        bw_u: LstmMatrix3<'data>,
        bw_b: LstmMatrix2<'data>,
        time_w: LstmMatrix3<'data>,
        time_b: LstmMatrix1<'data>,
    ) -> Self {
        Self {
            model,
            dic,
            embedding,
            fw_w,
            fw_u,
            fw_b,
            bw_w,
            bw_u,
            bw_b,
            time_w,
            time_b,
        }
    }

    #[cfg(any(feature = "serde", feature = "datagen"))]
    /// Creates a [`LstmDataFloat32`] with the given data. Fails if the matrix dimensions are inconsistent.
    #[expect(clippy::too_many_arguments)] // constructor
    pub fn try_from_parts(
        model: ModelType,
        dic: ZeroMap<'data, PotentialUtf8, u16>,
        embedding: LstmMatrix2<'data>,
        fw_w: LstmMatrix3<'data>,
        fw_u: LstmMatrix3<'data>,
        fw_b: LstmMatrix2<'data>,
        bw_w: LstmMatrix3<'data>,
        bw_u: LstmMatrix3<'data>,
        bw_b: LstmMatrix2<'data>,
        time_w: LstmMatrix3<'data>,
        time_b: LstmMatrix1<'data>,
    ) -> Result<Self, DataError> {
        let dic_len = u16::try_from(dic.len())
            .map_err(|_| DataError::custom("Dictionary does not fit in u16"))?;

        let num_classes = embedding.dims[0];
        let embedd_dim = embedding.dims[1];
        let hunits = fw_u.dims[2];
        if num_classes - 1 != dic_len
            || fw_w.dims != [4, hunits, embedd_dim]
            || fw_u.dims != [4, hunits, hunits]
            || fw_b.dims != [4, hunits]
            || bw_w.dims != [4, hunits, embedd_dim]
            || bw_u.dims != [4, hunits, hunits]
            || bw_b.dims != [4, hunits]
            || time_w.dims != [2, 4, hunits]
            || time_b.dims != [4]
        {
            return Err(DataError::custom("LSTM dimension mismatch"));
        }

        #[cfg(debug_assertions)]
        if !dic.iter_copied_values().all(|(_, g)| g < dic_len) {
            return Err(DataError::custom("Invalid cluster id"));
        }

        Ok(Self {
            model,
            dic,
            embedding,
            fw_w,
            fw_u,
            fw_b,
            bw_w,
            bw_u,
            bw_b,
            time_w,
            time_b,
        })
    }
}

#[cfg(feature = "serde")]
impl<'de: 'data, 'data> serde::Deserialize<'de> for LstmDataFloat32<'data> {
    fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
    where
        S: serde::de::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw<'data> {
            model: ModelType,
            #[cfg_attr(feature = "serde", serde(borrow))]
            dic: ZeroMap<'data, PotentialUtf8, u16>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            embedding: LstmMatrix2<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            fw_w: LstmMatrix3<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            fw_u: LstmMatrix3<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            fw_b: LstmMatrix2<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            bw_w: LstmMatrix3<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            bw_u: LstmMatrix3<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            bw_b: LstmMatrix2<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            time_w: LstmMatrix3<'data>,
            #[cfg_attr(feature = "serde", serde(borrow))]
            time_b: LstmMatrix1<'data>,
        }

        let raw = Raw::deserialize(deserializer)?;

        use serde::de::Error;
        Self::try_from_parts(
            raw.model,
            raw.dic,
            raw.embedding,
            raw.fw_w,
            raw.fw_u,
            raw.fw_b,
            raw.bw_w,
            raw.bw_u,
            raw.bw_b,
            raw.time_w,
            raw.time_b,
        )
        .map_err(|_| S::Error::custom("Invalid dimensions"))
    }
}

#[cfg(feature = "datagen")]
impl databake::Bake for LstmDataFloat32<'_> {
    fn bake(&self, env: &databake::CrateEnv) -> databake::TokenStream {
        let model = self.model.bake(env);
        let dic = self.dic.bake(env);
        let embedding = self.embedding.bake(env);
        let fw_w = self.fw_w.bake(env);
        let fw_u = self.fw_u.bake(env);
        let fw_b = self.fw_b.bake(env);
        let bw_w = self.bw_w.bake(env);
        let bw_u = self.bw_u.bake(env);
        let bw_b = self.bw_b.bake(env);
        let time_w = self.time_w.bake(env);
        let time_b = self.time_b.bake(env);
        databake::quote! {
            icu_segmenter::provider::LstmDataFloat32::from_parts_unchecked(
                #model,
                #dic,
                #embedding,
                #fw_w,
                #fw_u,
                #fw_b,
                #bw_w,
                #bw_u,
                #bw_b,
                #time_w,
                #time_b,
            )
        }
    }
}

#[cfg(feature = "datagen")]
impl databake::BakeSize for LstmDataFloat32<'_> {
    fn borrows_size(&self) -> usize {
        self.model.borrows_size()
            + self.dic.borrows_size()
            + self.embedding.borrows_size()
            + self.fw_w.borrows_size()
            + self.fw_u.borrows_size()
            + self.fw_b.borrows_size()
            + self.bw_w.borrows_size()
            + self.bw_u.borrows_size()
            + self.bw_b.borrows_size()
            + self.time_w.borrows_size()
            + self.time_b.borrows_size()
    }
}

/// The data to power the LSTM segmentation model.
///
/// This data enum is extensible: more backends may be added in the future.
/// Old data can be used with newer code but not vice versa.
///
/// Examples of possible future extensions:
///
/// 1. Variant to store data in 16 instead of 32 bits
/// 2. Minor changes to the LSTM model, such as different forward/backward matrix sizes
///
/// <div class="stab unstable">
/// 🚧 This code is considered unstable; it may change at any time, in breaking or non-breaking ways,
/// including in SemVer minor releases. While the serde representation of data structs is guaranteed
/// to be stable, their Rust representation might not be. Use with caution.
/// </div>
#[derive(Debug, PartialEq, Clone, yoke::Yokeable, zerofrom::ZeroFrom)]
#[cfg_attr(feature = "datagen", derive(serde::Serialize, databake::Bake))]
#[cfg_attr(feature = "datagen", databake(path = icu_segmenter::provider))]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[yoke(prove_covariance_manually)]
#[non_exhaustive]
pub enum LstmData<'data> {
    /// The data as matrices of zerovec f32 values.
    Float32(#[cfg_attr(feature = "serde", serde(borrow))] LstmDataFloat32<'data>),
    // new variants should go BELOW existing ones
    // Serde serializes based on variant name and index in the enum
    // https://docs.rs/serde/latest/serde/trait.Serializer.html#tymethod.serialize_unit_variant
}

icu_provider::data_struct!(
    LstmData<'_>,
    #[cfg(feature = "datagen")]
);

/// The struct that stores a LSTM model, with the gate weights laid out so that
/// the `4 * h` gate values of each input are contiguous.
///
/// Gate values are ordered input, forget, cell, output, each with `h` values.
///
/// <div class="stab unstable">
/// 🚧 This code is considered unstable; it may change at any time, in breaking or non-breaking ways,
/// including in SemVer minor releases. While the serde representation of data structs is guaranteed
/// to be stable, their Rust representation might not be. Use with caution.
/// </div>
#[derive(PartialEq, Debug, Clone, yoke::Yokeable, zerofrom::ZeroFrom)]
#[cfg_attr(feature = "datagen", derive(serde::Serialize))]
#[yoke(prove_covariance_manually)]
pub struct LstmDataFloat32V2<'data> {
    /// Type of the model
    pub(crate) model: ModelType,
    /// The grapheme cluster dictionary used to train the model
    pub(crate) dic: ZeroMap<'data, PotentialUtf8, u16>,
    /// The embedding layer. Shape (dic.len + 1, e)
    pub(crate) embedding: LstmMatrix2<'data>,
    /// The forward layer's input weights. Shape (e, 4 * h)
    pub(crate) fw_w: LstmMatrix2<'data>,
    /// The forward layer's recurrent weights. Shape (h, 4 * h)
    pub(crate) fw_u: LstmMatrix2<'data>,
    /// The forward layer's bias. Shape (4 * h)
    pub(crate) fw_b: LstmMatrix1<'data>,
    /// The backward layer's input weights. Shape (e, 4 * h)
    pub(crate) bw_w: LstmMatrix2<'data>,
    /// The backward layer's recurrent weights. Shape (h, 4 * h)
    pub(crate) bw_u: LstmMatrix2<'data>,
    /// The backward layer's bias. Shape (4 * h)
    pub(crate) bw_b: LstmMatrix1<'data>,
    /// The output layer's weights. Shape (2, 4, h)
    pub(crate) time_w: LstmMatrix3<'data>,
    /// The output layer's bias. Shape (4)
    pub(crate) time_b: LstmMatrix1<'data>,
}

impl<'data> LstmDataFloat32V2<'data> {
    #[doc(hidden)] // databake
    #[expect(clippy::too_many_arguments)] // constructor
    pub const fn from_parts_unchecked(
        model: ModelType,
        dic: ZeroMap<'data, PotentialUtf8, u16>,
        embedding: LstmMatrix2<'data>,
        fw_w: LstmMatrix2<'data>,
        fw_u: LstmMatrix2<'data>,
        fw_b: LstmMatrix1<'data>,
        bw_w: LstmMatrix2<'data>,
        bw_u: LstmMatrix2<'data>,
        bw_b: LstmMatrix1<'data>,
        time_w: LstmMatrix3<'data>,
        time_b: LstmMatrix1<'data>,
    ) -> Self {
        Self {
            model,
            dic,
            embedding,
            fw_w,
            fw_u,
            fw_b,
            bw_w,
            bw_u,
            bw_b,
            time_w,
            time_b,
        }
    }

    #[cfg(any(feature = "serde", feature = "datagen"))]
    /// Creates a [`LstmDataFloat32V2`] with the given data. Fails if the matrix dimensions are inconsistent.
    #[expect(clippy::too_many_arguments)] // constructor
    pub fn try_from_parts(
        model: ModelType,
        dic: ZeroMap<'data, PotentialUtf8, u16>,
        embedding: LstmMatrix2<'data>,
        fw_w: LstmMatrix2<'data>,
        fw_u: LstmMatrix2<'data>,
        fw_b: LstmMatrix1<'data>,
        bw_w: LstmMatrix2<'data>,
        bw_u: LstmMatrix2<'data>,
        bw_b: LstmMatrix1<'data>,
        time_w: LstmMatrix3<'data>,
        time_b: LstmMatrix1<'data>,
    ) -> Result<Self, DataError> {
        let dic_len = u16::try_from(dic.len())
            .map_err(|_| DataError::custom("Dictionary does not fit in u16"))?;

        let [num_classes, embedd_dim] = embedding.dims;
        let [hunits, gates] = fw_u.dims;
        if num_classes.checked_sub(1) != Some(dic_len)
            || u32::from(gates) != 4 * u32::from(hunits)
            || fw_w.dims != [embedd_dim, gates]
            || fw_b.dims != [gates]
            || bw_w.dims != [embedd_dim, gates]
            || bw_u.dims != [hunits, gates]
            || bw_b.dims != [gates]
            || time_w.dims != [2, 4, hunits]
            || time_b.dims != [4]
        {
            return Err(DataError::custom("LSTM dimension mismatch"));
        }

        #[cfg(debug_assertions)]
        if !dic.iter_copied_values().all(|(_, g)| g < dic_len) {
            return Err(DataError::custom("Invalid cluster id"));
        }

        Ok(Self {
            model,
            dic,
            embedding,
            fw_w,
            fw_u,
            fw_b,
            bw_w,
            bw_u,
            bw_b,
            time_w,
            time_b,
        })
    }
}

#[cfg(feature = "serde")]
impl<'de: 'data, 'data> serde::Deserialize<'de> for LstmDataFloat32V2<'data> {
    fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
    where
        S: serde::de::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw<'data> {
            model: ModelType,
            #[serde(borrow)]
            dic: ZeroMap<'data, PotentialUtf8, u16>,
            #[serde(borrow)]
            embedding: LstmMatrix2<'data>,
            #[serde(borrow)]
            fw_w: LstmMatrix2<'data>,
            #[serde(borrow)]
            fw_u: LstmMatrix2<'data>,
            #[serde(borrow)]
            fw_b: LstmMatrix1<'data>,
            #[serde(borrow)]
            bw_w: LstmMatrix2<'data>,
            #[serde(borrow)]
            bw_u: LstmMatrix2<'data>,
            #[serde(borrow)]
            bw_b: LstmMatrix1<'data>,
            #[serde(borrow)]
            time_w: LstmMatrix3<'data>,
            #[serde(borrow)]
            time_b: LstmMatrix1<'data>,
        }

        let raw = Raw::deserialize(deserializer)?;

        use serde::de::Error;
        Self::try_from_parts(
            raw.model,
            raw.dic,
            raw.embedding,
            raw.fw_w,
            raw.fw_u,
            raw.fw_b,
            raw.bw_w,
            raw.bw_u,
            raw.bw_b,
            raw.time_w,
            raw.time_b,
        )
        .map_err(|_| S::Error::custom("Invalid dimensions"))
    }
}

#[cfg(feature = "datagen")]
impl databake::Bake for LstmDataFloat32V2<'_> {
    fn bake(&self, env: &databake::CrateEnv) -> databake::TokenStream {
        let model = self.model.bake(env);
        let dic = self.dic.bake(env);
        let embedding = self.embedding.bake(env);
        let fw_w = self.fw_w.bake(env);
        let fw_u = self.fw_u.bake(env);
        let fw_b = self.fw_b.bake(env);
        let bw_w = self.bw_w.bake(env);
        let bw_u = self.bw_u.bake(env);
        let bw_b = self.bw_b.bake(env);
        let time_w = self.time_w.bake(env);
        let time_b = self.time_b.bake(env);
        databake::quote! {
            icu_segmenter::provider::LstmDataFloat32V2::from_parts_unchecked(
                #model,
                #dic,
                #embedding,
                #fw_w,
                #fw_u,
                #fw_b,
                #bw_w,
                #bw_u,
                #bw_b,
                #time_w,
                #time_b,
            )
        }
    }
}

#[cfg(feature = "datagen")]
impl databake::BakeSize for LstmDataFloat32V2<'_> {
    fn borrows_size(&self) -> usize {
        self.model.borrows_size()
            + self.dic.borrows_size()
            + self.embedding.borrows_size()
            + self.fw_w.borrows_size()
            + self.fw_u.borrows_size()
            + self.fw_b.borrows_size()
            + self.bw_w.borrows_size()
            + self.bw_u.borrows_size()
            + self.bw_b.borrows_size()
            + self.time_w.borrows_size()
            + self.time_b.borrows_size()
    }
}

#[cfg(any(feature = "serde", feature = "datagen"))]
impl From<&LstmDataFloat32<'_>> for LstmDataFloat32V2<'static> {
    /// Converts from the layout of [`SegmenterLstmAutoV1`](super::SegmenterLstmAutoV1),
    /// which stores the gate weights as `(4, h, n)`, by transposing them to `(n, 4 * h)`.
    fn from(v1: &LstmDataFloat32<'_>) -> Self {
        fn transpose(m: &LstmMatrix3<'_>) -> LstmMatrix2<'static> {
            let [g, h, n] = m.dims;
            let rows = usize::from(g) * usize::from(h);
            let cols = usize::from(n);
            let data: ZeroVec<'static, f32> = (0..rows * cols)
                .map(|i| m.data.get((i % rows) * cols + i / rows).unwrap_or_default())
                .collect();
            LstmMatrix2::from_parts_unchecked([n, g * h], data)
        }
        fn flatten(m: &LstmMatrix2<'_>) -> LstmMatrix1<'static> {
            let [g, h] = m.dims;
            LstmMatrix1::from_parts_unchecked([g * h], m.data.clone().into_owned())
        }
        Self {
            model: v1.model,
            dic: v1.dic.iter_copied_values().collect(),
            embedding: LstmMatrix2::from_parts_unchecked(
                v1.embedding.dims,
                v1.embedding.data.clone().into_owned(),
            ),
            fw_w: transpose(&v1.fw_w),
            fw_u: transpose(&v1.fw_u),
            fw_b: flatten(&v1.fw_b),
            bw_w: transpose(&v1.bw_w),
            bw_u: transpose(&v1.bw_u),
            bw_b: flatten(&v1.bw_b),
            time_w: LstmMatrix3::from_parts_unchecked(
                v1.time_w.dims,
                v1.time_w.data.clone().into_owned(),
            ),
            time_b: LstmMatrix1::from_parts_unchecked(
                v1.time_b.dims,
                v1.time_b.data.clone().into_owned(),
            ),
        }
    }
}

/// The data to power the LSTM segmentation model, see [`LstmDataFloat32V2`].
///
/// This data enum is extensible: more backends may be added in the future.
/// Old data can be used with newer code but not vice versa.
///
/// <div class="stab unstable">
/// 🚧 This code is considered unstable; it may change at any time, in breaking or non-breaking ways,
/// including in SemVer minor releases. While the serde representation of data structs is guaranteed
/// to be stable, their Rust representation might not be. Use with caution.
/// </div>
#[derive(Debug, PartialEq, Clone, yoke::Yokeable, zerofrom::ZeroFrom)]
#[cfg_attr(feature = "datagen", derive(serde::Serialize, databake::Bake))]
#[cfg_attr(feature = "datagen", databake(path = icu_segmenter::provider))]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[yoke(prove_covariance_manually)]
#[non_exhaustive]
pub enum LstmDataV2<'data> {
    /// The data as matrices of zerovec f32 values.
    Float32(#[cfg_attr(feature = "serde", serde(borrow))] LstmDataFloat32V2<'data>),
    // new variants should go BELOW existing ones
    // Serde serializes based on variant name and index in the enum
    // https://docs.rs/serde/latest/serde/trait.Serializer.html#tymethod.serialize_unit_variant
}

#[cfg(any(feature = "serde", feature = "datagen"))]
impl From<&LstmData<'_>> for LstmDataV2<'static> {
    fn from(v1: &LstmData<'_>) -> Self {
        let LstmData::Float32(v1) = v1;
        Self::Float32(v1.into())
    }
}

icu_provider::data_struct!(
    LstmDataV2<'_>,
    #[cfg(feature = "datagen")]
);

#[cfg(test)]
#[cfg(feature = "serde")]
mod tests {
    use super::*;

    #[test]
    fn v1_to_v2_transposes_gate_weights() {
        let (hunits, embedd_dim) = (2u16, 3u16);
        let gates = 4 * hunits;
        let seq = |len: u16, base: f32| -> ZeroVec<'static, f32> {
            (0..len).map(|i| base + f32::from(i)).collect()
        };
        let dic: ZeroMap<PotentialUtf8, u16> =
            [(PotentialUtf8::from_str("a"), 0u16)].into_iter().collect();
        let m1 = |dims: [u16; 1], base| LstmMatrix1::from_parts(dims, seq(dims[0], base)).unwrap();
        let m2 = |dims: [u16; 2], base| {
            LstmMatrix2::from_parts(dims, seq(dims[0] * dims[1], base)).unwrap()
        };
        let m3 = |dims: [u16; 3], base| {
            LstmMatrix3::from_parts(dims, seq(dims[0] * dims[1] * dims[2], base)).unwrap()
        };
        let v1 = LstmDataFloat32::try_from_parts(
            ModelType::Codepoints,
            dic,
            m2([2, embedd_dim], 0.0),
            m3([4, hunits, embedd_dim], 100.0),
            m3([4, hunits, hunits], 200.0),
            m2([4, hunits], 300.0),
            m3([4, hunits, embedd_dim], 400.0),
            m3([4, hunits, hunits], 500.0),
            m2([4, hunits], 600.0),
            m3([2, 4, hunits], 700.0),
            m1([4], 800.0),
        )
        .unwrap();
        let v2 = LstmDataFloat32V2::from(&v1);

        let check = |old: &LstmMatrix3, new: &LstmMatrix2, n: u16| {
            assert_eq!(new.dims, [n, gates]);
            for r in 0..usize::from(gates) {
                for k in 0..usize::from(n) {
                    assert_eq!(
                        new.data.get(k * usize::from(gates) + r),
                        old.data.get(r * usize::from(n) + k)
                    );
                }
            }
        };
        check(&v1.fw_w, &v2.fw_w, embedd_dim);
        check(&v1.fw_u, &v2.fw_u, hunits);
        check(&v1.bw_w, &v2.bw_w, embedd_dim);
        check(&v1.bw_u, &v2.bw_u, hunits);
        assert_eq!(v2.fw_b.dims, [gates]);
        assert_eq!(v2.fw_b.data, v1.fw_b.data);
        assert_eq!(v2.bw_b.data, v1.bw_b.data);
        assert_eq!(v2.embedding, v1.embedding);
        assert_eq!(v2.time_w, v1.time_w);
        assert_eq!(v2.time_b, v1.time_b);
        assert_eq!(v2.dic, v1.dic);

        let v2 = LstmDataFloat32V2::try_from_parts(
            v2.model,
            v2.dic,
            v2.embedding,
            v2.fw_w,
            v2.fw_u,
            v2.fw_b,
            v2.bw_w,
            v2.bw_u,
            v2.bw_b,
            v2.time_w,
            v2.time_b,
        );
        assert!(v2.is_ok());
    }
}
