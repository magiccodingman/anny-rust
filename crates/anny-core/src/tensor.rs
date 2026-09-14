//! Small row-major tensor container and portable Safetensors I/O.
use crate::{ensure, Error, Result};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    #[default]
    Float,
    Index,
    Bool,
}

/// Owned row-major data. Integer fields are range-checked before use as indices.
/// Float64 preserves upstream Anny's native construction/evaluation precision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    #[serde(default)]
    pub kind: Kind,
}
impl Tensor {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f64>) -> Result<Self> {
        let result = Self {
            shape: shape.into(),
            data,
            kind: Kind::Float,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n = shape.iter().product();
        Self {
            shape,
            data: vec![0.; n],
            kind: Kind::Float,
        }
    }
    pub fn indices(shape: impl Into<Vec<usize>>, data: Vec<usize>) -> Self {
        Self {
            shape: shape.into(),
            data: data.into_iter().map(|x| x as f64).collect(),
            kind: Kind::Index,
        }
    }
    pub fn validate(&self) -> Result<()> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Invalid("tensor shape overflow".into()))?;
        ensure(
            n == self.data.len(),
            format!(
                "shape {:?} needs {n} entries, got {}",
                self.shape,
                self.data.len()
            ),
        )?;
        ensure(
            self.data.iter().all(|x| x.is_finite()),
            "non-finite tensor value",
        )?;
        if self.kind == Kind::Index {
            ensure(
                self.data
                    .iter()
                    .all(|x| x.fract() == 0. && x.abs() <= 9_007_199_254_740_991.),
                "integer outside exact supported range",
            )?;
        }
        if self.kind == Kind::Bool {
            ensure(
                self.data.iter().all(|x| *x == 0. || *x == 1.),
                "invalid boolean",
            )?;
        }
        Ok(())
    }
    /// Check that this tensor has exactly `shape`, requiring an O(1) structural check only.
    ///
    /// This is deliberately *not* a full [`Tensor::validate`]: validation includes a finiteness scan
    /// over every element, and `expect_shape` sits on the evaluation hot path for tensors that can be
    /// hundreds of megabytes (the default model's `blendshapes` is 205 MB). Re-scanning that on every
    /// call cost ~9 ms, which was 93% of the fixed per-call cost of `forward`, and it re-derives a
    /// property the tensor already had when it was built: finiteness is enforced once, at
    /// construction (`Tensor::new`, `from_nested`, `from_bytes`, `checked_indices`). Debug builds
    /// still run the full scan so a tensor whose public `data` was mutated into a non-finite state is
    /// caught by the test suite.
    pub fn expect_shape(&self, shape: &[usize], name: &str) -> Result<()> {
        #[cfg(debug_assertions)]
        self.validate()?;
        self.checked_shape(shape, name)
    }
    /// The O(1) half of [`Tensor::expect_shape`]: rank/product/entry-count consistency plus the
    /// expected shape.
    pub fn checked_shape(&self, shape: &[usize], name: &str) -> Result<()> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Invalid("tensor shape overflow".into()))?;
        ensure(
            n == self.data.len(),
            format!(
                "shape {:?} needs {n} entries, got {}",
                self.shape,
                self.data.len()
            ),
        )?;
        ensure(
            self.shape == shape,
            format!("{name}: expected {shape:?}, got {:?}", self.shape),
        )
    }
    pub fn checked_indices(&self, bound: usize, name: &str) -> Result<Vec<usize>> {
        self.validate()?;
        // The failure message is built only when the check fails. `ensure(cond, format!(..))` builds
        // its message eagerly, and this loop runs once per element over mesh-size arrays: ~82k
        // allocations per call on the default model, which was 7.9 ms of an 8.5 ms `derive measure`.
        let mut out = Vec::with_capacity(self.data.len());
        for &x in &self.data {
            if !(x >= 0. && x.fract() == 0. && x < bound as f64) {
                return Err(Error::Invalid(format!(
                    "{name}: index {x} outside [0,{bound})"
                )));
            }
            out.push(x as usize);
        }
        Ok(out)
    }
    pub fn select(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        ensure(
            axis < self.shape.len(),
            "selection axis outside tensor rank",
        )?;
        ensure(
            indices.iter().all(|&i| i < self.shape[axis]),
            "selection index outside tensor",
        )?;
        let inner: usize = self.shape[axis + 1..].iter().product();
        let outer: usize = self.shape[..axis].iter().product();
        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        let mut data = Vec::with_capacity(outer * indices.len() * inner);
        for a in 0..outer {
            for &i in indices {
                let start = (a * self.shape[axis] + i) * inner;
                data.extend_from_slice(&self.data[start..start + inner]);
            }
        }
        Ok(Self {
            shape,
            data,
            kind: self.kind,
        })
    }
    pub fn nested_json(&self) -> Value {
        fn build(shape: &[usize], data: &[f64], kind: Kind) -> Value {
            if shape.is_empty() {
                return if kind == Kind::Bool {
                    Value::Bool(data[0] != 0.)
                } else if kind == Kind::Index {
                    Value::from(data[0] as i64)
                } else {
                    Value::from(data[0])
                };
            }
            let step: usize = shape[1..].iter().product();
            Value::Array(
                (0..shape[0])
                    .map(|i| build(&shape[1..], &data[i * step..(i + 1) * step], kind))
                    .collect(),
            )
        }
        build(&self.shape, &self.data, self.kind)
    }
    pub fn from_nested(value: &Value) -> Result<Self> {
        fn dimensions(v: &Value) -> Vec<usize> {
            if let Some(a) = v.as_array() {
                let mut s = vec![a.len()];
                if let Some(first) = a.first() {
                    s.extend(dimensions(first));
                }
                s
            } else {
                vec![]
            }
        }
        fn visit(v: &Value, shape: &[usize], data: &mut Vec<f64>) -> Result<()> {
            if shape.is_empty() {
                data.push(
                    v.as_f64()
                        .ok_or_else(|| Error::Invalid("tensor entries must be numbers".into()))?,
                );
            } else {
                let a = v
                    .as_array()
                    .ok_or_else(|| Error::Invalid("mixed tensor ranks".into()))?;
                ensure(a.len() == shape[0], "ragged tensor input")?;
                for x in a {
                    visit(x, &shape[1..], data)?;
                }
            }
            Ok(())
        }
        let shape = dimensions(value);
        let mut data = vec![];
        visit(value, &shape, &mut data)?;
        Self::new(shape, data)
    }
}

/// Also handles the one-time converter's nested state-dictionary envelope.
#[derive(Clone, Debug, Default)]
pub struct Archive {
    pub tensors: BTreeMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}
impl Archive {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_, info) = SafeTensors::read_metadata(bytes)?;
        let metadata = info.metadata().clone().unwrap_or_default();
        let s = SafeTensors::deserialize(bytes)?;
        let mut tensors = BTreeMap::new();
        for (name, v) in s.iter() {
            let (kind, data) = decode(v.dtype(), v.data())?;
            let t = Tensor {
                shape: v.shape().to_vec(),
                data,
                kind,
            };
            t.validate()?;
            tensors.insert(name.into(), t);
        }
        Ok(Self { tensors, metadata })
    }
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.serialize(false)
    }
    /// Serialize floating attributes as f32; integers retain their integer dtype.
    /// This affects storage only. Use AnnyF32 for single-precision evaluation.
    pub fn to_bytes_f32(&self) -> Result<Vec<u8>> {
        self.serialize(true)
    }
    pub fn save_f32(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_bytes_f32()?)?;
        Ok(())
    }
    fn serialize(&self, single: bool) -> Result<Vec<u8>> {
        let mut buffers = Vec::new();
        for (name, t) in &self.tensors {
            t.validate()?;
            let bytes: Vec<u8> = match t.kind {
                Kind::Float if single => {
                    ensure(
                        t.data.iter().all(|&x| (x as f32).is_finite()),
                        "output overflows f32",
                    )?;
                    t.data
                        .iter()
                        .flat_map(|&x| (x as f32).to_le_bytes())
                        .collect()
                }
                Kind::Float => t.data.iter().flat_map(|x| x.to_le_bytes()).collect(),
                Kind::Index => t
                    .data
                    .iter()
                    .flat_map(|x| (*x as i64).to_le_bytes())
                    .collect(),
                Kind::Bool => t.data.iter().map(|x| u8::from(*x != 0.)).collect(),
            };
            buffers.push((name, t, bytes));
        }
        let views = buffers
            .iter()
            .map(|(n, t, b)| {
                let dtype = match t.kind {
                    Kind::Float if single => Dtype::F32,
                    Kind::Float => Dtype::F64,
                    Kind::Index => Dtype::I64,
                    Kind::Bool => Dtype::BOOL,
                };
                Ok(((*n).clone(), TensorView::new(dtype, t.shape.clone(), b)?))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(safetensors::serialize(views, &Some(self.metadata.clone()))?)
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_bytes()?)?;
        Ok(())
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))
    }
    pub fn payload(&self) -> Result<Value> {
        Ok(serde_json::from_str(
            self.metadata
                .get("anny_port_payload")
                .map(String::as_str)
                .unwrap_or("{}"),
        )?)
    }
    pub fn payload_tensor(&self, path: &str) -> Result<&Tensor> {
        let mut value = self.payload()?;
        for key in path.split('/') {
            value = value
                .get(key)
                .ok_or_else(|| Error::Invalid(format!("missing payload entry {path}")))?
                .clone();
        }
        if let Some(name) = value.get("__tensor__").and_then(Value::as_str) {
            self.tensor(name)
        } else {
            Err(Error::Invalid(format!("{path} is not a tensor")))
        }
    }
}
fn decode(dtype: Dtype, bytes: &[u8]) -> Result<(Kind, Vec<f64>)> {
    macro_rules! convert {
        ($ty:ty,$size:literal) => {
            bytes
                .chunks_exact($size)
                .map(|x| <$ty>::from_le_bytes(x.try_into().unwrap()) as f64)
                .collect()
        };
    }
    let integer = matches!(
        dtype,
        Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
    );
    let kind = if integer {
        Kind::Index
    } else if dtype == Dtype::BOOL {
        Kind::Bool
    } else {
        Kind::Float
    };
    let data = match dtype {
        Dtype::F64 => convert!(f64, 8),
        Dtype::F32 => convert!(f32, 4),
        Dtype::I64 => convert!(i64, 8),
        Dtype::U64 => convert!(u64, 8),
        Dtype::I32 => convert!(i32, 4),
        Dtype::U32 => convert!(u32, 4),
        Dtype::I16 => convert!(i16, 2),
        Dtype::U16 => convert!(u16, 2),
        Dtype::I8 => bytes.iter().map(|x| (*x as i8) as f64).collect(),
        Dtype::U8 | Dtype::BOOL => bytes.iter().map(|x| *x as f64).collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|x| {
                f32::from_bits((u16::from_le_bytes(x.try_into().unwrap()) as u32) << 16) as f64
            })
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|x| {
                let h = u16::from_le_bytes(x.try_into().unwrap());
                let sign = if h & 0x8000 == 0 { 1. } else { -1. };
                let e = ((h >> 10) & 31) as i32;
                let f = (h & 1023) as f64;
                sign * if e == 0 {
                    f * 2f64.powi(-24)
                } else if e == 31 {
                    if f == 0. {
                        f64::INFINITY
                    } else {
                        f64::NAN
                    }
                } else {
                    (1. + f / 1024.) * 2f64.powi(e - 15)
                }
            })
            .collect(),
        _ => {
            return Err(Error::Invalid(format!(
                "unsupported tensor dtype {dtype:?}"
            )))
        }
    };
    Ok((kind, data))
}
