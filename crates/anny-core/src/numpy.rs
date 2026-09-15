//! Bounded, non-executing NPY/NPZ input for motion data. Supports numeric arrays
//! and fixed-width text metadata, NOT object arrays or arbitrary pickle values.
//! Format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
use crate::{ensure, tensor::Kind, Error, Result, Tensor};
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
};

#[derive(Clone, Debug)]
pub enum NpyValue {
    Numeric(Tensor),
    Text {
        shape: Vec<usize>,
        values: Vec<String>,
    },
}
impl NpyValue {
    pub fn numeric(&self) -> Result<&Tensor> {
        match self {
            Self::Numeric(t) => Ok(t),
            _ => Err(Error::Invalid("expected numeric NPY array".into())),
        }
    }
    pub fn scalar_text(&self) -> Result<&str> {
        match self {
            Self::Text { values, .. } if values.len() == 1 => Ok(&values[0]),
            _ => Err(Error::Invalid("expected one text value".into())),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_entries: usize,
    pub max_uncompressed_bytes: usize,
    pub max_numeric_values: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_entries: 128,
            max_uncompressed_bytes: 256 * 1024 * 1024,
            max_numeric_values: 32 * 1024 * 1024,
        }
    }
}
fn err(s: &str) -> Error {
    Error::Invalid(s.into())
}
struct Header<'a> {
    s: &'a [u8],
    p: usize,
}
impl Header<'_> {
    fn ws(&mut self) {
        while self.p < self.s.len() && self.s[self.p].is_ascii_whitespace() {
            self.p += 1;
        }
    }
    fn ch(&mut self, c: u8) -> Result<()> {
        self.ws();
        ensure(
            self.s.get(self.p) == Some(&c),
            "invalid NPY header punctuation",
        )?;
        self.p += 1;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        self.ws();
        let quote = *self
            .s
            .get(self.p)
            .ok_or_else(|| err("truncated NPY header"))?;
        ensure(
            quote == b'\'' || quote == b'"',
            "NPY descriptor must be a string; object/structured dtypes are unsupported",
        )?;
        self.p += 1;
        let mut v = Vec::new();
        loop {
            let c = *self
                .s
                .get(self.p)
                .ok_or_else(|| err("unterminated NPY string"))?;
            self.p += 1;
            if c == quote {
                break;
            }
            ensure(
                c != b'\\' && c.is_ascii(),
                "escaped/non-ASCII NPY header strings unsupported",
            )?;
            v.push(c);
        }
        Ok(String::from_utf8(v).unwrap())
    }
    fn word(&mut self) -> Result<&[u8]> {
        self.ws();
        let start = self.p;
        while self.p < self.s.len() && self.s[self.p].is_ascii_alphabetic() {
            self.p += 1;
        }
        ensure(self.p > start, "invalid NPY boolean")?;
        Ok(&self.s[start..self.p])
    }
    fn shape(&mut self) -> Result<Vec<usize>> {
        self.ch(b'(')?;
        let mut result = Vec::new();
        loop {
            self.ws();
            if self.s.get(self.p) == Some(&b')') {
                self.p += 1;
                return Ok(result);
            }
            let start = self.p;
            while self.p < self.s.len() && self.s[self.p].is_ascii_digit() {
                self.p += 1;
            }
            ensure(
                self.p > start,
                "NPY dimensions must be nonnegative integers",
            )?;
            let n = std::str::from_utf8(&self.s[start..self.p])
                .unwrap()
                .parse::<usize>()
                .map_err(|_| err("NPY dimension overflow"))?;
            result.push(n);
            ensure(result.len() <= 32, "NPY rank limit exceeded")?;
            self.ws();
            if self.s.get(self.p) == Some(&b')') {
                self.p += 1;
                return Ok(result);
            }
            self.ch(b',')?;
        }
    }
    fn parse(&mut self) -> Result<(String, bool, Vec<usize>)> {
        self.ch(b'{')?;
        let (mut dtype, mut fortran, mut shape) = (None, None, None);
        loop {
            self.ws();
            if self.s.get(self.p) == Some(&b'}') {
                self.p += 1;
                break;
            }
            let key = self.string()?;
            self.ch(b':')?;
            match key.as_str() {
                "descr" => {
                    ensure(dtype.is_none(), "duplicate NPY descr")?;
                    dtype = Some(self.string()?);
                }
                "fortran_order" => {
                    ensure(fortran.is_none(), "duplicate NPY order")?;
                    fortran = Some(match self.word()? {
                        b"True" => true,
                        b"False" => false,
                        _ => return Err(err("invalid NPY order")),
                    });
                }
                "shape" => {
                    ensure(shape.is_none(), "duplicate NPY shape")?;
                    shape = Some(self.shape()?);
                }
                _ => return Err(err("unsupported NPY header field")),
            }
            self.ws();
            if self.s.get(self.p) == Some(&b'}') {
                self.p += 1;
                break;
            }
            self.ch(b',')?;
        }
        self.ws();
        ensure(self.p == self.s.len(), "trailing NPY header data")?;
        Ok((
            dtype.ok_or_else(|| err("NPY missing descr"))?,
            fortran.ok_or_else(|| err("NPY missing order"))?,
            shape.ok_or_else(|| err("NPY missing shape"))?,
        ))
    }
}
pub fn read_npy(bytes: &[u8], limits: Limits) -> Result<NpyValue> {
    ensure(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "invalid NPY magic/header",
    )?;
    let (start, len) = match (bytes[6], bytes[7]) {
        (1, 0) => (
            10,
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize,
        ),
        (2, 0) | (3, 0) => {
            ensure(bytes.len() >= 12, "truncated NPY v2/v3 header")?;
            (
                12,
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            )
        }
        _ => return Err(err("unsupported NPY version")),
    };
    ensure(
        len <= 64 * 1024 && len <= bytes.len() - start,
        "NPY header limit or length invalid",
    )?;
    let (dtype, fortran, shape) = Header {
        s: &bytes[start..start + len],
        p: 0,
    }
    .parse()?;
    let count = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| err("NPY shape overflow"))?;
    ensure(
        count <= limits.max_numeric_values,
        "NPY value limit exceeded",
    )?;
    let descriptor = dtype.as_bytes();
    ensure(descriptor.len() >= 3, "invalid NPY dtype")?;
    let little = match descriptor[0] {
        b'<' => true,
        b'>' => false,
        b'=' | b'|' => cfg!(target_endian = "little"),
        _ => return Err(err("unsupported NPY byte order")),
    };
    let width = dtype[2..]
        .parse::<usize>()
        .map_err(|_| err("invalid NPY item width"))?;
    let kind = descriptor[1];
    ensure(
        kind != b'O',
        "NPY object arrays/pickle execution are forbidden",
    )?;
    let width = if kind == b'U' {
        width
            .checked_mul(4)
            .ok_or_else(|| err("NPY width overflow"))?
    } else {
        width
    };
    ensure(width > 0, "zero-width NPY types unsupported")?;
    let size = count
        .checked_mul(width)
        .ok_or_else(|| err("NPY array size overflow"))?;
    ensure(
        size <= limits.max_uncompressed_bytes && size == bytes.len() - start - len,
        "NPY payload length/limit mismatch",
    )?;
    let raw = &bytes[start + len..];
    let reorder = |i: usize| -> usize {
        if !fortran || shape.len() < 2 {
            return i;
        }
        let (mut remaining, mut offset) = (i, 0);
        for axis in (0..shape.len()).rev() {
            let coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            let stride: usize = shape[..axis].iter().product();
            offset += coordinate * stride;
        }
        offset
    };
    if kind == b'S' || kind == b'U' {
        ensure(
            count <= 4096 && width <= 65536,
            "NPY text metadata limit exceeded",
        )?;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let off = reorder(i) * width;
            let text = &raw[off..off + width];
            if kind == b'S' {
                let end = text.iter().position(|&x| x == 0).unwrap_or(text.len());
                values.push(
                    std::str::from_utf8(&text[..end])
                        .map_err(|_| err("NPY text is not UTF-8"))?
                        .to_string(),
                );
            } else {
                let mut value = String::new();
                for c in text.chunks_exact(4) {
                    let c = if little {
                        u32::from_le_bytes(c.try_into().unwrap())
                    } else {
                        u32::from_be_bytes(c.try_into().unwrap())
                    };
                    if c == 0 {
                        break;
                    }
                    value.push(char::from_u32(c).ok_or_else(|| err("invalid NPY Unicode scalar"))?);
                }
                values.push(value);
            }
        }
        return Ok(NpyValue::Text { shape, values });
    }
    ensure(
        matches!(
            (kind, width),
            (b'f', 4 | 8) | (b'i' | b'u', 1 | 2 | 4 | 8) | (b'b' | b'?', 1)
        ),
        "unsupported numeric NPY dtype",
    )?;
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let off = reorder(i) * width;
        let mut item = raw[off..off + width].to_vec();
        if !little {
            item.reverse();
        }
        macro_rules! num {
            ($ty:ty) => {
                <$ty>::from_le_bytes(item.as_slice().try_into().unwrap()) as f64
            };
        }
        let x = match (kind, width) {
            (b'f', 4) => num!(f32),
            (b'f', 8) => num!(f64),
            (b'i', 1) => (item[0] as i8) as f64,
            (b'u' | b'b' | b'?', 1) => item[0] as f64,
            (b'i', 2) => num!(i16),
            (b'u', 2) => num!(u16),
            (b'i', 4) => num!(i32),
            (b'u', 4) => num!(u32),
            (b'i', 8) => num!(i64),
            (b'u', 8) => num!(u64),
            _ => unreachable!(),
        };
        data.push(x);
    }
    let t = Tensor {
        shape,
        data,
        kind: match kind {
            b'f' => Kind::Float,
            b'b' | b'?' => Kind::Bool,
            _ => Kind::Index,
        },
    };
    t.validate()?;
    Ok(NpyValue::Numeric(t))
}
pub fn read_npz(bytes: &[u8], limits: Limits) -> Result<BTreeMap<String, NpyValue>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| Error::Invalid(format!("invalid NPZ: {e}")))?;
    ensure(zip.len() <= limits.max_entries, "NPZ entry limit exceeded")?;
    let (mut total, mut values) = (0usize, 0usize);
    let mut output = BTreeMap::new();
    for i in 0..zip.len() {
        let mut f = zip
            .by_index(i)
            .map_err(|e| Error::Invalid(format!("NPZ entry: {e}")))?;
        let name = f.name().to_string();
        ensure(
            !name.contains(['/', '\\']) && name.ends_with(".npy"),
            "NPZ entries must be root-level .npy arrays",
        )?;
        let key = name.trim_end_matches(".npy").to_string();
        ensure(
            !key.is_empty() && !output.contains_key(&key),
            "duplicate/empty NPZ array name",
        )?;
        let size = usize::try_from(f.size()).map_err(|_| err("NPZ size overflow"))?;
        total = total
            .checked_add(size)
            .ok_or_else(|| err("NPZ size overflow"))?;
        ensure(
            total <= limits.max_uncompressed_bytes,
            "NPZ decompression limit exceeded",
        )?;
        let mut raw = Vec::new();
        f.by_ref().take(size as u64 + 1).read_to_end(&mut raw)?;
        ensure(raw.len() == size, "NPZ declared size mismatch")?;
        let result = read_npy(
            &raw,
            Limits {
                max_numeric_values: limits.max_numeric_values - values,
                ..limits
            },
        )?;
        values += match &result {
            NpyValue::Numeric(t) => t.data.len(),
            NpyValue::Text { values, .. } => values.len(),
        };
        output.insert(key, result);
    }
    Ok(output)
}
