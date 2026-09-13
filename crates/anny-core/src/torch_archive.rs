//! Read-only decoder for the tensor/dictionary subset of PyTorch ZIP archives.
//!
//! This is NOT a Python interpreter or a general pickle loader. GLOBAL/REDUCE are
//! matched against explicit tensor-construction tags; arbitrary classes, imports,
//! persistent external references, executable hooks and unsupported opcodes fail.
//! The archive is never extracted to the filesystem. Decoded tensors preserve
//! dtype, storage offsets and strides, and sparse CSR/COO are materialized densely.
use crate::{ensure, tensor::Archive, Error, Result};
use safetensors::{tensor::TensorView, Dtype};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::{Cursor, Read},
    path::Path,
};

type Id = usize;
const MAX_BYTES: usize = 512 * 1024 * 1024;
const MAX_NODES: usize = 1_000_000;
const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug)]
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
impl RawTensor {
    fn indices(&self) -> Result<Vec<usize>> {
        ensure(
            matches!(self.dtype, Dtype::I64 | Dtype::I32),
            "sparse indices must be signed integer tensors",
        )?;
        self.data
            .chunks_exact(self.dtype.size())
            .map(|v| {
                let n = if self.dtype == Dtype::I64 {
                    i64::from_le_bytes(v.try_into().unwrap())
                } else {
                    i32::from_le_bytes(v.try_into().unwrap()) as i64
                };
                usize::try_from(n).map_err(|_| bad("negative or overflowing sparse index"))
            })
            .collect()
    }
}
#[derive(Clone, Debug)]
enum Node {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Seq(Vec<Id>),
    Dict(Vec<(String, Id)>),
    Global(String),
    Storage {
        dtype: Dtype,
        key: String,
        count: usize,
    },
    Tensor(RawTensor),
}
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(format!("PyTorch archive: {}", s.into()))
}
fn product(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .filter(|&n| n <= MAX_BYTES)
        .ok_or_else(|| bad("tensor dimensions exceed decoder limits"))
}
fn storage_dtype(s: &str) -> Option<Dtype> {
    Some(match s {
        "torch.FloatStorage" => Dtype::F32,
        "torch.DoubleStorage" => Dtype::F64,
        "torch.HalfStorage" => Dtype::F16,
        "torch.BFloat16Storage" => Dtype::BF16,
        "torch.LongStorage" => Dtype::I64,
        "torch.IntStorage" => Dtype::I32,
        "torch.ShortStorage" => Dtype::I16,
        "torch.CharStorage" => Dtype::I8,
        "torch.ByteStorage" => Dtype::U8,
        "torch.BoolStorage" => Dtype::BOOL,
        _ => return None,
    })
}
fn allowed_global(s: &str) -> bool {
    storage_dtype(s).is_some()
        || matches!(
            s,
            "collections.OrderedDict"
                | "torch._utils._rebuild_tensor_v2"
                | "torch._utils._rebuild_tensor"
                | "torch._utils._rebuild_sparse_tensor"
                | "torch.serialization._get_layout"
                | "torch.Size"
        )
}

#[derive(Default)]
struct OutputBudget {
    visits: usize,
    bytes: usize,
}

struct Decoder<'a> {
    bytes: &'a [u8],
    pos: usize,
    nodes: Vec<Node>,
    stack: Vec<Option<Id>>,
    memo: HashMap<usize, Id>,
    files: BTreeMap<String, Vec<u8>>,
    prefix: String,
    big_endian: bool,
    decoded_bytes: usize,
}
impl<'a> Decoder<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| bad("truncated pickle record"))?;
        let v = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(v)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn line(&mut self) -> Result<String> {
        let n = self.bytes[self.pos..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| bad("unterminated GLOBAL"))?;
        let s = std::str::from_utf8(self.take(n)?)
            .map_err(|_| bad("invalid UTF-8"))?
            .to_owned();
        self.take(1)?;
        Ok(s)
    }
    fn push(&mut self, n: Node) -> Result<()> {
        ensure(
            self.nodes.len() < MAX_NODES && self.stack.len() < MAX_NODES,
            "archive node limit exceeded",
        )?;
        let id = self.nodes.len();
        self.nodes.push(n);
        self.stack.push(Some(id));
        Ok(())
    }
    fn pop(&mut self) -> Result<Id> {
        self.stack
            .pop()
            .flatten()
            .ok_or_else(|| bad("invalid pickle stack"))
    }
    fn top(&self) -> Result<Id> {
        self.stack
            .last()
            .copied()
            .flatten()
            .ok_or_else(|| bad("empty pickle stack"))
    }
    fn marked(&mut self) -> Result<Vec<Id>> {
        let i = self
            .stack
            .iter()
            .rposition(Option::is_none)
            .ok_or_else(|| bad("missing MARK"))?;
        let items = self
            .stack
            .drain(i + 1..)
            .map(|x| x.ok_or_else(|| bad("nested MARK")))
            .collect::<Result<_>>()?;
        self.stack.pop();
        Ok(items)
    }
    fn seq(&self, id: Id) -> Result<&[Id]> {
        match &self.nodes[id] {
            Node::Seq(v) => Ok(v),
            _ => Err(bad("expected tuple/list")),
        }
    }
    fn string(&self, id: Id) -> Result<&str> {
        match &self.nodes[id] {
            Node::String(s) => Ok(s),
            _ => Err(bad("expected string")),
        }
    }
    fn integer(&self, id: Id) -> Result<usize> {
        match self.nodes[id] {
            Node::Int(x) => usize::try_from(x).map_err(|_| bad("negative/overflowing dimension")),
            _ => Err(bad("expected integer")),
        }
    }
    fn dims(&self, id: Id) -> Result<Vec<usize>> {
        let v = self.seq(id)?;
        ensure(v.len() <= MAX_DEPTH, "tensor rank limit exceeded")?;
        v.iter().map(|&i| self.integer(i)).collect()
    }
    fn pairs(&self, values: Vec<Id>) -> Result<Vec<(String, Id)>> {
        ensure(values.len().is_multiple_of(2), "odd dictionary stack")?;
        values
            .chunks_exact(2)
            .map(|p| Ok((self.string(p[0])?.to_owned(), p[1])))
            .collect()
    }
    fn append_dict(&mut self, id: Id, pairs: Vec<(String, Id)>) -> Result<()> {
        let Node::Dict(v) = &mut self.nodes[id] else {
            return Err(bad("SETITEM target is not dictionary"));
        };
        for (k, value) in pairs {
            ensure(
                !v.iter().any(|(old, _)| old == &k),
                format!("duplicate archive dictionary key {k}"),
            )?;
            v.push((k, value));
        }
        Ok(())
    }
    fn remember(&mut self, key: usize) -> Result<()> {
        ensure(
            key < MAX_NODES && self.memo.len() < MAX_NODES,
            "pickle memo limit exceeded",
        )?;
        self.memo.insert(key, self.top()?);
        Ok(())
    }
    fn recall(&mut self, key: usize) -> Result<()> {
        let id = *self
            .memo
            .get(&key)
            .ok_or_else(|| bad("invalid memo reference"))?;
        self.stack.push(Some(id));
        Ok(())
    }
    fn persistent(&self, id: Id) -> Result<Node> {
        let p = self.seq(id)?;
        ensure(
            p.len() == 5 && self.string(p[0])? == "storage",
            "unsupported persistent reference",
        )?;
        let Node::Global(name) = &self.nodes[p[1]] else {
            return Err(bad("storage dtype is not a known tag"));
        };
        let dtype = storage_dtype(name).ok_or_else(|| bad("unsupported storage dtype"))?;
        let key = self.string(p[2])?.to_owned();
        ensure(
            !key.is_empty() && key.chars().all(|c| c.is_ascii_digit()),
            "invalid storage key",
        )?;
        let location = self.string(p[3])?;
        ensure(
            location == "cpu" || location.starts_with("cuda:"),
            "unsupported storage location",
        )?;
        let count = self.integer(p[4])?;
        let n = count
            .checked_mul(dtype.size())
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| bad("storage too large"))?;
        let file = self
            .files
            .get(&format!("{}data/{key}", self.prefix))
            .ok_or_else(|| bad(format!("missing storage {key}")))?;
        ensure(file.len() == n, "storage length disagrees with dtype/count")?;
        Ok(Node::Storage { dtype, key, count })
    }
    fn dense_tensor(&mut self, args: &[Id]) -> Result<Node> {
        ensure(
            args.len() == 4 || args.len() == 6 || args.len() == 7,
            "unexpected tensor constructor arguments",
        )?;
        let Node::Storage { dtype, key, count } = &self.nodes[args[0]] else {
            return Err(bad("tensor has no storage"));
        };
        let (dtype, count, key) = (*dtype, *count, key.clone());
        let offset = self.integer(args[1])?;
        let shape = self.dims(args[2])?;
        let strides = self.dims(args[3])?;
        ensure(shape.len() == strides.len(), "rank/stride mismatch")?;
        if args.len() > 4 {
            ensure(
                matches!(self.nodes[args[4]], Node::Bool(_)),
                "invalid requires_grad flag",
            )?;
            ensure(
                matches!(&self.nodes[args[5]], Node::Dict(d) if d.is_empty()),
                "tensor hooks are forbidden",
            )?;
            if args.len() == 7 {
                ensure(
                    matches!(&self.nodes[args[6]], Node::Null | Node::Dict(_)),
                    "unsupported tensor metadata",
                )?;
            }
        }
        let elements = product(&shape)?;
        let bytes = elements
            .checked_mul(dtype.size())
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| bad("tensor too large"))?;
        self.reserve(bytes)?;
        let storage = &self.files[&format!("{}data/{key}", self.prefix)];
        let mut data = Vec::with_capacity(bytes);
        for flat in 0..elements {
            let mut q = flat;
            let mut index = offset;
            for (&d, &s) in shape.iter().zip(&strides).rev() {
                index = index
                    .checked_add(
                        (q % d)
                            .checked_mul(s)
                            .ok_or_else(|| bad("stride overflow"))?,
                    )
                    .ok_or_else(|| bad("offset overflow"))?;
                q /= d;
            }
            ensure(index < count, "tensor view outside storage")?;
            let at = index * dtype.size();
            if self.big_endian {
                data.extend(storage[at..at + dtype.size()].iter().rev().copied());
            } else {
                data.extend_from_slice(&storage[at..at + dtype.size()]);
            }
        }
        Ok(Node::Tensor(RawTensor { dtype, shape, data }))
    }
    fn reserve(&mut self, n: usize) -> Result<()> {
        self.decoded_bytes = self
            .decoded_bytes
            .checked_add(n)
            .ok_or_else(|| bad("decoded size overflow"))?;
        ensure(
            self.decoded_bytes <= MAX_BYTES,
            "decoded archive exceeds memory limit",
        )
    }
    fn sparse_tensor(&mut self, args: &[Id]) -> Result<Node> {
        ensure(args.len() == 2, "unexpected sparse constructor")?;
        let layout = self.string(args[0])?.to_owned();
        let parts = self.seq(args[1])?.to_vec();
        let (indices, values, shape) = match layout.as_str() {
            "torch.sparse_csr" => {
                ensure(parts.len() == 4 || parts.len() == 5, "invalid CSR tuple")?;
                let (Node::Tensor(rows), Node::Tensor(cols), Node::Tensor(values)) = (
                    &self.nodes[parts[0]],
                    &self.nodes[parts[1]],
                    &self.nodes[parts[2]],
                ) else {
                    return Err(bad("invalid CSR tensors"));
                };
                let shape = self.dims(parts[3])?;
                ensure(
                    shape.len() == 2
                        && rows.shape == [shape[0] + 1]
                        && cols.shape.len() == 1
                        && values.shape == cols.shape,
                    "only 2-D scalar CSR supported",
                )?;
                let rows = rows.indices()?;
                let cols = cols.indices()?;
                ensure(
                    rows[0] == 0
                        && rows.last() == Some(&cols.len())
                        && rows.windows(2).all(|w| w[0] <= w[1])
                        && cols.iter().all(|&c| c < shape[1]),
                    "malformed CSR indices",
                )?;
                let mut indices = Vec::with_capacity(cols.len());
                for r in 0..shape[0] {
                    for &c in &cols[rows[r]..rows[r + 1]] {
                        indices.push(r * shape[1] + c);
                    }
                }
                (indices, values.clone(), shape)
            }
            "torch.sparse_coo" => {
                ensure(parts.len() == 3 || parts.len() == 4, "invalid COO tuple")?;
                let (Node::Tensor(ids), Node::Tensor(values)) =
                    (&self.nodes[parts[0]], &self.nodes[parts[1]])
                else {
                    return Err(bad("invalid COO tensors"));
                };
                let shape = self.dims(parts[2])?;
                let rank = shape.len();
                ensure(
                    ids.shape.len() == 2 && ids.shape[0] == rank && values.shape == [ids.shape[1]],
                    "only scalar COO supported",
                )?;
                let nnz = ids.shape[1];
                let ids = ids.indices()?;
                product(&shape)?;
                let mut indices = vec![0; nnz];
                for k in 0..nnz {
                    for d in 0..rank {
                        let i = ids[d * nnz + k];
                        ensure(i < shape[d], "COO index outside tensor")?;
                        indices[k] = indices[k] * shape[d] + i;
                    }
                }
                (indices, values.clone(), shape)
            }
            _ => return Err(bad(format!("unsupported sparse layout {layout}"))),
        };
        let size = product(&shape)?
            .checked_mul(values.dtype.size())
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| bad("dense sparse tensor too large"))?;
        self.reserve(size)?;
        let mut data = vec![0u8; size];
        let width = values.dtype.size();
        let mut written = HashSet::new();
        for (k, i) in indices.into_iter().enumerate() {
            // Canonical input is coalesced. Reject ambiguous duplicates rather than
            // silently overwriting values and calling that sparse-to-dense parity.
            ensure(
                written.insert(i),
                "duplicate sparse indices: coalesce source tensor first",
            )?;
            data[i * width..(i + 1) * width]
                .copy_from_slice(&values.data[k * width..(k + 1) * width]);
        }
        Ok(Node::Tensor(RawTensor {
            dtype: values.dtype,
            shape,
            data,
        }))
    }
    fn reduce(&mut self, callable: Id, arguments: Id) -> Result<Node> {
        let Node::Global(name) = &self.nodes[callable] else {
            return Err(bad("REDUCE requires allowlisted constructor tag"));
        };
        let name = name.clone();
        let args = self.seq(arguments)?.to_vec();
        match name.as_str() {
            "collections.OrderedDict" => {
                ensure(
                    args.is_empty(),
                    "only empty OrderedDict constructor supported",
                )?;
                Ok(Node::Dict(vec![]))
            }
            "torch.Size" => {
                ensure(args.len() == 1, "invalid Size")?;
                Ok(Node::Seq(self.seq(args[0])?.to_vec()))
            }
            "torch.serialization._get_layout" => {
                ensure(args.len() == 1, "invalid layout")?;
                Ok(Node::String(self.string(args[0])?.to_owned()))
            }
            "torch._utils._rebuild_tensor_v2" | "torch._utils._rebuild_tensor" => {
                self.dense_tensor(&args)
            }
            "torch._utils._rebuild_sparse_tensor" => self.sparse_tensor(&args),
            _ => Err(bad(format!("cannot invoke {name}"))),
        }
    }
    fn parse(&mut self) -> Result<Id> {
        let mut steps = 0usize;
        while self.pos < self.bytes.len() {
            steps += 1;
            ensure(
                steps <= 4_000_000 && self.stack.len() <= MAX_NODES,
                "pickle operation limit exceeded",
            )?;
            let opcode = self.byte()?;
            match opcode {
                0x80 => {
                    let version = self.byte()?;
                    ensure(
                        (2..=5).contains(&version),
                        "only pickle protocols 2-5 are supported",
                    )?;
                }
                0x95 => {
                    let length = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
                    ensure(
                        length <= (self.bytes.len() - self.pos) as u64,
                        "invalid pickle frame",
                    )?;
                }
                b'.' => {
                    let root = self.pop()?;
                    ensure(
                        self.stack.is_empty() && self.pos == self.bytes.len(),
                        "trailing pickle data",
                    )?;
                    return Ok(root);
                }
                b'(' => self.stack.push(None),
                b'N' => self.push(Node::Null)?,
                0x88 => self.push(Node::Bool(true))?,
                0x89 => self.push(Node::Bool(false))?,
                b'K' => {
                    let n = self.byte()? as i64;
                    self.push(Node::Int(n))?;
                }
                b'M' => {
                    let n = u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64;
                    self.push(Node::Int(n))?;
                }
                b'J' => {
                    let n = i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64;
                    self.push(Node::Int(n))?;
                }
                0x8a => {
                    let n = self.byte()? as usize;
                    ensure(n <= 8, "large pickle integer is unsupported")?;
                    let v = self.take(n)?;
                    let mut a = [if v.last().is_some_and(|v| v & 0x80 != 0) {
                        255
                    } else {
                        0
                    }; 8];
                    a[..n].copy_from_slice(v);
                    self.push(Node::Int(i64::from_le_bytes(a)))?;
                }
                b'G' => {
                    let n = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    ensure(n.is_finite(), "non-finite metadata")?;
                    self.push(Node::Float(n))?;
                }
                b'X' | 0x8c => {
                    let n = if opcode == b'X' {
                        self.u32()?
                    } else {
                        self.byte()? as usize
                    };
                    let s = std::str::from_utf8(self.take(n)?)
                        .map_err(|_| bad("invalid Unicode"))?
                        .to_owned();
                    self.push(Node::String(s))?;
                }
                b'}' => self.push(Node::Dict(vec![]))?,
                b']' | b')' => self.push(Node::Seq(vec![]))?,
                b'l' | b't' => {
                    let v = self.marked()?;
                    self.push(Node::Seq(v))?;
                }
                b'd' => {
                    let v = self.marked()?;
                    let pairs = self.pairs(v)?;
                    self.push(Node::Dict(vec![]))?;
                    let id = self.top()?;
                    self.append_dict(id, pairs)?;
                }
                0x85..=0x87 => {
                    let n = (opcode - 0x84) as usize;
                    let mut values = Vec::new();
                    for _ in 0..n {
                        values.push(self.pop()?);
                    }
                    values.reverse();
                    self.push(Node::Seq(values))?;
                }
                b'a' | b'e' => {
                    let v = if opcode == b'a' {
                        vec![self.pop()?]
                    } else {
                        self.marked()?
                    };
                    let id = self.top()?;
                    let Node::Seq(list) = &mut self.nodes[id] else {
                        return Err(bad("APPEND target is not list"));
                    };
                    list.extend(v);
                }
                b's' | b'u' => {
                    let values = if opcode == b's' {
                        let val = self.pop()?;
                        let key = self.pop()?;
                        vec![key, val]
                    } else {
                        self.marked()?
                    };
                    let pairs = self.pairs(values)?;
                    self.append_dict(self.top()?, pairs)?;
                }
                b'q' | b'r' => {
                    let key = if opcode == b'q' {
                        self.byte()? as usize
                    } else {
                        self.u32()?
                    };
                    self.remember(key)?;
                }
                0x94 => self.remember(self.memo.len())?,
                b'h' | b'j' => {
                    let key = if opcode == b'h' {
                        self.byte()? as usize
                    } else {
                        self.u32()?
                    };
                    self.recall(key)?;
                }
                b'c' | 0x93 => {
                    let name = if opcode == b'c' {
                        format!("{}.{}", self.line()?, self.line()?)
                    } else {
                        let symbol = self.pop()?;
                        let module = self.pop()?;
                        format!("{}.{}", self.string(module)?, self.string(symbol)?)
                    };
                    ensure(
                        allowed_global(&name),
                        format!("forbidden pickle GLOBAL {name}"),
                    )?;
                    self.push(Node::Global(name))?;
                }
                b'Q' => {
                    let id = self.pop()?;
                    let storage = self.persistent(id)?;
                    self.push(storage)?;
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let value = self.reduce(callable, args)?;
                    self.push(value)?;
                }
                _ => {
                    return Err(bad(format!(
                        "unsupported opcode 0x{opcode:02x} at {}",
                        self.pos - 1
                    )))
                }
            }
        }
        Err(bad("missing STOP"))
    }
    fn encode(
        &self,
        id: Id,
        path: &str,
        depth: usize,
        active: &mut HashSet<Id>,
        tensors: &mut BTreeMap<String, RawTensor>,
        budget: &mut OutputBudget,
    ) -> Result<Value> {
        ensure(
            depth <= MAX_DEPTH && active.insert(id),
            "cyclic/deep pickle payload",
        )?;
        budget.visits += 1;
        ensure(
            budget.visits <= MAX_NODES,
            "expanded metadata node limit exceeded",
        )?;
        let added = match &self.nodes[id] {
            Node::String(v) => v.len(),
            Node::Tensor(t) => t.data.len(),
            _ => 32,
        };
        budget.bytes = budget
            .bytes
            .checked_add(added + path.len())
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| bad("expanded output limit exceeded"))?;
        let value = match &self.nodes[id] {
            Node::Null => Value::Null,
            Node::Bool(v) => json!(v),
            Node::Int(v) => json!(v),
            Node::Float(v) => json!(v),
            Node::String(v) => json!(v),
            Node::Seq(v) => Value::Array(
                v.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        self.encode(
                            x,
                            &format!("{path}/{i}"),
                            depth + 1,
                            active,
                            tensors,
                            budget,
                        )
                    })
                    .collect::<Result<_>>()?,
            ),
            Node::Dict(v) => {
                let mut o = serde_json::Map::new();
                for (k, x) in v {
                    ensure(
                        !k.contains('/'),
                        "dictionary key conflicts with tensor path separator",
                    )?;
                    o.insert(
                        k.clone(),
                        self.encode(
                            *x,
                            &format!("{path}/{k}"),
                            depth + 1,
                            active,
                            tensors,
                            budget,
                        )?,
                    );
                }
                Value::Object(o)
            }
            Node::Tensor(t) => {
                ensure(!tensors.contains_key(path), "duplicate tensor path")?;
                tensors.insert(path.into(), t.clone());
                json!({"__tensor__":path})
            }
            _ => return Err(bad("unresolved callable/storage in payload")),
        };
        active.remove(&id);
        Ok(value)
    }
}

/// Convert a known PyTorch archive to the existing portable Anny envelope.
/// Maximum input and aggregate decoded payload are 512 MiB; no code is executed.
pub fn convert(bytes: &[u8]) -> Result<Vec<u8>> {
    ensure(
        bytes.len() <= MAX_BYTES,
        "PyTorch file exceeds decoder limit",
    )?;
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| bad(e.to_string()))?;
    ensure(zip.len() <= 16384, "too many archive records")?;
    let mut files = BTreeMap::new();
    let mut total = 0usize;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| bad(e.to_string()))?;
        ensure(
            f.enclosed_name().is_some() && !f.is_dir(),
            "invalid archive entry path",
        )?;
        let size = usize::try_from(f.size()).map_err(|_| bad("record too large"))?;
        total = total
            .checked_add(size)
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| bad("expanded archive exceeds limit"))?;
        let name = f.name().to_owned();
        ensure(!files.contains_key(&name), "duplicate archive entry")?;
        let mut data = Vec::new();
        f.by_ref().take(size as u64 + 1).read_to_end(&mut data)?;
        ensure(data.len() == size, "archive record length mismatch")?;
        files.insert(name, data);
    }
    let names: Vec<_> = files
        .keys()
        .filter(|k| k.ends_with("/data.pkl"))
        .cloned()
        .collect();
    ensure(names.len() == 1, "expected one PyTorch data.pkl record")?;
    let prefix = names[0].strip_suffix("data.pkl").unwrap().to_owned();
    let big_endian = match files.get(&format!("{prefix}byteorder")).map(Vec::as_slice) {
        Some(b"big") => true,
        Some(b"little") | None => false,
        _ => return Err(bad("unsupported byte order")),
    };
    let pickle = files.remove(&names[0]).unwrap();
    let mut d = Decoder {
        bytes: &pickle,
        pos: 0,
        nodes: vec![],
        stack: vec![],
        memo: HashMap::new(),
        files,
        prefix,
        big_endian,
        decoded_bytes: 0,
    };
    let root = d.parse()?;
    let mut tensors = BTreeMap::new();
    let payload = d.encode(
        root,
        "root",
        0,
        &mut HashSet::new(),
        &mut tensors,
        &mut OutputBudget::default(),
    )?;
    let metadata = HashMap::from([
        ("anny_port_format".into(), "1".into()),
        ("anny_port_payload".into(), serde_json::to_string(&payload)?),
        (
            "source_sha256".into(),
            format!("{:x}", Sha256::digest(bytes)),
        ),
    ]);
    let views = tensors
        .iter()
        .map(|(k, t)| {
            Ok((
                k.clone(),
                TensorView::new(t.dtype, t.shape.clone(), &t.data)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors::serialize(views, &Some(metadata))?)
}
pub fn load(path: impl AsRef<Path>) -> Result<Archive> {
    ensure(
        std::fs::metadata(path.as_ref())?.len() <= MAX_BYTES as u64,
        "PyTorch file exceeds decoder limit",
    )?;
    Archive::from_bytes(&convert(&std::fs::read(path)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn archive(pickle: &[u8], files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut z = zip::ZipWriter::new(Cursor::new(vec![]));
        z.start_file("archive/data.pkl", zip::write::FileOptions::default())
            .unwrap();
        z.write_all(pickle).unwrap();
        for (name, bytes) in files {
            z.start_file(
                format!("archive/{name}"),
                zip::write::FileOptions::default(),
            )
            .unwrap();
            z.write_all(bytes).unwrap();
        }
        z.finish().unwrap().into_inner()
    }
    #[test]
    fn metadata_only() {
        let b = archive(
            b"\x80\x02}q\x00(X\x01\x00\x00\x00a]q\x01(K\x01K\x02eu.",
            &[],
        );
        let a = Archive::from_bytes(&convert(&b).unwrap()).unwrap();
        assert_eq!(a.payload().unwrap(), json!({"a":[1,2]}));
    }
    #[test]
    fn forbidden_globals_and_truncation() {
        assert!(convert(&archive(b"\x80\x02cos\nsystem\nX\x02\0\0\0id\x85R.", &[])).is_err());
        assert!(convert(&archive(b"\x80\x02X\xff\xff\xff\xff", &[])).is_err());
        assert!(convert(b"not a zip").is_err());
    }
    #[test]
    fn reject_cycles() {
        assert!(convert(&archive(b"\x80\x02]q\x00h\x00a.", &[])).is_err());
    }
    #[test]
    fn strided_tensor() {
        let p=b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n((X\x07\0\0\0storagectorch\nFloatStorage\nX\x01\0\0\x000X\x03\0\0\0cpuK\x06tQK\x00K\x03K\x02\x86K\x01K\x03\x86\x89ccollections\nOrderedDict\n)RtR.";
        let data: Vec<_> = (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let a = Archive::from_bytes(&convert(&archive(p, &[("data/0", &data)])).unwrap()).unwrap();
        assert_eq!(a.tensor("root").unwrap().data, vec![0., 3., 1., 4., 2., 5.]);
    }
}
