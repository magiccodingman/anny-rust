//! Where `derive measure` spends its time: the forward pass, the anthropometry
//! construction, or the measurement query itself.
//!
//! `cargo run -p anny-core --release --example measure_split`

use anny_core::tools::Anthropometry;
use anny_core::{Anny, AnnyConfig, Parameters};
use std::time::Instant;

const REPS: u32 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output/ci-model.safetensors".to_string());
    let bytes = std::fs::read(&path)?;
    let model = Anny::from_bytes(&bytes, Some(AnnyConfig::default()))?;
    let params = Parameters::default();

    let start = Instant::now();
    let mut result = None;
    for _ in 0..REPS {
        result = Some(model.forward(&params)?);
    }
    let forward = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    let rest = result
        .as_ref()
        .expect("one iteration")
        .get("rest_vertices")?;

    let start = Instant::now();
    let mut anthropometry = None;
    for _ in 0..REPS {
        anthropometry = Some(Anthropometry::new(&model)?);
    }
    let new = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    let anthropometry = anthropometry.expect("one iteration");

    let start = Instant::now();
    for _ in 0..REPS {
        std::hint::black_box(anthropometry.measure(rest)?);
    }
    let query = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

    println!("forward            {forward:>8.3} ms");
    println!("Anthropometry::new {new:>8.3} ms");
    println!("measure(rest)      {query:>8.3} ms");
    println!("total              {:>8.3} ms", forward + new + query);
    println!(
        "{}",
        serde_json::to_string_pretty(&anthropometry.measure(rest)?)?
    );
    Ok(())
}
