// Port of NAVER Anny age mappings and conditional shape distributions.
// Copyright (C) 2025 NAVER Corp. SPDX-License-Identifier: Apache-2.0
use crate::{
    assets::AssetStore, ensure, math::linear_interpolation, model::Anny, tensor::Archive, Error,
    Parameters, Result,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MorphologicalAgeMapping {
    pub anny_age_anchors: Vec<f64>,
    pub morphological_age_anchors: Vec<f64>,
}
fn interpolate(x: f64, anchors: &[f64], values: &[f64], extrapolate: bool) -> Result<f64> {
    ensure(anchors.len() == values.len(), "anchor/value count mismatch")?;
    Ok(linear_interpolation(x, anchors, extrapolate)?
        .iter()
        .zip(values)
        .map(|(a, b)| a * b)
        .sum())
}
impl MorphologicalAgeMapping {
    pub fn morphological_to_anny_age(&self, x: f64) -> Result<f64> {
        interpolate(
            x,
            &self.morphological_age_anchors,
            &self.anny_age_anchors,
            true,
        )
    }
    pub fn anny_to_morphological_age(&self, x: f64) -> Result<f64> {
        interpolate(
            x,
            &self.anny_age_anchors,
            &self.morphological_age_anchors,
            true,
        )
    }
    fn load(a: &Archive) -> Result<Self> {
        Ok(Self {
            anny_age_anchors: a
                .payload_tensor("morphological_age_mapping/anny_age_anchors")?
                .data
                .clone(),
            morphological_age_anchors: a
                .payload_tensor("morphological_age_mapping/morphological_age_anchors")?
                .data
                .clone(),
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionalBetaDistribution {
    pub age_anchors: Vec<f64>,
    pub alpha_anchors: Vec<f64>,
    pub beta_anchors: Vec<f64>,
}
impl ConditionalBetaDistribution {
    pub fn parameters(&self, age: f64) -> Result<(f64, f64)> {
        let a = interpolate(age, &self.age_anchors, &self.alpha_anchors, false)?;
        let b = interpolate(age, &self.age_anchors, &self.beta_anchors, false)?;
        ensure(
            a > 0. && b > 0. && a.is_finite() && b.is_finite(),
            "beta shape parameters must be finite and positive",
        )?;
        Ok((a, b))
    }
    pub fn sample(&self, age: f64, rng: &mut ShapeRng) -> Result<f64> {
        let (a, b) = self.parameters(age)?;
        let x = rng.gamma(a)?;
        let y = rng.gamma(b)?;
        ensure(
            x + y > 0. && (x + y).is_finite(),
            "beta sampler underflow; parameters are too small",
        )?;
        Ok((x / (x + y)).clamp(f64::MIN_POSITIVE, 1. - f64::EPSILON))
    }
    pub fn log_probability(&self, age: f64, value: f64) -> Result<f64> {
        ensure(
            value > 0. && value < 1.,
            "beta density expects a value strictly inside (0,1)",
        )?;
        let (a, b) = self.parameters(age)?;
        Ok(
            (a - 1.) * value.ln() + (b - 1.) * (-value).ln_1p() - log_gamma(a) - log_gamma(b)
                + log_gamma(a + b),
        )
    }
    fn load(a: &Archive, name: &str) -> Result<Self> {
        let prefix = format!("conditional_{name}_distribution");
        Ok(Self {
            age_anchors: a
                .payload_tensor(&format!("{prefix}/age_anchors"))?
                .data
                .clone(),
            alpha_anchors: a
                .payload_tensor(&format!("{prefix}/alpha_anchors"))?
                .data
                .clone(),
            beta_anchors: a
                .payload_tensor(&format!("{prefix}/beta_anchors"))?
                .data
                .clone(),
        })
    }
}
/// Same calibrated distribution as upstream. The explicit, portable RNG is not
/// PyTorch's RNG, so equal seeds do not promise identical Python sample sequences.
#[derive(Clone, Debug)]
pub struct SimpleShapeDistribution {
    pub age_mapping: MorphologicalAgeMapping,
    pub boys: BTreeMap<String, ConditionalBetaDistribution>,
    pub girls: BTreeMap<String, ConditionalBetaDistribution>,
    pub phenotype_labels: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SampleOptions {
    pub count: usize,
    pub seed: u64,
    pub morphological_age_range: [f64; 2],
    pub gender_range: [f64; 2],
}
impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            count: 1,
            seed: 0,
            morphological_age_range: [0., 90.],
            gender_range: [0., 1.],
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct ShapeSamples {
    pub morphological_age: Vec<f64>,
    pub parameters: Parameters,
}
impl SimpleShapeDistribution {
    pub fn load(store: &AssetStore, model: &Anny) -> Result<Self> {
        let boys = store.converted("shape_calibration/boys.pth")?;
        let girls = store.converted("shape_calibration/girls.pth")?;
        let age_mapping = MorphologicalAgeMapping::load(&boys)?;
        let other = MorphologicalAgeMapping::load(&girls)?;
        ensure(
            age_mapping.anny_age_anchors == other.anny_age_anchors
                && age_mapping.morphological_age_anchors == other.morphological_age_anchors,
            "boys/girls age calibration mismatch",
        )?;
        let load = |a: &Archive| {
            ["height", "weight", "muscle", "proportions"]
                .iter()
                .map(|s| Ok((s.to_string(), ConditionalBetaDistribution::load(a, s)?)))
                .collect::<Result<BTreeMap<_, _>>>()
        };
        Ok(Self {
            age_mapping,
            boys: load(&boys)?,
            girls: load(&girls)?,
            phenotype_labels: model.phenotype_labels.clone(),
        })
    }
    pub fn sample(&self, options: &SampleOptions) -> Result<ShapeSamples> {
        ensure(options.count > 0, "sample count must be positive")?;
        for r in [options.morphological_age_range, options.gender_range] {
            ensure(
                r.iter().all(|x| x.is_finite()) && r[0] <= r[1],
                "invalid sampling range",
            )?;
        }
        ensure(
            options.gender_range[0] >= 0. && options.gender_range[1] <= 1.,
            "gender range must be inside [0,1]",
        )?;
        let mut rng = ShapeRng::new(options.seed);
        let mut ph: BTreeMap<String, Vec<f64>> = self
            .phenotype_labels
            .iter()
            .map(|s| (s.clone(), Vec::with_capacity(options.count)))
            .collect();
        let mut ages = Vec::new();
        for _ in 0..options.count {
            let age = options.morphological_age_range[0]
                + rng.uniform()
                    * (options.morphological_age_range[1] - options.morphological_age_range[0]);
            let gender = options.gender_range[0]
                + rng.uniform() * (options.gender_range[1] - options.gender_range[0]);
            let a = self.age_mapping.morphological_to_anny_age(age)?;
            ages.push(age);
            for (name, values) in &mut ph {
                let value = match name.as_str() {
                    "age" => a,
                    "gender" => gender,
                    "height" | "weight" | "muscle" | "proportions" => {
                        let d = if gender <= 0.5 {
                            &self.boys
                        } else {
                            &self.girls
                        };
                        d[name].sample(a, &mut rng)?
                    }
                    _ => rng.uniform(),
                };
                values.push(value);
            }
        }
        Ok(ShapeSamples {
            morphological_age: ages,
            parameters: Parameters {
                phenotype_kwargs: json!(ph),
                ..Default::default()
            },
        })
    }
    pub fn prior_loss(&self, phenotypes: &BTreeMap<String, f64>) -> Result<f64> {
        let value = |s: &str| phenotypes.get(s).copied().unwrap_or(0.5);
        let gender = value("gender").clamp(1e-6, 1. - 1e-6);
        let mut logp = 0.;
        for key in ["height", "weight", "muscle", "proportions"] {
            let b = self
                .boys
                .get(key)
                .ok_or_else(|| Error::Invalid(format!("missing {key} calibration")))?;
            let g = &self.girls[key];
            let x = value(key).clamp(1e-6, 1. - 1e-6);
            let a = (-gender).ln_1p() + b.log_probability(value("age"), x)?;
            let b = gender.ln() + g.log_probability(value("age"), x)?;
            let m = a.max(b);
            logp += m + ((a - m).exp() + (b - m).exp()).ln();
        }
        Ok(-logp)
    }
}
/// SplitMix64 plus Box-Muller and Marsaglia-Tsang: no platform entropy required.
#[derive(Clone, Debug)]
pub struct ShapeRng {
    state: u64,
}
impl ShapeRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9007199254740992.
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    pub fn gamma(&mut self, a: f64) -> Result<f64> {
        ensure(a > 0. && a.is_finite(), "gamma shape must be positive")?;
        if a < 1. {
            return Ok(self.gamma(a + 1.)? * self.uniform().powf(1. / a));
        }
        let d = a - 1. / 3.;
        let c = (9. * d).sqrt().recip();
        for _ in 0..100_000 {
            let x = self.normal();
            let v = 1. + c * x;
            if v <= 0. {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u < 1. - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1. - v + v.ln()) {
                return Ok(d * v);
            }
        }
        Err(Error::Invalid("gamma sampler failed to converge".into()))
    }
}
fn log_gamma(x: f64) -> f64 {
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - log_gamma(1. - x);
    }
    let z = x - 1.;
    let mut a = 0.9999999999998099;
    for (i, p) in p.iter().enumerate() {
        a += p / (z + i as f64 + 1.);
    }
    let t = z + 7.5;
    0.5 * (2. * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}
