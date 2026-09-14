//! Analytic gradient of the upstream calibrated, age-conditioned shape prior.
use crate::{
    distribution::{ConditionalBetaDistribution, SimpleShapeDistribution},
    ensure, Error, Result,
};
use std::collections::BTreeMap;

// All beta-shape parameters are positive. Recurrence plus the asymptotic
// expansion avoids a numerical perturbation of the calibration or the model.
fn digamma(mut x: f64) -> f64 {
    let mut answer = 0.;
    while x < 8. {
        answer -= 1. / x;
        x += 1.;
    }
    let r = 1. / x;
    let r2 = r * r;
    answer + x.ln()
        - 0.5 * r
        - r2 * (1. / 12. - r2 * (1. / 120. - r2 * (1. / 252. - r2 * (1. / 240. - r2 * 5. / 660.))))
}
fn slope(x: f64, anchors: &[f64], values: &[f64]) -> f64 {
    if x < anchors[0] || x > anchors[anchors.len() - 1] {
        return 0.;
    }
    let i = anchors
        .partition_point(|&a| a < x)
        .clamp(1, anchors.len() - 1);
    (values[i] - values[i - 1]) / (anchors[i] - anchors[i - 1])
}
fn beta_gradient(d: &ConditionalBetaDistribution, age: f64, x: f64) -> Result<(f64, f64, f64)> {
    let (a, b) = d.parameters(age)?; // Also validates all interpolation dimensions/anchors.
    let logp = d.log_probability(age, x)?;
    let dx = (a - 1.) / x - (b - 1.) / (1. - x);
    let common = digamma(a + b);
    let da = x.ln() - digamma(a) + common;
    let db = (-x).ln_1p() - digamma(b) + common;
    let dage = da * slope(age, &d.age_anchors, &d.alpha_anchors)
        + db * slope(age, &d.age_anchors, &d.beta_anchors);
    Ok((logp, dx, dage))
}
/// Return the calibrated loss and its derivatives w.r.t. age, gender, height,
/// weight, muscle and proportions. The same epsilon clamp and gender mixture
/// as `SimpleShapeDistribution::prior_loss` are used. At interpolation knots
/// the left segment is used, and clamp derivatives vanish outside the interval.
pub fn shape_prior_gradient(
    d: &SimpleShapeDistribution,
    ph: &BTreeMap<String, f64>,
) -> Result<(f64, BTreeMap<String, f64>)> {
    ensure(
        ph.values().all(|x| x.is_finite()),
        "non-finite prior phenotype",
    )?;
    let value = |s: &str| ph.get(s).copied().unwrap_or(0.5);
    let active = |x: f64| (1e-6..=1. - 1e-6).contains(&x);
    let gender = value("gender").clamp(1e-6, 1. - 1e-6);
    let mut gradient = BTreeMap::from([("age".into(), 0.), ("gender".into(), 0.)]);
    let mut loss = 0.;
    for name in ["height", "weight", "muscle", "proportions"] {
        let boys = d
            .boys
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing {name} calibration")))?;
        let girls = d
            .girls
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing {name} calibration")))?;
        let x = value(name).clamp(1e-6, 1. - 1e-6);
        let (lb, xb, ab) = beta_gradient(boys, value("age"), x)?;
        let (lg, xg, ag) = beta_gradient(girls, value("age"), x)?;
        let lb = lb + (-gender).ln_1p();
        let lg = lg + gender.ln();
        let maximum = lb.max(lg);
        let wb = (lb - maximum).exp();
        let wg = (lg - maximum).exp();
        let total = wb + wg;
        loss -= maximum + total.ln();
        let (wb, wg) = (wb / total, wg / total);
        gradient.insert(
            name.into(),
            if active(value(name)) {
                -(wb * xb + wg * xg)
            } else {
                0.
            },
        );
        *gradient.get_mut("age").unwrap() -= wb * ab + wg * ag;
        if active(value("gender")) {
            *gradient.get_mut("gender").unwrap() += wb / (1. - gender) - wg / gender;
        }
    }
    ensure(
        loss.is_finite() && gradient.values().all(|x| x.is_finite()),
        "non-finite shape prior gradient",
    )?;
    Ok((loss, gradient))
}
