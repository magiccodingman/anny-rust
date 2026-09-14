// Shared scalar-specialized coefficient evaluation. No output-only casting.
impl Anny {
    pub fn coefficients(&self, p: &Parameters) -> Result<Tensor> {
        let ph = parse_values(
            &p.phenotype_kwargs,
            &self.phenotype_labels,
            0.5,
            "phenotype_kwargs",
        )?;
        let lo = parse_values(
            &p.local_changes_kwargs,
            &self.local_change_labels,
            0.,
            "local_changes_kwargs",
        )?;
        let fa = parse_values(
            &p.facial_actions,
            &self.facial_action_labels,
            0.,
            "facial_actions",
        )?;
        let batch = broadcast(&[ph.shape[0], lo.shape[0], fa.shape[0]])?;
        let mask = self.data.get("stacked_phenotype_blend_shapes_mask")?;
        let c = self.data.blendshape_count();
        let mut out = Tensor::zeros(vec![batch, c]);
        for b in 0..batch {
            let value = |key: &str| {
                self.phenotype_labels
                    .iter()
                    .position(|x| x == key)
                    .map_or(0.5, |i| ph.data[(b % ph.shape[0]) * ph.shape[1] + i])
            };
            let race = [value("african"), value("asian"), value("caucasian")];
            let sum: Scalar = race.iter().sum();
            let mut features = Vec::with_capacity(26);
            for (feature, anchors_names) in PHENOTYPE_VARIATIONS {
                if *feature == "race" {
                    features.extend(race.iter().map(|r| {
                        let v = r / sum;
                        if v.is_finite() {
                            v
                        } else {
                            1. / 3.
                        }
                    }));
                    continue;
                }
                let min = if *feature == "age" { -1. / 3. } else { 0. };
                let max = 1.;
                let anchors: Vec<_> = (0..anchors_names.len())
                    .map(|i| {
                        min + (max - min) * (i as Scalar) / (anchors_names.len() - 1) as Scalar
                    })
                    .collect();
                features.extend(linear_interpolation(
                    value(feature),
                    &anchors,
                    self.config.extrapolate_phenotypes,
                )?);
            }
            for i in 0..mask.shape[0] {
                let mut w = 1.;
                for (k, &v) in features.iter().enumerate() {
                    if mask.data[i * 26 + k] != 0. {
                        w *= v;
                    }
                }
                out.data[b * c + i] = w;
            }
            let off = mask.shape[0];
            for i in 0..fa.shape[1] {
                out.data[b * c + off + i] = fa.data[(b % fa.shape[0]) * fa.shape[1] + i];
            }
            let off = off + fa.shape[1];
            for i in 0..lo.shape[1] {
                let v = lo.data[(b % lo.shape[0]) * lo.shape[1] + i];
                out.data[b * c + off + 2 * i] = v.max(0.);
                out.data[b * c + off + 2 * i + 1] = (-v).max(0.);
            }
        }
        Ok(out)
    }
}
