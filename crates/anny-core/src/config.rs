// Port of NAVER Anny configuration semantics, Copyright (C) 2025 NAVER Corp.
// SPDX-License-Identifier: Apache-2.0
use crate::{ensure, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PoseParameterization {
    World,
    LocalBoneWorld,
    LocalBone,
    #[default]
    LocalRef,
    WorldOrient,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkinningMethod {
    #[default]
    Lbs,
    Dqs,
    WarpLbs,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoneOrientation {
    Blender,
    Procrustes,
    #[default]
    Cached,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Selection {
    Preset(String),
    Labels(Vec<String>),
}
impl Default for Selection {
    fn default() -> Self {
        Self::Preset("none".into())
    }
}
impl Selection {
    pub fn all() -> Self {
        Self::Preset("all".into())
    }
    pub fn mask(&self, available: &[String], local: bool) -> Result<Vec<bool>> {
        match self {
            Self::Preset(p) => match p.as_str() {
                "none" => Ok(vec![false; available.len()]),
                "all" => Ok(vec![true; available.len()]),
                "default" if local => Ok(available
                    .iter()
                    .map(|x| !x.to_lowercase().contains("nipple"))
                    .collect()),
                _ => Err(Error::Invalid(format!("unknown parameter selection: {p}"))),
            },
            Self::Labels(labels) => {
                let unique: BTreeSet<_> = labels.iter().collect();
                ensure(unique.len() == labels.len(), "duplicate selected label")?;
                for name in labels {
                    ensure(
                        available.contains(name),
                        format!("unknown selected label {name}"),
                    )?;
                }
                Ok(available.iter().map(|x| unique.contains(x)).collect())
            }
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RigSpec {
    Name(String),
    Config(RigConfig),
}
impl Default for RigSpec {
    fn default() -> Self {
        Self::Name("anny".into())
    }
}
impl RigSpec {
    pub fn resolve(&self) -> Result<RigConfig> {
        match self {
            Self::Name(n) => RigConfig::parse(n),
            Self::Config(c) => Ok(c.clone()),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TopologySpec {
    Name(String),
    Config(TopologyConfig),
}
impl Default for TopologySpec {
    fn default() -> Self {
        Self::Name("anny".into())
    }
}
impl TopologySpec {
    pub fn resolve(&self) -> Result<TopologyConfig> {
        match self {
            Self::Name(n) => TopologyConfig::parse(n),
            Self::Config(c) => Ok(c.clone()),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnnyConfig {
    pub rig: RigSpec,
    pub topology: TopologySpec,
    pub local_changes: Selection,
    pub facial_actions: Selection,
    pub phenotypes: String,
    pub extrapolate_phenotypes: bool,
    pub pose_parameterization: PoseParameterization,
    pub skinning_method: SkinningMethod,
}
impl Default for AnnyConfig {
    fn default() -> Self {
        Self {
            rig: RigSpec::default(),
            topology: TopologySpec::default(),
            local_changes: Selection::default(),
            facial_actions: Selection::default(),
            phenotypes: "default".into(),
            extrapolate_phenotypes: false,
            pose_parameterization: PoseParameterization::default(),
            skinning_method: SkinningMethod::default(),
        }
    }
}
impl AnnyConfig {
    pub fn validate(&self) -> Result<()> {
        self.rig.resolve()?;
        self.topology.resolve()?;
        ensure(
            matches!(self.phenotypes.as_str(), "default" | "all"),
            "phenotypes must be default or all",
        )
    }
    pub fn phenotype_labels(&self) -> Vec<String> {
        PHENOTYPE_LABELS
            .iter()
            .filter(|&&x| {
                self.phenotypes == "all"
                    || !["cupsize", "firmness", "african", "asian", "caucasian"].contains(&x)
            })
            .map(|s| (*s).into())
            .collect()
    }
    /// The deprecated Python fullbody factory defaults, not modern Anny defaults.
    pub fn legacy_fullbody() -> Self {
        Self {
            rig: RigSpec::Name("makehuman".into()),
            topology: TopologySpec::Name("anny-quads".into()),
            pose_parameterization: PoseParameterization::LocalBone,
            ..Self::default()
        }
    }
    pub fn hand(side: &str) -> Result<Self> {
        ensure(matches!(side, "L" | "R"), "hand side must be L or R")?;
        Ok(Self {
            rig: RigSpec::Name(format!("anny-hand.{side}")),
            topology: TopologySpec::Name(format!("hand.{side}")),
            ..Self::default()
        })
    }
    pub fn head() -> Self {
        Self {
            rig: RigSpec::Name("makehuman-head".into()),
            topology: TopologySpec::Name("head".into()),
            ..Self::default()
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RigConfig {
    pub base_rig: String,
    pub bone_orientation: BoneOrientation,
    pub root_identity_orientation: bool,
    pub weights_filename: Option<String>,
    pub bones_to_remove: BTreeSet<String>,
    pub subtree_root: Option<String>,
}
impl Default for RigConfig {
    fn default() -> Self {
        Self {
            base_rig: "anny".into(),
            bone_orientation: BoneOrientation::Cached,
            root_identity_orientation: true,
            weights_filename: None,
            bones_to_remove: BTreeSet::new(),
            subtree_root: None,
        }
    }
}
impl RigConfig {
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<_> = spec.split('-').collect();
        let base = parts[0];
        let mods = &parts[1..];
        ensure(
            [
                "anny",
                "makehuman",
                "cmu_mb",
                "game_engine",
                "mixamo",
                "soma",
            ]
            .contains(&base),
            format!("unknown rig {base}"),
        )?;
        let mut r = Self {
            base_rig: base.into(),
            bone_orientation: if ["anny", "soma"].contains(&base) {
                BoneOrientation::Cached
            } else {
                BoneOrientation::Blender
            },
            root_identity_orientation: !["anny", "soma"].contains(&base),
            ..Self::default()
        };
        if base == "soma" {
            ensure(mods.is_empty(), "soma rig does not support modifiers")?;
            return Ok(r);
        }
        ensure(
            !(mods.contains(&"blender") && mods.contains(&"procrustes")),
            "conflicting bone orientation modifiers",
        )?;
        for m in mods {
            match *m {
                "head" | "hand.L" | "hand.R" => {
                    ensure(
                        ["anny", "makehuman"].contains(&base) && r.subtree_root.is_none(),
                        "invalid or multiple subtree selectors",
                    )?;
                    r.subtree_root = Some(
                        match *m {
                            "head" => "neck01",
                            "hand.L" => "wrist.L",
                            _ => "wrist.R",
                        }
                        .into(),
                    );
                }
                "blender" => {
                    r.bone_orientation = BoneOrientation::Blender;
                    r.root_identity_orientation = false;
                }
                "procrustes" => r.bone_orientation = BoneOrientation::Procrustes,
                "rootidentity" => {} // applied after orientation, as upstream
                _ => r.remove_modifier(m)?,
            }
        }
        if mods.contains(&"rootidentity") {
            r.root_identity_orientation = true;
        }
        if base == "anny" {
            for m in ["notongue", "nobreasts", "nofacialexpression", "pruned"] {
                r.remove_modifier(m)?;
            }
        }
        Ok(r)
    }
    fn remove_modifier(&mut self, m: &str) -> Result<()> {
        let labels: &[&str] = match m {
            "pruned" => ZERO_WEIGHT_BONE_LABELS,
            "noeyes" => EYE_BONE_LABELS,
            "notongue" => TONGUE_BONE_LABELS,
            "nofacialexpression" => FACIAL_EXPRESSION_BONE_LABELS,
            "notoes" => TOE_BONE_LABELS,
            "nohands" => HAND_BONE_LABELS,
            "nobreasts" => BREAST_BONE_LABELS,
            "noexpression" => {
                for m in ["noeyes", "notongue", "nofacialexpression"] {
                    self.remove_modifier(m)?;
                }
                return Ok(());
            }
            _ => return Err(Error::Invalid(format!("unknown rig modifier {m}"))),
        };
        self.bones_to_remove
            .extend(labels.iter().map(|x| (*x).into()));
        Ok(())
    }
    pub fn preset_files(&self) -> Result<(String, String)> {
        let preset = match self.base_rig.as_str() {
            "anny" | "makehuman" => "default",
            "cmu_mb" => "cmu_mb",
            "game_engine" => "game_engine",
            "mixamo" => "mixamo",
            _ => {
                let weights = self
                    .weights_filename
                    .clone()
                    .ok_or_else(|| Error::Invalid("custom rig needs weights_filename".into()))?;
                return Ok((self.base_rig.clone(), weights));
            }
        };
        Ok((
            format!("mpfb2/rigs/standard/rig.{preset}.json"),
            self.weights_filename
                .clone()
                .unwrap_or_else(|| format!("mpfb2/rigs/standard/weights.{preset}.json")),
        ))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TopologyConfig {
    pub base_mesh: String,
    pub nudity_edits: bool,
    pub remove_unattached_vertices: bool,
    pub triangulate_faces: bool,
    pub eyes: bool,
    pub tongue: bool,
    pub submodel: String,
}
impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            base_mesh: "makehuman".into(),
            nudity_edits: true,
            remove_unattached_vertices: true,
            triangulate_faces: true,
            eyes: true,
            tongue: true,
            submodel: "body".into(),
        }
    }
}
impl TopologyConfig {
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<_> = spec.split('-').collect();
        let base = parts[0];
        let mut t = Self::default();
        match base {
            "anny" => {}
            "makehuman" => {
                t.nudity_edits = false;
                t.remove_unattached_vertices = false;
                t.triangulate_faces = false;
            }
            "head" | "hand.L" | "hand.R" => {
                t.submodel = base.into();
                t.nudity_edits = false;
                t.eyes = base == "head";
                t.tongue = base == "head";
            }
            _ if ALTERNATIVE_TOPOLOGIES.contains(&base) => {
                t.base_mesh = base.into();
                t.nudity_edits = false;
            }
            _ => return Err(Error::Invalid(format!("unknown topology {base}"))),
        }
        for m in &parts[1..] {
            match *m {
                "noeyes" if t.base_mesh == "makehuman" => t.eyes = false,
                "notongue" if t.base_mesh == "makehuman" => t.tongue = false,
                "quads" => t.triangulate_faces = false,
                "tris" => t.triangulate_faces = true,
                "full" => t.remove_unattached_vertices = false,
                _ => return Err(Error::Invalid(format!("unknown topology modifier {m}"))),
            }
        }
        Ok(t)
    }
}
pub const ALTERNATIVE_TOPOLOGIES: &[&str] = &[
    "smplx",
    "smpl",
    "soma",
    "anny_from_soma",
    "notoes",
    "notoes_collapse3pc",
    "notoes_collapse5pc",
    "notoes_collapse10pc",
    "legacy_default",
];
pub const PHENOTYPE_LABELS: &[&str] = &[
    "gender",
    "age",
    "muscle",
    "weight",
    "height",
    "proportions",
    "cupsize",
    "firmness",
    "african",
    "asian",
    "caucasian",
];
pub const PHENOTYPE_VARIATIONS: &[(&str, &[&str])] = &[
    ("race", &["african", "asian", "caucasian"]),
    ("gender", &["male", "female"]),
    ("age", &["newborn", "baby", "child", "young", "old"]),
    ("muscle", &["minmuscle", "averagemuscle", "maxmuscle"]),
    ("weight", &["minweight", "averageweight", "maxweight"]),
    ("height", &["minheight", "maxheight"]),
    ("proportions", &["idealproportions", "uncommonproportions"]),
    ("cupsize", &["mincup", "averagecup", "maxcup"]),
    (
        "firmness",
        &["minfirmness", "averagefirmness", "maxfirmness"],
    ),
];
pub const EYE_BONE_LABELS: &[&str] = &["eye.L", "eye.R"];
pub const TONGUE_BONE_LABELS: &[&str] = &[
    "tongue00",
    "tongue01",
    "tongue02",
    "tongue03",
    "tongue04",
    "tongue05.L",
    "tongue05.R",
    "tongue06.L",
    "tongue06.R",
    "tongue07.L",
    "tongue07.R",
];
pub const FACIAL_EXPRESSION_BONE_LABELS: &[&str] = &[
    "jaw",
    "levator02.L",
    "levator02.R",
    "levator03.L",
    "levator03.R",
    "levator04.L",
    "levator04.R",
    "levator05.L",
    "levator05.R",
    "levator06.L",
    "levator06.R",
    "oculi01.L",
    "oculi01.R",
    "oculi02.L",
    "oculi02.R",
    "orbicularis03.L",
    "orbicularis03.R",
    "orbicularis04.L",
    "orbicularis04.R",
    "oris01",
    "oris02",
    "oris03.L",
    "oris03.R",
    "oris04.L",
    "oris04.R",
    "oris05",
    "oris06",
    "oris06.L",
    "oris06.R",
    "oris07.L",
    "oris07.R",
    "risorius02.L",
    "risorius02.R",
    "risorius03.L",
    "risorius03.R",
    "special01",
    "special03",
    "special04",
    "special05.L",
    "special05.R",
    "special06.L",
    "special06.R",
    "temporalis01.L",
    "temporalis01.R",
    "temporalis02.L",
    "temporalis02.R",
];
pub const ZERO_WEIGHT_BONE_LABELS: &[&str] = &[
    "levator02.L",
    "levator02.R",
    "levator03.L",
    "levator03.R",
    "levator04.L",
    "levator04.R",
    "oculi02.L",
    "oculi02.R",
    "oris02",
    "oris04.L",
    "oris04.R",
    "oris06",
    "oris06.L",
    "oris06.R",
    "risorius02.L",
    "risorius02.R",
    "special01",
    "special03",
    "special06.L",
    "special06.R",
    "temporalis01.L",
    "temporalis01.R",
    "temporalis02.L",
    "temporalis02.R",
];
pub const TOE_BONE_LABELS: &[&str] = &[
    "toe1-1.L", "toe1-1.R", "toe1-2.L", "toe1-2.R", "toe2-1.L", "toe2-1.R", "toe2-2.L", "toe2-2.R",
    "toe2-3.L", "toe2-3.R", "toe3-1.L", "toe3-1.R", "toe3-2.L", "toe3-2.R", "toe3-3.L", "toe3-3.R",
    "toe4-1.L", "toe4-1.R", "toe4-2.L", "toe4-2.R", "toe4-3.L", "toe4-3.R", "toe5-1.L", "toe5-1.R",
    "toe5-2.L", "toe5-2.R", "toe5-3.L", "toe5-3.R",
];
pub const HAND_BONE_LABELS: &[&str] = &[
    "finger1-1.L",
    "finger1-1.R",
    "finger1-2.L",
    "finger1-2.R",
    "finger1-3.L",
    "finger1-3.R",
    "finger2-1.L",
    "finger2-1.R",
    "finger2-2.L",
    "finger2-2.R",
    "finger2-3.L",
    "finger2-3.R",
    "finger3-1.L",
    "finger3-1.R",
    "finger3-2.L",
    "finger3-2.R",
    "finger3-3.L",
    "finger3-3.R",
    "finger4-1.L",
    "finger4-1.R",
    "finger4-2.L",
    "finger4-2.R",
    "finger4-3.L",
    "finger4-3.R",
    "finger5-1.L",
    "finger5-1.R",
    "finger5-2.L",
    "finger5-2.R",
    "finger5-3.L",
    "finger5-3.R",
    "metacarpal1.L",
    "metacarpal1.R",
    "metacarpal2.L",
    "metacarpal2.R",
    "metacarpal3.L",
    "metacarpal3.R",
    "metacarpal4.L",
    "metacarpal4.R",
];
pub const BREAST_BONE_LABELS: &[&str] = &["breast.L", "breast.R"];
pub const FACIAL_ACTION_LABELS: &[&str] = &[
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
];
