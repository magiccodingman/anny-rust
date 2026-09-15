using System.Collections.Generic;
using UnityEngine;

namespace Anny
{
    /// <summary>What the anny rig offers Unity's humanoid definition, and what it cannot.</summary>
    public sealed class AnnyHumanoidReport
    {
        /// <summary>Labels mapped onto a <see cref="HumanBodyBones"/> slot, in slot order.</summary>
        public string[] MappedLabels;

        /// <summary>The slots those labels fill.</summary>
        public HumanBodyBones[] MappedBones;

        /// <summary>Required slots the rig does not carry. Empty means the mapping is complete.</summary>
        public HumanBodyBones[] MissingRequired;

        /// <summary>
        /// Labels with no humanoid slot. These are real bones that Unity's humanoid clips will not
        /// drive: the rig puts two bones in each limb segment and fifteen in each foot, where the
        /// humanoid definition has one slot each.
        /// </summary>
        public string[] UnmappedLabels;

        public int BoneCount;
        public bool Complete { get { return MissingRequired.Length == 0; } }

        public override string ToString()
        {
            return "mapped " + MappedLabels.Length + "/" + BoneCount + " bones onto " + MappedBones.Length
                + " humanoid slots, missing " + MissingRequired.Length + " required, "
                + UnmappedLabels.Length + " bones left outside the humanoid definition";
        }
    }

    /// <summary>
    /// Maps an Anny rig onto Unity's humanoid (Mecanim) definition so an <see cref="Animator"/> can
    /// drive a character from any humanoid clip set.
    /// <para>
    /// <b>The hierarchy is not touched.</b> The anny rig already satisfies Unity's topology — its
    /// <c>root</c> bone sits at the pelvis, so it can be Hips with both legs and the spine beneath it
    /// — and every pose in this package is applied as global bone matrices, so nothing here depends
    /// on the parent chain either way. <see cref="AnnyHumanoidTests"/> asserts both facts.
    /// </para>
    /// <para>
    /// Two things are deliberately not papered over. This rig carries <em>two</em> bones per limb
    /// segment (<c>upperarm01</c>/<c>upperarm02</c> and so on) and fifteen toe bones per foot, while
    /// the humanoid definition has one slot for each. The surplus bones are reported as
    /// <see cref="AnnyHumanoidReport.UnmappedLabels"/>: humanoid clips will not drive them, so a
    /// humanoid playback is an approximation of Anny's own pose, not a reproduction of it. Anny's
    /// baked clips remain the exact path.
    /// </para>
    /// </summary>
    public static class AnnyHumanoid
    {
        /// <summary>The slots a humanoid avatar cannot do without.</summary>
        public static readonly HumanBodyBones[] Required =
        {
            HumanBodyBones.Hips,
            HumanBodyBones.Spine,
            HumanBodyBones.Head,
            HumanBodyBones.LeftUpperArm,
            HumanBodyBones.LeftLowerArm,
            HumanBodyBones.LeftHand,
            HumanBodyBones.RightUpperArm,
            HumanBodyBones.RightLowerArm,
            HumanBodyBones.RightHand,
            HumanBodyBones.LeftUpperLeg,
            HumanBodyBones.LeftLowerLeg,
            HumanBodyBones.LeftFoot,
            HumanBodyBones.RightUpperLeg,
            HumanBodyBones.RightLowerLeg,
            HumanBodyBones.RightFoot,
        };

        private static readonly KeyValuePair<string, HumanBodyBones>[] Table = BuildTable();

        /// <summary>Label to slot, in the order the slots are filled.</summary>
        public static IReadOnlyList<KeyValuePair<string, HumanBodyBones>> Mapping
        {
            get { return Table; }
        }

        /// <summary>The slot a label fills, or <see cref="HumanBodyBones.LastBone"/> when it has none.</summary>
        public static HumanBodyBones SlotOf(string label)
        {
            for (int i = 0; i < Table.Length; i++)
            {
                if (Table[i].Key == label)
                {
                    return Table[i].Value;
                }
            }

            return HumanBodyBones.LastBone;
        }

        /// <summary>Fills in which labels the rig has, which required slots are missing, and what is left over.</summary>
        public static AnnyHumanoidReport Describe(AnnyRig rig)
        {
            if (rig == null)
            {
                throw new System.ArgumentNullException(nameof(rig));
            }

            List<string> mappedLabels = new List<string>();
            List<HumanBodyBones> mappedBones = new List<HumanBodyBones>();
            List<string> unmapped = new List<string>();

            for (int b = 0; b < rig.Count; b++)
            {
                string label = rig.Labels[b];
                if (SlotOf(label) != HumanBodyBones.LastBone)
                {
                    mappedLabels.Add(label);
                    mappedBones.Add(SlotOf(label));
                }
                else
                {
                    unmapped.Add(label);
                }
            }

            List<HumanBodyBones> missing = new List<HumanBodyBones>();
            for (int i = 0; i < Required.Length; i++)
            {
                if (!mappedBones.Contains(Required[i]))
                {
                    missing.Add(Required[i]);
                }
            }

            return new AnnyHumanoidReport
            {
                MappedLabels = mappedLabels.ToArray(),
                MappedBones = mappedBones.ToArray(),
                MissingRequired = missing.ToArray(),
                UnmappedLabels = unmapped.ToArray(),
                BoneCount = rig.Count,
            };
        }

        /// <summary>
        /// The humanoid definition for a rig: the mapped bones plus the whole skeleton, so Unity sees
        /// the extra bones in the bind pose even though it will not animate them.
        /// </summary>
        public static HumanDescription BuildHumanDescription(AnnyRig rig, bool includeSkeleton = true)
        {
            AnnyHumanoidReport report = Describe(rig);

            List<HumanBone> bones = new List<HumanBone>(report.MappedLabels.Length);
            for (int i = 0; i < report.MappedLabels.Length; i++)
            {
                bones.Add(new HumanBone
                {
                    boneName = report.MappedLabels[i],
                    humanName = report.MappedBones[i].ToString(),
                    limit = new HumanLimit { useDefaultValues = true },
                });
            }

            List<SkeletonBone> skeleton = new List<SkeletonBone>();
            if (includeSkeleton)
            {
                for (int b = 0; b < rig.Count; b++)
                {
                    Transform bone = rig.Bones[b];
                    skeleton.Add(new SkeletonBone
                    {
                        name = rig.Labels[b],
                        position = bone.localPosition,
                        rotation = bone.localRotation,
                        scale = bone.localScale,
                    });
                }
            }

            return new HumanDescription
            {
                human = bones.ToArray(),
                skeleton = skeleton.ToArray(),
                armStretch = 0.05f,
                legStretch = 0.05f,
                upperArmTwist = 0.5f,
                lowerArmTwist = 0.5f,
                upperLegTwist = 0.5f,
                lowerLegTwist = 0.5f,
                feetSpacing = 0f,
                hasTranslationDoF = false,
            };
        }

        /// <summary>
        /// Builds a Unity humanoid avatar for a rig, using the hierarchy as it stands — this reads
        /// transforms and names and writes nothing back, so an avatar can be made at any time.
        /// </summary>
        public static Avatar Build(AnnyRig rig, string name = "anny_humanoid")
        {
            if (rig == null)
            {
                throw new System.ArgumentNullException(nameof(rig));
            }

            AnnyHumanoidReport report = Describe(rig);
            if (!report.Complete)
            {
                string[] names = new string[report.MissingRequired.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    names[i] = report.MissingRequired[i].ToString();
                }

                throw new System.InvalidOperationException(
                    "the rig has no bone for " + string.Join(", ", names));
            }

            Avatar avatar = AvatarBuilder.BuildHumanAvatar(
                rig.Root.gameObject, BuildHumanDescription(rig));
            if (avatar == null)
            {
                throw new System.InvalidOperationException("Unity returned no humanoid avatar for the rig");
            }

            avatar.name = name;
            return avatar;
        }

        private static KeyValuePair<string, HumanBodyBones>[] BuildTable()
        {
            List<KeyValuePair<string, HumanBodyBones>> table =
                new List<KeyValuePair<string, HumanBodyBones>>(64);

            // Hips is the anny rig's own root: it sits at the pelvis, at the same point as pelvis.L
            // and pelvis.R, which is what makes the untouched hierarchy a legal humanoid one.
            Pair(table, "root", HumanBodyBones.Hips);

            // The spine is numbered downwards: spine05 is the lowest, spine01 the highest, so the
            // humanoid slots are filled by height rather than by number. spine01 and spine02 stay
            // outside, and spine01 is also the parent of the clavicles and neck.
            Pair(table, "spine05", HumanBodyBones.Spine);
            Pair(table, "spine04", HumanBodyBones.Chest);
            Pair(table, "spine03", HumanBodyBones.UpperChest);
            Pair(table, "neck01", HumanBodyBones.Neck);
            Pair(table, "head", HumanBodyBones.Head);
            Pair(table, "eye.L", HumanBodyBones.LeftEye);
            Pair(table, "eye.R", HumanBodyBones.RightEye);
            Pair(table, "jaw", HumanBodyBones.Jaw);

            Side(table, "clavicle", HumanBodyBones.LeftShoulder, HumanBodyBones.RightShoulder);
            Side(table, "upperarm01", HumanBodyBones.LeftUpperArm, HumanBodyBones.RightUpperArm);
            Side(table, "lowerarm01", HumanBodyBones.LeftLowerArm, HumanBodyBones.RightLowerArm);
            Side(table, "wrist", HumanBodyBones.LeftHand, HumanBodyBones.RightHand);
            Side(table, "upperleg01", HumanBodyBones.LeftUpperLeg, HumanBodyBones.RightUpperLeg);
            Side(table, "lowerleg01", HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg);
            Side(table, "foot", HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot);
            Side(table, "toe1-1", HumanBodyBones.LeftToes, HumanBodyBones.RightToes);

            // Finger order is from geometry, not from the name: finger1's base is the nearest to the
            // wrist (0.042 against 0.089-0.100) and deviates 37.9 degrees from the middle-finger
            // axis, which makes it the thumb; finger5 deviates 24.5 degrees and is the pinky.
            Finger(table, 1, HumanBodyBones.LeftThumbProximal, HumanBodyBones.LeftThumbIntermediate, HumanBodyBones.LeftThumbDistal,
                HumanBodyBones.RightThumbProximal, HumanBodyBones.RightThumbIntermediate, HumanBodyBones.RightThumbDistal);
            Finger(table, 2, HumanBodyBones.LeftIndexProximal, HumanBodyBones.LeftIndexIntermediate, HumanBodyBones.LeftIndexDistal,
                HumanBodyBones.RightIndexProximal, HumanBodyBones.RightIndexIntermediate, HumanBodyBones.RightIndexDistal);
            Finger(table, 3, HumanBodyBones.LeftMiddleProximal, HumanBodyBones.LeftMiddleIntermediate, HumanBodyBones.LeftMiddleDistal,
                HumanBodyBones.RightMiddleProximal, HumanBodyBones.RightMiddleIntermediate, HumanBodyBones.RightMiddleDistal);
            Finger(table, 4, HumanBodyBones.LeftRingProximal, HumanBodyBones.LeftRingIntermediate, HumanBodyBones.LeftRingDistal,
                HumanBodyBones.RightRingProximal, HumanBodyBones.RightRingIntermediate, HumanBodyBones.RightRingDistal);
            Finger(table, 5, HumanBodyBones.LeftLittleProximal, HumanBodyBones.LeftLittleIntermediate, HumanBodyBones.LeftLittleDistal,
                HumanBodyBones.RightLittleProximal, HumanBodyBones.RightLittleIntermediate, HumanBodyBones.RightLittleDistal);

            return table.ToArray();
        }

        private static void Pair(
            List<KeyValuePair<string, HumanBodyBones>> table, string label, HumanBodyBones bone)
        {
            table.Add(new KeyValuePair<string, HumanBodyBones>(label, bone));
        }

        private static void Side(
            List<KeyValuePair<string, HumanBodyBones>> table,
            string stem, HumanBodyBones left, HumanBodyBones right)
        {
            Pair(table, stem + ".L", left);
            Pair(table, stem + ".R", right);
        }

        private static void Finger(
            List<KeyValuePair<string, HumanBodyBones>> table, int finger,
            HumanBodyBones leftProximal, HumanBodyBones leftIntermediate, HumanBodyBones leftDistal,
            HumanBodyBones rightProximal, HumanBodyBones rightIntermediate, HumanBodyBones rightDistal)
        {
            string stem = "finger" + finger + "-";
            Pair(table, stem + "1.L", leftProximal);
            Pair(table, stem + "2.L", leftIntermediate);
            Pair(table, stem + "3.L", leftDistal);
            Pair(table, stem + "1.R", rightProximal);
            Pair(table, stem + "2.R", rightIntermediate);
            Pair(table, stem + "3.R", rightDistal);
        }
    }
}
