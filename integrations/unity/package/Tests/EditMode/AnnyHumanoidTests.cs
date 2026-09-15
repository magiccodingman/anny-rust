using System;
using System.Collections.Generic;
using Anny;
using NUnit.Framework;
using UnityEngine;

namespace Anny.Tests
{
    /// <summary>
    /// Validates the humanoid mapping against the real rig and against Unity itself. The mapping
    /// table is a claim about the anny rig's bone names and geometry, so it is checked the same way
    /// the rest of the integration is: against the native description, not against a restatement of
    /// the table.
    /// </summary>
    public sealed class AnnyHumanoidTests
    {
        private AnnyModel wide;
        private AnnyModelF32 runtime;
        private AnnyOutputF32 output;
        private AnnyRig rig;
        private Avatar avatar;

        [SetUp]
        public void SetUp()
        {
            if (!AnnyTestModels.Available)
            {
                Assert.Ignore("no prepared model available");
            }

            wide = AnnyTestModels.Load();
            runtime = wide.ToSinglePrecision();
            output = runtime.Evaluate(new AnnyParameters());
            rig = AnnySkeleton.Build(wide.Description, output, null);
            avatar = null;
        }

        [TearDown]
        public void TearDown()
        {
            if (avatar != null)
            {
                UnityEngine.Object.DestroyImmediate(avatar);
            }

            if (rig != null && rig.Root != null)
            {
                UnityEngine.Object.DestroyImmediate(rig.Root.gameObject);
            }

            if (output != null)
            {
                output.Dispose();
            }

            if (runtime != null)
            {
                runtime.Dispose();
            }

            if (wide != null)
            {
                wide.Dispose();
            }
        }

        [Test]
        public void MappingFillsEveryRequiredSlot()
        {
            // The table targets the anny rig. If the prepared model ever ships a different rig this
            // is the test that says so, before a humanoid avatar quietly comes out wrong.
            Assert.AreEqual(104, rig.Count, "the anny rig carries 104 bones");

            AnnyHumanoidReport report = AnnyHumanoid.Describe(rig);
            if (!report.Complete)
            {
                string[] missing = new string[report.MissingRequired.Length];
                for (int i = 0; i < missing.Length; i++)
                {
                    missing[i] = report.MissingRequired[i].ToString();
                }

                Assert.Fail("required slots without a bone: " + string.Join(", ", missing));
            }

            Assert.AreEqual(54, report.MappedLabels.Length, "mapped bones (24 body slots and 30 finger bones)");
            Assert.AreEqual(50, report.UnmappedLabels.Length, "bones outside the humanoid definition");
            Assert.AreEqual(rig.Count, report.MappedLabels.Length + report.UnmappedLabels.Length);

            HashSet<HumanBodyBones> slots = new HashSet<HumanBodyBones>();
            for (int i = 0; i < report.MappedBones.Length; i++)
            {
                Assert.IsTrue(slots.Add(report.MappedBones[i]),
                    "slot filled twice: " + report.MappedBones[i]);
                Assert.GreaterOrEqual(rig.IndexOf(report.MappedLabels[i]), 0,
                    "mapped label is not in the rig: " + report.MappedLabels[i]);
            }
        }

        [Test]
        public void HipsSitsAtThePelvis()
        {
            // Why the untouched hierarchy is a legal humanoid one: the anny root bone is at the
            // pelvis, so it can be Hips with both legs and the spine beneath it.
            Vector3 hips = rig.Position("root");
            Assert.Less((hips - rig.Position("pelvis.L")).magnitude, 1e-5f, "root and left pelvis");
            Assert.Less((hips - rig.Position("pelvis.R")).magnitude, 1e-5f, "root and right pelvis");

            int root = rig.IndexOf("root");
            Assert.AreEqual(root, rig.Parents[rig.IndexOf("pelvis.L")], "left leg hangs from Hips");
            Assert.AreEqual(root, rig.Parents[rig.IndexOf("pelvis.R")], "right leg hangs from Hips");
            Assert.AreEqual(root, rig.Parents[rig.IndexOf("spine05")], "the spine hangs from Hips");
        }

        [Test]
        public void SpineSlotsFollowTheHeightNotTheNumber()
        {
            // The rig numbers its spine downwards: spine05 is the lowest and spine01 the highest,
            // which is also the parent of the clavicles and the neck.
            float spine = rig.Position("spine05").y;
            float chest = rig.Position("spine04").y;
            float upperChest = rig.Position("spine03").y;

            Assert.Less(spine, chest, "Spine sits below Chest");
            Assert.Less(chest, upperChest, "Chest sits below UpperChest");
            Assert.Less(upperChest, rig.Position("neck01").y, "UpperChest sits below the neck");
            Assert.Less(rig.Position("neck01").y, rig.Position("head").y, "the neck sits below the head");

            Assert.AreEqual(HumanBodyBones.Spine, AnnyHumanoid.SlotOf("spine05"));
            Assert.AreEqual(HumanBodyBones.Chest, AnnyHumanoid.SlotOf("spine04"));
            Assert.AreEqual(HumanBodyBones.UpperChest, AnnyHumanoid.SlotOf("spine03"));
            Assert.AreEqual(HumanBodyBones.LastBone, AnnyHumanoid.SlotOf("spine01"),
                "spine01 is above UpperChest and has no slot");
        }

        [Test]
        public void FingerOrderMatchesTheHandGeometry()
        {
            // finger1 is the thumb because its base is the nearest to the wrist and the furthest
            // from the middle-finger axis; finger5 is the pinky.
            Assert.AreEqual(HumanBodyBones.LeftThumbProximal, AnnyHumanoid.SlotOf("finger1-1.L"));
            Assert.AreEqual(HumanBodyBones.RightThumbDistal, AnnyHumanoid.SlotOf("finger1-3.R"));
            Assert.AreEqual(HumanBodyBones.LeftIndexProximal, AnnyHumanoid.SlotOf("finger2-1.L"));
            Assert.AreEqual(HumanBodyBones.LeftLittleDistal, AnnyHumanoid.SlotOf("finger5-3.L"));

            float thumb = Vector3.Distance(rig.Position("wrist.L"), rig.Position("finger1-1.L"));
            for (int finger = 2; finger <= 5; finger++)
            {
                float other = Vector3.Distance(rig.Position("wrist.L"), rig.Position("finger" + finger + "-1.L"));
                Assert.Greater(other, thumb * 1.5f,
                    "finger" + finger + " starts further from the wrist than the thumb does");
            }
        }

        [Test]
        public void HumanDescriptionCoversTheWholeSkeleton()
        {
            AnnyHumanoidReport report = AnnyHumanoid.Describe(rig);
            HumanDescription description = AnnyHumanoid.BuildHumanDescription(rig);

            Assert.AreEqual(report.MappedLabels.Length, description.human.Length, "mapped bones");
            Assert.AreEqual(rig.Count, description.skeleton.Length, "every bone is in the skeleton");

            for (int i = 0; i < description.human.Length; i++)
            {
                HumanBone bone = description.human[i];
                Assert.GreaterOrEqual(rig.IndexOf(bone.boneName), 0, "unknown bone " + bone.boneName);
                Assert.IsTrue(Enum.IsDefined(typeof(HumanBodyBones), bone.humanName),
                    "not a humanoid slot: " + bone.humanName);
                Assert.IsTrue(bone.limit.useDefaultValues, "the rig's own limits are Unity's defaults");
            }

            for (int i = 0; i < description.skeleton.Length; i++)
            {
                SkeletonBone bone = description.skeleton[i];
                Assert.AreEqual(rig.Labels[i], bone.name, "skeleton order follows the rig");
                Assert.Less((bone.position - rig.Bones[i].localPosition).magnitude, 1e-6f,
                    "bind position of " + bone.name);
                Assert.Less(Quaternion.Angle(bone.rotation, rig.Bones[i].localRotation), 1e-4f,
                    "bind rotation of " + bone.name);
            }
        }

        [Test]
        public void UnityAcceptsTheAvatar()
        {
            avatar = AnnyHumanoid.Build(rig);
            AnnyHumanoidReport report = AnnyHumanoid.Describe(rig);
            Debug.Log("anny humanoid: " + report + "; valid=" + avatar.isValid
                + " human=" + avatar.isHuman);

            Assert.IsTrue(avatar.isHuman, "Unity must read the rig as humanoid");
            Assert.IsTrue(avatar.isValid, "Unity must accept the humanoid definition");

            Animator animator = rig.Root.gameObject.AddComponent<Animator>();
            animator.avatar = avatar;
            Assert.IsTrue(animator.isHuman, "the animator must see a humanoid avatar");
            Assert.IsTrue(animator.avatar.isValid, "an assigned avatar stays valid");
        }

        [Test]
        public void BuildingAnAvatarLeavesTheHierarchyAlone()
        {
            // The hierarchy is not the humanoid's to change: pose application goes through global
            // bone matrices, so an avatar must leave every parent and local transform as it found it.
            int[] parentsBefore = (int[])rig.Parents.Clone();
            Vector3[] positionsBefore = new Vector3[rig.Count];
            Quaternion[] rotationsBefore = new Quaternion[rig.Count];
            for (int b = 0; b < rig.Count; b++)
            {
                positionsBefore[b] = rig.Bones[b].localPosition;
                rotationsBefore[b] = rig.Bones[b].localRotation;
            }

            AnnyHumanoid.BuildHumanDescription(rig);
            avatar = AnnyHumanoid.Build(rig);

            for (int b = 0; b < rig.Count; b++)
            {
                Assert.AreEqual(parentsBefore[b], rig.Parents[b], "parent of " + rig.Labels[b]);
                Assert.Less((positionsBefore[b] - rig.Bones[b].localPosition).magnitude, 1e-7f,
                    "local position of " + rig.Labels[b]);
                Assert.Less(Quaternion.Angle(rotationsBefore[b], rig.Bones[b].localRotation), 1e-5f,
                    "local rotation of " + rig.Labels[b]);
            }
        }

        [Test]
        public void MissingBonesAreReportedNotGuessed()
        {
            GameObject host = new GameObject("anny_humanoid_gap");
            try
            {
                string[] labels = { "root", "spine05", "head" };
                Transform[] bones = new Transform[labels.Length];
                for (int b = 0; b < labels.Length; b++)
                {
                    bones[b] = new GameObject(labels[b]).transform;
                    bones[b].SetParent(b == 0 ? host.transform : bones[0], false);
                }

                AnnyRig partial = new AnnyRig
                {
                    Root = host.transform,
                    Labels = labels,
                    Bones = bones,
                    Parents = new[] { -1, 0, 1 },
                };

                AnnyHumanoidReport report = AnnyHumanoid.Describe(partial);
                Assert.IsFalse(report.Complete, "a three-bone rig cannot be humanoid");
                Assert.AreEqual(12, report.MissingRequired.Length, "15 required slots, 3 of them filled");
                CollectionAssert.Contains(report.MissingRequired, HumanBodyBones.LeftHand);
                CollectionAssert.Contains(report.MissingRequired, HumanBodyBones.RightFoot);

                InvalidOperationException error =
                    Assert.Throws<InvalidOperationException>(() => AnnyHumanoid.Build(partial));
                StringAssert.Contains("LeftHand", error.Message);

                Assert.AreEqual(3, report.MappedLabels.Length, "only root, spine05 and head are mapped");
                Assert.AreEqual(0, report.UnmappedLabels.Length, "nothing is left over");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }
    }
}