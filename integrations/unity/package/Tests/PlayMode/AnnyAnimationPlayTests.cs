using System.Collections;
using System.Collections.Generic;
using System.IO;
using Anny;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Anny.Tests
{
    /// <summary>
    /// The baked-clip claim is that Unity's animation system can reproduce a native pose, so the tests
    /// bake a clip from real evaluations and then sample it, comparing every bone against the same
    /// native pose. A clip that merely exists, or that has the right length, proves nothing.
    /// </summary>
    public sealed class AnnyAnimationPlayTests
    {
        private const float FrameRate = 30f;

        private GameObject host;
        private AnnyCharacter character;

        private static string ModelPath
        {
            get
            {
                string fromEnvironment = System.Environment.GetEnvironmentVariable("ANNY_MODEL");
                if (!string.IsNullOrEmpty(fromEnvironment) && File.Exists(fromEnvironment))
                {
                    return fromEnvironment;
                }

                DirectoryInfo project = new DirectoryInfo(Application.dataPath).Parent;
                string candidate = Path.GetFullPath(
                    Path.Combine(project.FullName, "..", "..", "..", "output", "ci-model.safetensors"));
                return File.Exists(candidate) ? candidate : null;
            }
        }

        [SetUp]
        public void SetUp()
        {
            if (ModelPath == null)
            {
                Assert.Ignore("prepared model not found; set ANNY_MODEL");
            }
        }

        [TearDown]
        public void TearDown()
        {
            if (host != null)
            {
                Object.Destroy(host);
                host = null;
                character = null;
            }
        }

        private void Spawn(AnnyUpdateMode mode)
        {
            host = new GameObject("anny-animation");
            host.SetActive(false);
            character = host.AddComponent<AnnyCharacter>();
            character.generateOnAwake = false;
            character.mode = mode;
            character.modelAsset = AnnyModelAsset.CreateInMemory(File.ReadAllBytes(ModelPath));
            host.SetActive(true);
            character.Generate();
        }

        private static Vector3[] Positions(AnnyRig rig)
        {
            Vector3[] positions = new Vector3[rig.Count];
            for (int bone = 0; bone < rig.Count; bone++)
            {
                positions[bone] = rig.Bones[bone].position;
            }

            return positions;
        }

        private static string BoneNamed(AnnyRig rig, string label)
        {
            for (int bone = 0; bone < rig.Count; bone++)
            {
                if (rig.Labels[bone] == label)
                {
                    return label;
                }
            }

            return rig.Labels[0];
        }

        [Test]
        public void ABakedPoseSequenceReproducesTheNativePoses()
        {
            Spawn(AnnyUpdateMode.Exact);

            string spine = BoneNamed(character.Rig, "spine");
            List<Dictionary<string, Matrix4x4>> keyframes = new List<Dictionary<string, Matrix4x4>>
            {
                new Dictionary<string, Matrix4x4>(),
                new Dictionary<string, Matrix4x4>
                {
                    { spine, Matrix4x4.Rotate(Quaternion.Euler(20f, 5f, 0f)) },
                },
                new Dictionary<string, Matrix4x4>
                {
                    { spine, Matrix4x4.Rotate(Quaternion.Euler(-15f, 0f, 10f)) },
                },
            };

            // Native reference: the same poses, driven through the optimized session path.
            Vector3[][] expected = new Vector3[keyframes.Count][];
            for (int frame = 0; frame < keyframes.Count; frame++)
            {
                character.ApplyPose(AnnyRepresentation.NamedPosesToJson(keyframes[frame]));
                expected[frame] = Positions(character.Rig);
            }

            AnimationClip clip = AnnyClipBaker.BakePoseSequence(character, keyframes, FrameRate);
            Assert.IsNotNull(clip, "the bakery should return a clip");
            Assert.IsFalse(clip.empty, "the clip should carry curves");
            // Three keyframes one frame apart span two frames of time, so the clip is two frames long.
            Assert.AreEqual((keyframes.Count - 1) / FrameRate, clip.length, 1e-4f, "clip length in seconds");

            float worst = Sample(character, clip, expected);
            Debug.Log(string.Format("ANNY-CLIP-POSE bones={0} frames={1} worst={2:0.########}", character.Rig.Count, keyframes.Count, worst));
            Assert.Less(worst, 1e-3f, "sampling the clip should reproduce the native pose");
        }

        [Test]
        public void ABakedPhenotypeSequenceReproducesTheNativePoses()
        {
            Spawn(AnnyUpdateMode.Exact);

            List<IList<AnnySliderEntry>> keyframes = new List<IList<AnnySliderEntry>>
            {
                new List<AnnySliderEntry> { new AnnySliderEntry { name = "gender", value = 0f } },
                new List<AnnySliderEntry> { new AnnySliderEntry { name = "gender", value = 1f } },
            };

            Vector3[][] expected = new Vector3[keyframes.Count][];
            for (int frame = 0; frame < keyframes.Count; frame++)
            {
                IList<AnnySliderEntry> entries = keyframes[frame];
                for (int i = 0; i < entries.Count; i++)
                {
                    character.SetPhenotype(entries[i].name, entries[i].value);
                }

                character.Apply();
                expected[frame] = Positions(character.Rig);
            }

            AnimationClip clip = AnnyClipBaker.BakePhenotypeSequence(character, keyframes, FrameRate);
            Assert.IsNotNull(clip);
            Assert.IsFalse(clip.empty, "the clip should carry curves");

            float worst = Sample(character, clip, expected);
            Debug.Log(string.Format("ANNY-CLIP-PHENOTYPE bones={0} frames={1} worst={2:0.########}", character.Rig.Count, keyframes.Count, worst));
            Assert.Less(worst, 1e-3f, "sampling the clip should reproduce the native pose");
        }

        /// <summary>Samples the clip frame by frame and returns the worst bone disagreement, in metres.</summary>
        private static float Sample(AnnyCharacter animated, AnimationClip clip, Vector3[][] expected)
        {
            float worst = 0f;
            for (int frame = 0; frame < expected.Length; frame++)
            {
                clip.SampleAnimation(animated.Rig.Root.gameObject, frame / FrameRate);
                Vector3[] sampled = Positions(animated.Rig);
                for (int bone = 0; bone < sampled.Length; bone++)
                {
                    worst = Mathf.Max(worst, (sampled[bone] - expected[frame][bone]).magnitude);
                }
            }

            return worst;
        }
    }
}
