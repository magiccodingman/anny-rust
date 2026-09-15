using System.IO;
using Anny;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Anny.Tests
{
    /// <summary>
    /// A preset is only worth having if applying one reproduces the character it was captured from.
    /// These tests drive real characters in the editor, so they cover exactly what the inspector
    /// does: configure, capture, apply, regenerate, compare.
    /// </summary>
    public sealed class AnnyPresetTests
    {
        private const string Folder = "Assets/Anny/TestArtifacts";
        private const string AssetPath = Folder + "/TestPreset.asset";

        private static byte[] cachedPayload;

        private GameObject hostA;
        private GameObject hostB;

        [SetUp]
        public void SetUp()
        {
            if (!AnnyTestModels.Available)
            {
                Assert.Ignore("no prepared model available");
            }
        }

        [TearDown]
        public void TearDown()
        {
            DestroyHost(hostA);
            DestroyHost(hostB);
            AssetDatabase.DeleteAsset(AssetPath);
        }

        private static void DestroyHost(GameObject host)
        {
            if (host != null)
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        private static byte[] Payload()
        {
            if (cachedPayload == null)
            {
                cachedPayload = File.ReadAllBytes(AnnyTestModels.PreparedModelPath);
            }

            return cachedPayload;
        }

        private AnnyCharacter NewCharacter(out GameObject host, string name, float gender)
        {
            host = new GameObject(name);
            AnnyCharacter character = host.AddComponent<AnnyCharacter>();
            character.generateOnAwake = false;
            character.modelAsset = AnnyModelAsset.CreateInMemory(Payload());
            character.phenotype.Add(new AnnySliderEntry { name = "gender", value = gender });
            return character;
        }

        [Test]
        public void ApplyingAPresetReproducesTheCapturedGeometryExactly()
        {
            AnnyCharacter captured = NewCharacter(out hostA, "preset-source", 1f);
            captured.Generate();

            AnnyPreset preset = ScriptableObject.CreateInstance<AnnyPreset>();
            preset.CaptureFrom(captured);
            Assert.IsTrue(preset.Matches(captured), "a captured preset should describe its source");

            AnnyCharacter applied = NewCharacter(out hostB, "preset-target", 0f);
            Assert.IsFalse(preset.Matches(applied), "the targets must start out different");
            preset.ApplyTo(applied);
            Assert.IsTrue(preset.Matches(applied), "applying a preset should configure the target");
            applied.Generate();

            Vector3[] fromSource = captured.GeneratedMesh.vertices;
            Vector3[] fromTarget = applied.GeneratedMesh.vertices;
            Assert.AreEqual(fromSource.Length, fromTarget.Length, "vertex count");

            float worst = 0f;
            for (int i = 0; i < fromSource.Length; i++)
            {
                worst = Mathf.Max(worst, (fromSource[i] - fromTarget[i]).magnitude);
            }

            Assert.AreEqual(0f, worst, "a preset round trip must not move a single vertex");
            Assert.AreEqual(
                captured.Report.MaxInfluencesPerVertex,
                applied.Report.MaxInfluencesPerVertex,
                "influence width");
            Assert.AreEqual(
                captured.GeneratedMesh.triangles.Length,
                applied.GeneratedMesh.triangles.Length,
                "triangle count");
            Assert.AreEqual(captured.GeneratedMesh.bindposes.Length, applied.GeneratedMesh.bindposes.Length, "bind poses");
        }

        [Test]
        public void PresetSurvivesAssetSerialization()
        {
            AnnyCharacter character = NewCharacter(out hostA, "preset-serialise", 0.25f);
            character.mode = AnnyUpdateMode.Exact;
            character.meshOptions.BuildBlendshapes = true;
            character.meshOptions.MaxBoneInfluences = 6;
            character.Generate();

            if (!AssetDatabase.IsValidFolder(Folder))
            {
                if (!AssetDatabase.IsValidFolder("Assets/Anny"))
                {
                    AssetDatabase.CreateFolder("Assets", "Anny");
                }

                AssetDatabase.CreateFolder("Assets/Anny", "TestArtifacts");
            }

            AnnyPreset preset = ScriptableObject.CreateInstance<AnnyPreset>();
            preset.CaptureFrom(character);
            AssetDatabase.CreateAsset(preset, AssetPath);

            AnnyPreset loaded = AssetDatabase.LoadAssetAtPath<AnnyPreset>(AssetPath);
            Assert.IsNotNull(loaded, "the preset should load back as a project asset");
            Assert.AreEqual(AnnyUpdateMode.Exact, loaded.mode, "mode");
            Assert.AreEqual(6, loaded.maxBoneInfluences, "influence cap");
            Assert.IsTrue(loaded.buildBlendshapes, "blendshape flag");
            Assert.AreEqual(1, loaded.phenotype.Count, "slider count");
            Assert.AreEqual("gender", loaded.phenotype[0].name, "slider label");
            Assert.AreEqual(0.25f, loaded.phenotype[0].value, "slider value");
            Assert.IsTrue(loaded.Matches(character), "the reloaded preset should still describe its source");
        }

        [Test]
        public void GenerationRunsInTheEditorOutsidePlayMode()
        {
            // The inspector drives Generate() from edit mode, so that path has to work there.
            Assert.IsFalse(Application.isPlaying, "this test asserts the editor path");

            AnnyCharacter character = NewCharacter(out hostA, "preset-editmode", 0.5f);
            character.Generate();

            Assert.IsNotNull(character.GeneratedMesh, "generation must produce a mesh in the editor");
            Assert.Greater(character.GeneratedMesh.vertexCount, 0);
            Assert.AreEqual(AnnyTestModels.ExpectedBones, character.Rig.Count, "bone count");
            Assert.IsNotNull(character.Skinned, "the skinned renderer should be wired up");
        }
    }
}
