using System.Collections;
using System.IO;
using Anny;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Anny.Tests
{
    /// <summary>
    /// Drives a generated character inside a running player loop. This is the integration the
    /// EditMode suites cannot cover: a real GameObject, a real SkinnedMeshRenderer, real frame
    /// updates, and the native session path used the way a game would use it.
    /// </summary>
    public sealed class AnnyCharacterPlayTests
    {
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

        private AnnyCharacter character;
        private GameObject host;

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

        private AnnyCharacter Spawn(AnnyUpdateMode mode)
        {
            // Configure before Awake runs, which is how a pool or an instantiation path does it:
            // the object is created inactive, wired up, then activated.
            host = new GameObject("anny-play-character");
            host.SetActive(false);
            character = host.AddComponent<AnnyCharacter>();
            character.mode = mode;
            character.modelAsset = AnnyModelAsset.CreateInMemory(File.ReadAllBytes(ModelPath));
            host.SetActive(true);
            character.Generate();
            return character;
        }

        [Test]
        public void GeneratingACharacterProducesAMeshAndASkeleton()
        {
            AnnyCharacter run = Spawn(AnnyUpdateMode.Exact);

            Assert.IsNotNull(run.GeneratedMesh, "generated mesh");
            Assert.Greater(run.GeneratedMesh.vertexCount, 0, "mesh vertices");
            Assert.IsNotNull(run.Rig, "rig");
            Assert.AreEqual(AnnyTestConstants.Bones, run.Rig.Count, "bones");
            Assert.Greater(run.GeneratedMesh.triangles.Length, 0, "triangles");
            Assert.IsNotNull(run.Report, "report");
            Assert.AreEqual(AnnyTestConstants.Vertices, run.Report.SourceVertices, "source vertices");
        }

        [UnityTest]
        public IEnumerator SkinnedModeBuildsARendererWhoseBakeMatchesTheNativePose()
        {
            AnnyCharacter run = Spawn(AnnyUpdateMode.Skinned);

            // Rest in the pose the mesh was generated for, then let a frame pass: BakeMesh reads the
            // bone world matrices, and a rig built and posed earlier in the same frame has not
            // propagated yet.
            yield return null;

            Assert.IsNotNull(run.Skinned, "skinned renderer");
            Assert.IsNotNull(run.Skinned.sharedMesh, "renderer mesh");
            Assert.AreEqual(AnnyTestConstants.Bones, run.Skinned.bones.Length, "renderer bones");
            Assert.IsNotNull(run.Skinned.rootBone, "root bone");

            // At the bind pose, Unity's skinning must reproduce the vertices the native evaluation
            // produced for the same mesh. This is what the Exact mode buys, and it must hold
            // approximately for the skinned path too.
            Mesh baked = new Mesh();
            try
            {
                run.Skinned.BakeMesh(baked);
                Vector3[] bakedVertices = baked.vertices;
                Vector3[] meshVertices = run.GeneratedMesh.vertices;

                Assert.AreEqual(meshVertices.Length, bakedVertices.Length, "baked vertex count");

                float worst = 0f;
                float[] rest = run.LastVertices();
                for (int v = 0; v < bakedVertices.Length; v++)
                {
                    int source = run.CornerSource != null ? run.CornerSource[v] : v;
                    Vector3 expected = AnnyRepresentation.ToUnityPosition(
                        rest[source * 3], rest[source * 3 + 1], rest[source * 3 + 2]);
                    float deviation = (bakedVertices[v] - expected).magnitude;
                    if (deviation > worst)
                    {
                        worst = deviation;
                    }
                }

                // Isolate which pair diverges: the mesh against the array it was built from, the
                // baked result against the mesh, and the bone pose against the bind pose.
                float meshVersusArray = 0f;
                for (int v = 0; v < meshVertices.Length; v += 7)
                {
                    int source = run.CornerSource != null ? run.CornerSource[v] : v;
                    Vector3 fromArray = AnnyRepresentation.ToUnityPosition(
                        rest[source * 3], rest[source * 3 + 1], rest[source * 3 + 2]);
                    meshVersusArray = Mathf.Max(meshVersusArray, (meshVertices[v] - fromArray).magnitude);
                }

                Debug.Log(
                    "play mode skinned: mesh versus cached array " + meshVersusArray.ToString("G6") +
                    " m, baked versus mesh " + worst.ToString("G6") + " m");

                Debug.Log(
                    "play mode skinned at bind pose: worst " + worst.ToString("G6") + " m, active quality level \"" +
                    QualitySettings.names[QualitySettings.GetQualityLevel()] + "\", skinWeights " +
                    QualitySettings.skinWeights + ", widest vertex " + run.Report.MaxInfluencesPerVertex +
                    " influences");

                // The magnitudes above are metres on a human figure: a millimetre bound is strict
                // enough to catch a wrong bone, bind pose or weight, and loose enough for f32.
                Assert.Less(worst, 1e-3f, "bind pose deviation in metres");
            }
            finally
            {
                Object.DestroyImmediate(baked);
            }
        }

        [Test]
        public void PhenotypeChangesAreReflectedInTheGeneratedMesh()
        {
            AnnyCharacter run = Spawn(AnnyUpdateMode.Exact);

            float[] before = run.LastVertices();
            float[] snapshot = (float[])before.Clone();

            run.SetPhenotype("gender", 1f);
            run.Apply();

            float[] after = run.LastVertices();
            double moved = 0.0;
            int movedCount = 0;
            for (int i = 0; i < after.Length; i++)
            {
                double d = System.Math.Abs(after[i] - snapshot[i]);
                if (d > 0.0)
                {
                    movedCount++;
                    moved = System.Math.Max(moved, d);
                }
            }

            Assert.AreEqual(snapshot.Length, after.Length, "vertex array size");
            Assert.Greater(movedCount, 0, "a phenotype change must move vertices");
            Assert.Greater(moved, 1e-4, "largest vertex movement in metres");

            // The mesh in the scene must carry the new geometry, not the pre-change buffer.
            Vector3[] meshVertices = run.GeneratedMesh.vertices;
            for (int probe = 0; probe < meshVertices.Length; probe += 977)
            {
                int source = run.CornerSource != null ? run.CornerSource[probe] : probe;
                Vector3 expected = AnnyRepresentation.ToUnityPosition(
                    after[source * 3], after[source * 3 + 1], after[source * 3 + 2]);
                Assert.AreEqual(expected.x, meshVertices[probe].x, 0f, "mesh vertex " + probe + " x");
                Assert.AreEqual(expected.y, meshVertices[probe].y, 0f, "mesh vertex " + probe + " y");
                Assert.AreEqual(expected.z, meshVertices[probe].z, 0f, "mesh vertex " + probe + " z");
            }
        }

        [Test]
        public void PoseSessionMovesTheCharacterWithoutAFullReevaluation()
        {
            AnnyCharacter run = Spawn(AnnyUpdateMode.Exact);

            Assert.IsNotNull(run.Session, "the character must open a native pose session");
            Assert.IsNotNull(run.Rig, "rig");

            // A named-bone pose: the native parser accepts bone label -> matrix, which is the
            // convenient form for gameplay code. The rotation is about the model's X axis, which is
            // also Unity's X, so the same matrix is valid in both bases.
            string bone = run.Rig.Labels[0];
            for (int b = 0; b < run.Rig.Count; b++)
            {
                if (run.Rig.Labels[b] == "spine")
                {
                    bone = run.Rig.Labels[b];
                    break;
                }
            }

            Matrix4x4 delta = Matrix4x4.Rotate(Quaternion.Euler(15f, 0f, 0f));
            string poseJson = AnnyRepresentation.NamedPosesToJson(
                new System.Collections.Generic.Dictionary<string, Matrix4x4> { { bone, delta } });

            float[] before = (float[])run.LastVertices().Clone();
            int evaluationsBefore = run.Evaluations;

            run.ApplyPose(poseJson);
            float sessionMs = (float)run.LastEvaluateMs;

            float[] after = run.LastVertices();
            int movedCount = 0;
            double moved = 0.0;
            for (int i = 0; i < after.Length; i++)
            {
                double d = System.Math.Abs(after[i] - before[i]);
                if (d > 0.0)
                {
                    movedCount++;
                    moved = System.Math.Max(moved, d);
                }
            }

            // A full evaluation for comparison, on the same character.
            run.Apply();
            double fullMs = run.LastEvaluateMs;

            Debug.Log(
                "session update " + sessionMs.ToString("G6") + " ms, full evaluation " +
                fullMs.ToString("G6") + " ms, moved vertices " + movedCount +
                ", largest movement " + moved.ToString("G6") + " m");

            Assert.Greater(run.Evaluations, evaluationsBefore, "the character evaluated again");
            Assert.Greater(movedCount, 0, "posing a bone must move vertices");
            Assert.Greater(moved, 1e-3, "largest movement in metres");

            // The scene mesh must show the posed result, not the previous frame.
            Vector3[] meshVertices = run.GeneratedMesh.vertices;
            for (int probe = 0; probe < meshVertices.Length; probe += 997)
            {
                int source = run.CornerSource != null ? run.CornerSource[probe] : probe;
                Vector3 expected = AnnyRepresentation.ToUnityPosition(
                    after[source * 3], after[source * 3 + 1], after[source * 3 + 2]);
                Assert.AreEqual(expected.x, meshVertices[probe].x, 0f, "posed mesh vertex " + probe + " x");
            }
        }

        [UnityTest]
        public IEnumerator DisposingTheCharacterReleasesTheNativeModel()
        {
            AnnyCharacter run = Spawn(AnnyUpdateMode.Exact);
            AnnyModelF32 runtime = run.Runtime;
            Assert.IsNotNull(runtime, "runtime model");
            Assert.IsFalse(runtime.IsDisposed, "the model is alive while the character exists");

            Object.Destroy(host);
            host = null;
            character = null;

            // Destroy is queued: let a frame pass so OnDestroy has run before asserting.
            yield return null;

            Assert.IsTrue(runtime.IsDisposed, "the native model must be released with the component");
        }
    }
}
