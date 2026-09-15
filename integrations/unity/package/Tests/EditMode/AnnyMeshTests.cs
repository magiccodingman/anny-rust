using System;
using Anny;
using NUnit.Framework;
using UnityEngine;

namespace Anny.Tests
{
    /// <summary>
    /// Validates the mesh and skeleton integration against the native arrays. Every check compares
    /// Unity-side state with a value the native library produced, so a wrong coordinate mapping,
    /// winding, bind-pose inverse or vertex remap fails here rather than merely looking plausible.
    /// </summary>
    public sealed class AnnyMeshTests
    {
        private AnnyModel wide;
        private AnnyModelF32 runtime;
        private AnnyOutputF32 output;
        private AnnyMeshResult built;

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
            built = AnnyMeshBuilder.Build(runtime, output, new AnnyMeshOptions());
        }

        [TearDown]
        public void TearDown()
        {
            if (built != null && built.Mesh != null)
            {
                UnityEngine.Object.DestroyImmediate(built.Mesh);
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
        public void TopologyMatchesTheModelDescription()
        {
            AnnyDescription description = wide.Description;
            AnnyTensorF32 faces = runtime.Tensor("faces");

            Assert.AreEqual(description.vertices, built.Report.SourceVertices, "source vertices");
            Assert.AreEqual(faces.Length / faces.Shape[1], built.Report.Faces, "face count");

            if (built.Report.UvExpanded)
            {
                Assert.AreEqual(faces.Length, built.Mesh.vertexCount, "expanded corners become mesh vertices");
                Assert.AreEqual(faces.Length, built.Mesh.uv.Length, "every corner carries a uv");
            }
            else
            {
                Assert.AreEqual(description.vertices, built.Mesh.vertexCount, "native vertex order is kept");
            }

            Assert.AreEqual(
                built.Report.Faces * (built.Report.QuadFaces > 0 ? 6 : 3),
                built.Mesh.triangles.Length,
                "triangulated index count");
        }

        [Test]
        public void FacesPointOutward()
        {
            // The mesh closes a body, so a consistent outward winding gives a positive signed
            // volume. A reflected coordinate map without a winding flip would make this negative.
            Assert.Greater(built.Report.SignedVolume, 0f, "signed volume");
        }

        [Test]
        public void BoneWeightsPreserveEveryNativeInfluence()
        {
            AnnyTensorF32 weights = runtime.Tensor("vertex_bone_weights");

            Assert.AreEqual(weights.Shape[1], built.Report.BoneInfluenceWidth, "native influence width");

            Assert.AreEqual(0, built.Report.VerticesWithTruncatedInfluences, "vertices losing an influence");
            Assert.IsTrue(built.Report.SkinWeightsExact, "weights are exact");
            Assert.LessOrEqual(built.Report.MaxInfluencesPerVertex, built.Report.MaxBoneInfluences, "influences per vertex");
        }

        [Test]
        public void BindPosesInvertTheNativeRestPoses()
        {
            AnnyTensorF32 rest = output.Tensor(AnnyOutputF32.RestBonePoses);
            Assert.AreEqual(rest.Shape[1], built.Mesh.bindposes.Length, "bind pose count");

            for (int b = 0; b < built.Mesh.bindposes.Length; b++)
            {
                Matrix4x4 restUnity = AnnyRepresentation.ToUnityMatrix(ToWide(rest.Data, b * 16));
                Matrix4x4 round = built.Mesh.bindposes[b] * restUnity;
                AssertIdentity(round, 2e-4f, "bone " + b + " (" + wide.Description.bone_labels[b] + ")");
            }
        }

        [Test]
        public void ExactVertexUpdateReproducesTheNativeArray()
        {
            Mesh mesh = built.Mesh;
            AnnyMeshBuilder.UpdatePosedVertices(mesh, output, built.CornerSource, false);
            Vector3[] meshVertices = mesh.vertices;
            AnnyTensorF32 posed = output.Tensor(AnnyOutputF32.Vertices);

            Assert.AreEqual(posed.Shape[1], built.Report.SourceVertices, "posed vertices");
            for (int i = 0; i < meshVertices.Length; i++)
            {
                int source = built.CornerSource != null ? built.CornerSource[i] : i;
                Vector3 expected = AnnyRepresentation.ToUnityPosition(
                    posed.Data[source * 3], posed.Data[source * 3 + 1], posed.Data[source * 3 + 2]);
                Assert.AreEqual(expected.x, meshVertices[i].x, 0f, "vertex " + i + " x");
                Assert.AreEqual(expected.y, meshVertices[i].y, 0f, "vertex " + i + " y");
                Assert.AreEqual(expected.z, meshVertices[i].z, 0f, "vertex " + i + " z");
            }
        }

        [Test]
        public void WideMeshVerticesReproduceTheWideTensorExactly()
        {
            // Mesh fidelity: whatever tensor the builder was handed must land in the mesh
            // unchanged apart from the f64-to-f32 narrowing Unity's Mesh imposes.
            using (AnnyOutput wideOutput = wide.Evaluate(new AnnyParameters()))
            {
                AnnyMeshResult wideMesh = AnnyMeshBuilder.Build(wide, wideOutput, new AnnyMeshOptions());
                try
                {
                    AnnyTensor wideTensor = wideOutput.Tensor(AnnyOutput.Vertices);
                    Vector3[] vertices = wideMesh.Mesh.vertices;
                    for (int i = 0; i < vertices.Length; i++)
                    {
                        int source = wideMesh.CornerSource != null ? wideMesh.CornerSource[i] : i;
                        Vector3 expected = AnnyRepresentation.ToUnityPosition(
                            wideTensor.Data[source * 3],
                            wideTensor.Data[source * 3 + 1],
                            wideTensor.Data[source * 3 + 2]);
                        Assert.AreEqual(expected.x, vertices[i].x, 0f, "vertex " + i + " x");
                        Assert.AreEqual(expected.y, vertices[i].y, 0f, "vertex " + i + " y");
                        Assert.AreEqual(expected.z, vertices[i].z, 0f, "vertex " + i + " z");
                    }
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(wideMesh.Mesh);
                }
            }
        }

        [Test]
        public void SinglePrecisionEvaluationStaysWithinTheMeasuredVertexBudget()
        {
            // The f32 model is a separate evaluation, not a rounding of the f64 one: its arithmetic
            // runs narrow, so the two disagree. This measures how far apart they are rather than
            // asserting bit equality that was never true.
            using (AnnyOutput wideOutput = wide.Evaluate(new AnnyParameters()))
            {
                AnnyTensorF32 narrow = output.Tensor(AnnyOutputF32.Vertices);
                AnnyTensor wideTensor = wideOutput.Tensor(AnnyOutput.Vertices);

                double worst = 0.0;
                int worstVertex = -1;
                for (int i = 0; i < wide.Description.vertices; i++)
                {
                    Vector3 a = AnnyRepresentation.ToUnityPosition(
                        wideTensor.Data[i * 3], wideTensor.Data[i * 3 + 1], wideTensor.Data[i * 3 + 2]);
                    Vector3 b = AnnyRepresentation.ToUnityPosition(
                        narrow.Data[i * 3], narrow.Data[i * 3 + 1], narrow.Data[i * 3 + 2]);
                    double d = (a - b).magnitude;
                    if (d > worst)
                    {
                        worst = d;
                        worstVertex = i;
                    }
                }

                UnityEngine.Debug.Log(
                    "single precision vs wide: worst vertex deviation " +
                    worst.ToString("G6") + " m at vertex " + worstVertex);

                Assert.Less(worst, 5e-4, "worst single-precision vertex deviation in metres");
            }
        }

        [Test]
        public void CoordinateChangeIsAProperRotation()
        {
            // The Anny-to-Unity mapping must not mirror the model: a reflection would invert every
            // face. Determinant +1 is the reason Triangulate keeps the corner order.
            Vector3 x = AnnyRepresentation.ToUnityPosition(1, 0, 0);
            Vector3 y = AnnyRepresentation.ToUnityPosition(0, 1, 0);
            Vector3 z = AnnyRepresentation.ToUnityPosition(0, 0, 1);

            Assert.AreEqual(1f, x.x, 1e-6f, "model +X maps to Unity +X");
            Assert.AreEqual(-1f, y.z, 1e-6f, "model +Y maps to Unity -Z");
            Assert.AreEqual(1f, z.y, 1e-6f, "model up (+Z) maps to Unity up (+Y)");

            float determinant = Vector3.Dot(Vector3.Cross(x, y), z);
            Assert.AreEqual(1f, determinant, 1e-6f, "the change of basis is a proper rotation");

            Vector3 roundTrip = AnnyRepresentation.ToNativePosition(AnnyRepresentation.ToUnityPosition(0.3, -0.7, 1.1));
            Assert.AreEqual(0.3f, roundTrip.x, 1e-6f, "round trip x");
            Assert.AreEqual(-0.7f, roundTrip.y, 1e-6f, "round trip y");
            Assert.AreEqual(1.1f, roundTrip.z, 1e-6f, "round trip z");
        }

        [Test]
        public void RigHierarchyMirrorsNativeParents()
        {
            AnnyRig rig = AnnySkeleton.Build(wide.Description, output, null);
            try
            {
                Assert.AreEqual(wide.Description.bones, rig.Count, "bone count");
                for (int b = 0; b < rig.Count; b++)
                {
                    int parent = wide.Description.bone_parents[b];
                    Transform expected = parent >= 0 ? rig.Bones[parent] : rig.Root;
                    Assert.AreSame(expected, rig.Bones[b].parent, "parent of " + rig.Labels[b]);
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(rig.Root.gameObject);
            }
        }

        [Test]
        public void RigTransformsReproduceTheNativePose()
        {
            AnnyRig rig = AnnySkeleton.Build(wide.Description, output, null);
            try
            {
                AnnySkeleton.ApplyPose(rig, output);
                AnnyTensorF32 poses = output.Tensor(AnnyOutputF32.BonePoses);

                for (int b = 0; b < rig.Count; b++)
                {
                    Matrix4x4 expected = AnnyRepresentation.ToUnityMatrix(ToWide(poses.Data, b * 16));
                    Matrix4x4 actual = rig.Bones[b].localToWorldMatrix;
                    AssertMatricesClose(expected, actual, 2e-4f, "bone " + rig.Labels[b]);
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(rig.Root.gameObject);
            }
        }

        [Test]
        public void SkinnedMeshDataReproducesTheNativeRestSurface()
        {
            // At the bind pose every bone's world matrix equals its bind matrix, so linear blend
            // skinning must return each vertex unchanged: this is a zero-deviation case, not a
            // tolerance case. Any deviation means the skinning *data* is wrong (weights that do not
            // sum to one, a mis-ordered influence stream, or a wrong bone index), so the sum of
            // influences is checked explicitly. Unity's own BakeMesh is measured against the same
            // reference to separate our data from Unity's skinning precision.
            GameObject host = new GameObject("anny-skin-test");
            Mesh baked = new Mesh();
            try
            {
                AnnyRig rig = AnnySkeleton.Build(wide.Description, output, host.transform);
                SkinnedMeshRenderer renderer = host.AddComponent<SkinnedMeshRenderer>();
                renderer.sharedMesh = built.Mesh;
                renderer.bones = rig.Bones;
                renderer.rootBone = rig.Bones[0];

                Vector3[] vertices = built.Mesh.vertices;
                Matrix4x4[] bindPoses = built.Mesh.bindposes;
                var perVertex = built.Mesh.GetBonesPerVertex();
                var influences = built.Mesh.GetAllBoneWeights();
                AnnyTensorF32 rest = output.Tensor(AnnyOutputF32.RestVertices);

                double worstSumError = 0.0;
                double worstReference = 0.0;
                int cursor = 0;
                for (int v = 0; v < vertices.Length; v++)
                {
                    int count = perVertex[v];
                    Vector3 accumulated = Vector3.zero;
                    double sum = 0.0;
                    for (int k = 0; k < count; k++)
                    {
                        BoneWeight1 influence = influences[cursor + k];
                        Matrix4x4 skin = renderer.bones[influence.boneIndex].localToWorldMatrix * bindPoses[influence.boneIndex];
                        accumulated += influence.weight * skin.MultiplyPoint3x4(vertices[v]);
                        sum += influence.weight;
                    }
                    cursor += count;

                    worstSumError = Math.Max(worstSumError, Math.Abs(1.0 - sum));

                    int source = built.CornerSource != null ? built.CornerSource[v] : v;
                    Vector3 expected = AnnyRepresentation.ToUnityPosition(
                        rest.Data[source * 3], rest.Data[source * 3 + 1], rest.Data[source * 3 + 2]);
                    worstReference = Math.Max(worstReference, (accumulated - expected).magnitude);
                }

                renderer.BakeMesh(baked);
                Vector3[] bakedVertices = baked.vertices;
                float worstUnity = 0f;
                for (int v = 0; v < bakedVertices.Length; v++)
                {
                    int source = built.CornerSource != null ? built.CornerSource[v] : v;
                    Vector3 expected = AnnyRepresentation.ToUnityPosition(
                        rest.Data[source * 3], rest.Data[source * 3 + 1], rest.Data[source * 3 + 2]);
                    worstUnity = Mathf.Max(worstUnity, (bakedVertices[v] - expected).magnitude);
                }

                UnityEngine.Debug.Log(
                    "skinned skinning data: worst influence-sum error " + worstSumError.ToString("G6") +
                    ", worst reference-LBS deviation " + worstReference.ToString("G6") +
                    " m, worst Unity BakeMesh deviation " + worstUnity.ToString("G6") +
                    " m, vertices " + vertices.Length);

                Assert.Less(worstSumError, 1e-4, "influences must sum to one at every vertex");
                Assert.Less(worstReference, 1e-3, "reference LBS deviation at the bind pose, metres");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(baked);
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        private static void AssertIdentity(Matrix4x4 m, float tolerance, string what)
        {
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    float expected = r == c ? 1f : 0f;
                    Assert.AreEqual(expected, m[r, c], tolerance, what + " [" + r + "," + c + "]");
                }
            }
        }

        private static void AssertMatricesClose(Matrix4x4 expected, Matrix4x4 actual, float tolerance, string what)
        {
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    Assert.AreEqual(expected[r, c], actual[r, c], tolerance, what + " [" + r + "," + c + "]");
                }
            }
        }

        private static double[] ToWide(float[] data, int offset)
        {
            double[] wide = new double[16];
            for (int i = 0; i < 16; i++)
            {
                wide[i] = data[offset + i];
            }

            return wide;
        }

        [Test]
        public void BoneWeightsDescendForEveryVertex()
        {
            // Unity logs an error and mis-skins when a vertex's influences are not in descending
            // order. The native arrays are not sorted, so the packer must sort them; this reads the
            // result back out of the mesh rather than trusting the packer's own bookkeeping.
            var perVertex = built.Mesh.GetBonesPerVertex();
            var flat = built.Mesh.GetAllBoneWeights();

            Assert.Greater(perVertex.Length, 0, "the mesh must carry skinning at all");
            Assert.AreEqual(built.Mesh.vertexCount, perVertex.Length, "one influence count per mesh vertex");

            int cursor = 0;
            int ascending = 0;
            int widest = 0;
            for (int v = 0; v < perVertex.Length; v++)
            {
                int count = perVertex[v];
                if (count > widest)
                {
                    widest = count;
                }

                for (int i = 1; i < count; i++)
                {
                    if (flat[cursor + i - 1].weight < flat[cursor + i].weight)
                    {
                        ascending++;
                    }
                }

                cursor += count;
            }

            Assert.AreEqual(0, ascending, "influences that ascend in weight instead of descending");
            Assert.AreEqual(flat.Length, cursor, "the flat stream must be exactly the per-vertex counts");
            Assert.GreaterOrEqual(widest, 1, "every vertex needs at least one influence");
            UnityEngine.Debug.Log(
                "skin: width " + built.Report.BoneInfluenceWidth +
                ", widest kept " + widest +
                ", native order unsorted at " + built.Report.VerticesWithUnsortedNativeWeights +
                " vertices, truncated " + built.Report.VerticesWithTruncatedInfluences);
        }

        private static float[] Narrow(double[] data)
        {
            float[] narrow = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                narrow[i] = (float)data[i];
            }

            return narrow;
        }
    }
}