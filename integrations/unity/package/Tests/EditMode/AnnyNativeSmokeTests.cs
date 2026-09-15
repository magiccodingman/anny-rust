using System;
using System.IO;
using Anny;
using NUnit.Framework;

namespace Anny.Tests
{
    /// <summary>
    /// Exercises the native library from inside the Unity runtime: ABI handshake, model loading,
    /// description parsing, evaluation and the error path. These tests need a prepared model; set
    /// <c>ANNY_MODEL</c> or run from the repository so the default path resolves.
    /// </summary>
    public class AnnyNativeSmokeTests
    {
        [Test]
        public void AbiVersionMatchesTheRuntime()
        {
            Assert.AreEqual(AnnyNative.ExpectedAbi, AnnyNative.AbiVersion(), "native ABI version mismatch");
        }

        [Test]
        public void SuccessfulCallLeavesNoErrorBehind()
        {
            AnnyNative.RequireAbi();
            Assert.AreEqual(string.Empty, AnnyNative.LastError(), "a clean call must not leave error text");
        }

        [Test]
        public void MissingModelReportsThePathItCouldNotOpen()
        {
            string missing = Path.Combine(Path.GetTempPath(), "anny-does-not-exist-91af.safetensors");
            File.Delete(missing);
            AnnyNativeException error = Assert.Throws<AnnyNativeException>(() => AnnyModel.Load(missing));
            StringAssert.Contains("91af", error.Message, "the native error text should name the missing path");
        }

        [Test]
        public void ModelDescribesItsTopology()
        {
            using (AnnyModel model = AnnyTestModels.Load())
            {
                AnnyDescription description = model.Description;
                Assert.Greater(description.vertices, 0, "vertex count");
                Assert.Greater(description.faces, 0, "face count");
                Assert.Greater(description.bones, 0, "bone count");
                Assert.AreEqual(description.vertices, model.Tensor("template_vertices").Shape[0], "vertex count matches the array");
                Assert.AreEqual(description.bones, description.bone_labels.Length, "every bone has a label");
            }
        }

        [Test]
        public void EvaluationProducesPosedVerticesAndBoneMatrices()
        {
            using (AnnyModel model = AnnyTestModels.Load())
            using (AnnyOutput output = model.Evaluate(null))
            {
                AnnyTensor vertices = output.Tensor(AnnyOutput.Vertices);
                Assert.AreEqual(3, vertices.Rank, "vertices rank");
                Assert.AreEqual(1, vertices.Shape[0], "a single evaluation has one batch");
                Assert.AreEqual(model.Description.vertices, vertices.Shape[1], "posed vertex count");

                AnnyTensor bones = output.Tensor(AnnyOutput.BonePoses);
                Assert.AreEqual(4, bones.Rank, "bone pose rank");
                Assert.AreEqual(model.Description.bones, bones.Shape[1], "bone count");

                Assert.IsTrue(output.HasTensor(AnnyOutput.RestVertices), "rest vertices come back with an evaluation");
                Assert.IsFalse(output.HasTensor("no_such_tensor"), "an absent tensor reports absence");
            }
        }

        [Test]
        public void RestPoseEvaluationIsStableAcrossCalls()
        {
            using (AnnyModel model = AnnyTestModels.Load())
            {
                float[] first = model.Evaluate(null).Tensor(AnnyOutput.Vertices).ToFloats();
                try
                {
                    float[] second = model.Evaluate(null).Tensor(AnnyOutput.Vertices).ToFloats();
                    try
                    {
                        Assert.AreEqual(first.Length, second.Length, "length");
                        for (int i = 0; i < first.Length; i++)
                        {
                            if (first[i] != second[i])
                            {
                                Assert.Fail("rest evaluation differs at element " + i + ": " + first[i] + " vs " + second[i]);
                            }
                        }
                    }
                    finally
                    {
                        second = null;
                    }
                }
                finally
                {
                    first = null;
                }
            }
        }
    }
}