using System.IO;
using Anny;
using Anny.EditorTools;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Anny.Tests
{
    /// <summary>
    /// Exercises the bake-to-Unity-assets workflow end to end: a prepared model payload on disk
    /// becomes ordinary Unity assets, and the baked mesh is the same geometry the runtime path
    /// produces directly. The bake writes real files, so it cleans up after itself.
    /// </summary>
    public sealed class AnnyBakeTests
    {
        private const string Folder = "Assets/AnnyBakeProbe";

        [TearDown]
        public void TearDown()
        {
            if (AssetDatabase.IsValidFolder(Folder))
            {
                AssetDatabase.DeleteAsset(Folder);
                AssetDatabase.Refresh();
            }
        }

        [Test]
        public void BakeProducesUnityAssetsForTheWholeCharacter()
        {
            AnnyBaker.Result baked = AnnyBaker.Bake(
                AnnyTestModels.PreparedModelPath, Folder, "BakeProbe");

            Assert.IsNotNull(baked.ModelAsset, "model asset");
            Assert.IsTrue(File.Exists(Path.Combine(Folder, "BakeProbe.prepared.bytes")), "payload asset");
            Assert.IsNotNull(AssetDatabase.LoadAssetAtPath<Mesh>(baked.MeshPath), "mesh asset");
            Assert.IsNotNull(AssetDatabase.LoadAssetAtPath<Material>(baked.MaterialPath), "material asset");
            Assert.IsNotNull(
                AssetDatabase.LoadAssetAtPath<GameObject>(baked.PrefabPath), "prefab asset");

            Assert.AreEqual(AnnyTestModels.ExpectedVertices, baked.ModelAsset.Description.vertices, "baked description");
            Assert.IsNotNull(baked.ModelAsset.Description.bone_labels, "baked bone labels");

            // The baked mesh must carry skinning, not just geometry.
            Assert.Greater(baked.Mesh.vertexCount, 0, "baked mesh vertices");
            Assert.AreEqual(baked.ModelAsset.Description.bones, baked.Mesh.bindposes.Length, "baked bind poses");
            Assert.Greater(baked.Mesh.subMeshCount, 0, "baked submeshes");

            // And it must be the geometry the runtime path produces directly.
            using (AnnyModelF32 runtime = baked.ModelAsset.OpenRuntime())
            using (AnnyOutputF32 output = runtime.Evaluate(new AnnyParameters()))
            {
                AnnyMeshResult direct = AnnyMeshBuilder.Build(runtime, output, new AnnyMeshOptions());
                try
                {
                    Assert.AreEqual(direct.Mesh.vertexCount, baked.Mesh.vertexCount, "vertex count matches direct build");
                    Assert.AreEqual(
                        direct.Mesh.triangles.Length, baked.Mesh.triangles.Length, "triangle count matches direct build");
                    Assert.AreEqual(
                        direct.Report.Faces, baked.Report.Faces, "face count matches direct build");
                }
                finally
                {
                    Object.DestroyImmediate(direct.Mesh);
                }
            }
        }

        [Test]
        public void BakedPrefabCarriesTheSkeletonAndRenderer()
        {
            AnnyBaker.Result baked = AnnyBaker.Bake(
                AnnyTestModels.PreparedModelPath, Folder, "BakeProbeRig");

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(baked.PrefabPath);
            Assert.IsNotNull(prefab, "prefab");

            SkinnedMeshRenderer renderer = prefab.GetComponentInChildren<SkinnedMeshRenderer>();
            Assert.IsNotNull(renderer, "baked prefab carries a SkinnedMeshRenderer");
            Assert.IsNotNull(renderer.sharedMesh, "renderer mesh");
            Assert.AreEqual(renderer.sharedMesh.bindposes.Length, renderer.bones.Length, "bones match bind poses");
            Assert.IsNotNull(renderer.rootBone, "root bone");
        }
    }
}
