using System;
using System.IO;
using Anny;
using UnityEditor;
using UnityEngine;

namespace Anny.EditorTools
{
    /// <summary>
    /// Turns a generated character into ordinary Unity assets: the prepared payload as a .bytes
    /// text asset, a model asset carrying the cached description, a mesh asset with bind poses,
    /// weights and morph targets, a material, and a prefab holding the bone hierarchy and the
    /// skinned renderer.
    /// <para>
    /// Everything the baker writes is a normal Unity object that can be edited, versioned and
    /// shipped without the Anny toolchain present. The native library is only needed to author the
    /// bake, and again at runtime when a behaviour asks Anny to pose.
    /// </para>
    /// </summary>
    public static class AnnyBaker
    {
        public const string DefaultFolder = "Assets/Anny/Baked";

        /// <summary>What one bake produced, so callers and tests can assert on it.</summary>
        public sealed class Result
        {
            public string Folder;
            public string PayloadPath;
            public string ModelPath;
            public string MeshPath;
            public string MaterialPath;
            public string PrefabPath;
            public AnnyModelAsset ModelAsset;
            public Mesh Mesh;
            public GameObject Prefab;
            public AnnyMeshReport Report;
            public string[] BlendShapeNames;
        }

        /// <summary>Bakes a prepared payload into Unity assets under <paramref name="folder"/>.</summary>
        public static Result Bake(
            string payloadPath,
            string folder = DefaultFolder,
            string assetName = "AnnyCharacter",
            AnnyMeshOptions meshOptions = null,
            AnnyParameters parameters = null)
        {
            if (string.IsNullOrEmpty(payloadPath) || !File.Exists(payloadPath))
            {
                throw new FileNotFoundException("prepared model payload not found", payloadPath);
            }

            meshOptions = meshOptions ?? new AnnyMeshOptions();
            EnsureFolder(folder);

            Result result = new Result { Folder = folder };

            // 1. The payload travels as a .bytes text asset, which Unity imports byte for byte.
            string payloadName = assetName + ".prepared.bytes";
            result.PayloadPath = folder + "/" + payloadName;
            File.Copy(payloadPath, ToAbsolute(result.PayloadPath), true);
            AssetDatabase.ImportAsset(result.PayloadPath, ImportAssetOptions.ForceUpdate);

            // 2. The model asset caches the description so runtime code needs no extra parse.
            result.ModelPath = folder + "/" + assetName + ".asset";
            using (AnnyModel wide = AnnyModel.Load(payloadPath, null))
            {
                AnnyModelAsset modelAsset = ScriptableObject.CreateInstance<AnnyModelAsset>();
                modelAsset.payload = AssetDatabase.LoadAssetAtPath<TextAsset>(result.PayloadPath);
                modelAsset.singlePrecision = true;
                modelAsset.CacheDescription(wide);
                AssetDatabase.CreateAsset(modelAsset, result.ModelPath);
                result.ModelAsset = modelAsset;

                // 3. The mesh, built through the same path the runtime uses.
                using (AnnyModelF32 runtime = wide.ToSinglePrecision())
                {
                    AnnyParameters effective = parameters ?? new AnnyParameters();
                    using (AnnyOutputF32 output = runtime.Evaluate(effective))
                    {
                        AnnyMeshResult built = AnnyMeshBuilder.Build(runtime, output, meshOptions);
                        built.Mesh.name = assetName;
                        result.MeshPath = folder + "/" + assetName + ".mesh.asset";
                        AssetDatabase.CreateAsset(built.Mesh, result.MeshPath);
                        result.Mesh = built.Mesh;
                        result.Report = built.Report;
                        result.BlendShapeNames = built.Report.BlendShapeNames;
                    }
                }
            }

            // 4. A prefab holding the hierarchy, so a scene can just reference it.
            result.MaterialPath = folder + "/" + assetName + ".mat";
            Material material = AnnyMaterials.CreateDefault(assetName);
            AssetDatabase.CreateAsset(material, result.MaterialPath);

            result.PrefabPath = folder + "/" + assetName + ".prefab";
            result.Prefab = SavePrefab(result, assetName);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return result;
        }

        private static GameObject SavePrefab(Result result, string assetName)
        {
            GameObject root = new GameObject(assetName);
            try
            {
                AnnyCharacter character = root.AddComponent<AnnyCharacter>();
                character.modelAsset = result.ModelAsset;
                character.generateOnAwake = true;
                character.meshOptions = new AnnyMeshOptions
                {
                    ExpandUvCorners = true,
                    BuildBlendshapes = result.BlendShapeNames != null && result.BlendShapeNames.Length > 0,
                };
                character.material = AssetDatabase.LoadAssetAtPath<Material>(result.MaterialPath);

                // Generate once so the prefab is authored with a real bone hierarchy, then point the
                // renderer at the saved mesh asset so the prefab does not carry a duplicate.
                character.name = assetName;
                character.Generate();

                if (character.Skinned != null)
                {
                    character.Skinned.sharedMesh = result.Mesh;
                }

                MeshFilter filter = root.GetComponent<MeshFilter>();
                if (filter != null)
                {
                    filter.sharedMesh = result.Mesh;
                }

                GameObject saved = PrefabUtility.SaveAsPrefabAsset(root, result.PrefabPath);
                return saved;
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        /// <summary>Creates a folder chain under Assets if it does not exist yet.</summary>
        public static void EnsureFolder(string folder)
        {
            if (AssetDatabase.IsValidFolder(folder))
            {
                return;
            }

            string[] parts = folder.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(current, parts[i]);
                }

                current = next;
            }
        }

        private static string ToAbsolute(string assetPath)
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectRoot, assetPath);
        }
    }
}