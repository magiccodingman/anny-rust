using System.IO;
using Anny;
using UnityEditor;
using UnityEngine;

namespace Anny.EditorTools
{
    /// <summary>
    /// Turns a prepared payload into ordinary Unity assets. This is the step that lets an Anny
    /// character ship without the native plugin, so it is a window rather than a menu item: the
    /// payload, destination and geometry options are worth seeing before the bake runs, and the
    /// options are the ones that change the generated geometry.
    /// </summary>
    public sealed class AnnyBakeWindow : EditorWindow
    {
        // Window state is public so a test can drive the mapping without simulating the GUI.
        public string payloadPath = "";
        public string folder = AnnyBaker.DefaultFolder;
        public string assetName = "AnnyCharacter";

        public bool expandUvCorners = true;
        public bool buildBlendshapes = true;
        public int maxBoneInfluences;

        private AnnyBaker.Result baked;
        private string message;
        private Vector2 scroll;

        [MenuItem("Anny/Bake prepared payload...")]
        public static void Open()
        {
            AnnyBakeWindow window = GetWindow<AnnyBakeWindow>(false, "Anny Bake", true);
            window.minSize = new Vector2(380f, 300f);
            window.Show();
        }

        /// <summary>Options exactly as the window shows them, so a test can check the mapping.</summary>
        public AnnyMeshOptions CurrentOptions()
        {
            return new AnnyMeshOptions
            {
                ExpandUvCorners = expandUvCorners,
                BuildBlendshapes = buildBlendshapes,
                MaxBoneInfluences = maxBoneInfluences,
            };
        }

        /// <summary>True when the payload field points at something the baker can open.</summary>
        public bool PayloadExists()
        {
            return !string.IsNullOrEmpty(payloadPath) && File.Exists(payloadPath);
        }

        private void OnGUI()
        {
            scroll = EditorGUILayout.BeginScrollView(scroll);

            EditorGUILayout.HelpBox(
                "Bakes a prepared .safetensors payload into a model asset, mesh, material and prefab. " +
                "The result needs no native plugin at runtime.",
                MessageType.Info);

            EditorGUILayout.LabelField("Payload", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                payloadPath = EditorGUILayout.TextField("Prepared payload", payloadPath);
                if (GUILayout.Button("Use selection", GUILayout.Width(100f)))
                {
                    UseSelection();
                }
            }

            if (!string.IsNullOrEmpty(payloadPath) && !File.Exists(payloadPath))
            {
                EditorGUILayout.HelpBox("No file at that path.", MessageType.Warning);
            }

            EditorGUILayout.LabelField("Output", EditorStyles.boldLabel);
            folder = EditorGUILayout.TextField("Folder", folder);
            assetName = EditorGUILayout.TextField("Asset name", assetName);

            EditorGUILayout.LabelField("Geometry options", EditorStyles.boldLabel);
            expandUvCorners = EditorGUILayout.Toggle("Expand UV corners", expandUvCorners);
            buildBlendshapes = EditorGUILayout.Toggle("Build blendshapes", buildBlendshapes);
            maxBoneInfluences = EditorGUILayout.IntField("Max bone influences (0 = all)", maxBoneInfluences);

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(!PayloadExists()))
            {
                if (GUILayout.Button("Bake"))
                {
                    Bake();
                }
            }

            if (!string.IsNullOrEmpty(message))
            {
                EditorGUILayout.Space();
                EditorGUILayout.HelpBox(message, baked != null ? MessageType.Info : MessageType.Error);
            }

            if (baked != null && baked.Prefab != null)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Select prefab"))
                    {
                        Selection.activeObject = baked.Prefab;
                    }

                    if (GUILayout.Button("Ping folder"))
                    {
                        EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<Object>(baked.Folder));
                    }
                }
            }

            EditorGUILayout.EndScrollView();
        }

        private void UseSelection()
        {
            TextAsset selected = Selection.activeObject as TextAsset;
            if (selected == null)
            {
                message = "Select an imported payload (.bytes) asset first.";
                baked = null;
                return;
            }

            payloadPath = AssetDatabase.GetAssetPath(selected);
            message = null;
        }

        private void Bake()
        {
            try
            {
                baked = AnnyBaker.Bake(payloadPath, folder, assetName, CurrentOptions());
                message = string.Format(
                    "Baked {0} vertices and {1} triangles into {2}.",
                    baked.Report.SourceVertices,
                    baked.Report.Triangles,
                    baked.Folder);
            }
            catch (System.Exception error)
            {
                baked = null;
                message = "Bake failed: " + error.Message;
            }
        }
    }
}
