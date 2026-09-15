using Anny;
using UnityEditor;
using UnityEngine;

namespace Anny.EditorTools
{
    /// <summary>
    /// Inspector for a generated character. It drives generation from edit mode, exposes the phenotype
    /// sliders by label, applies and captures presets, and reports what the mesh builder actually
    /// produced. The report matters because the usual failure is not an exception: it is a
    /// plausible-looking character that quietly dropped bone influences.
    /// </summary>
    [CustomEditor(typeof(AnnyCharacter))]
    public sealed class AnnyCharacterEditor : Editor
    {
        private AnnyPreset preset;

        public override void OnInspectorGUI()
        {
            AnnyCharacter character = (AnnyCharacter)target;

            DrawDefaultInspector();

            EditorGUILayout.Space();
            DrawGeneration(character);
            DrawReport(character);
            DrawPhenotype(character);
            DrawPresets(character);
            DrawBake(character);
        }

        private void DrawGeneration(AnnyCharacter character)
        {
            EditorGUILayout.LabelField("Generation", EditorStyles.boldLabel);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(character.GeneratedMesh == null ? "Generate" : "Regenerate"))
                {
                    Generate(character);
                }
            }

            EditorGUILayout.LabelField(
                "Update mode",
                character.mode == AnnyUpdateMode.Exact ? "Exact (native vertices, zero tolerance)" : "Skinned (bones, measured tolerance)");
        }

        private void Generate(AnnyCharacter character)
        {
            try
            {
                character.Generate();
                EditorUtility.SetDirty(character);
                SceneView.RepaintAll();
            }
            catch (System.Exception error)
            {
                Debug.LogError("Anny: generation failed: " + error.Message, character);
            }
        }

        private static void DrawReport(AnnyCharacter character)
        {
            AnnyMeshReport report = character.Report;
            if (report == null)
            {
                EditorGUILayout.HelpBox("Not generated yet.", MessageType.Info);
                return;
            }

            EditorGUILayout.LabelField("Result", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                "Vertices",
                string.Format("{0} source -> {1} mesh", report.SourceVertices, report.MeshVertices));
            EditorGUILayout.LabelField(
                "Faces",
                string.Format("{0} triangles from {1} faces, {2} of them quads", report.Triangles, report.Faces, report.QuadFaces));
            EditorGUILayout.LabelField(
                "Influences",
                string.Format("{0} per vertex, buffer width {1}", report.MaxInfluencesPerVertex, report.BoneInfluenceWidth));
            EditorGUILayout.LabelField("Skin weights", report.SkinWeightsExact ? "exact" : "approximate");
            EditorGUILayout.LabelField("Signed volume", string.Format("{0:0.########}", report.SignedVolume));
            EditorGUILayout.LabelField("Last evaluation", string.Format("{0:0.###} ms", character.LastEvaluateMs));

            if (report.VerticesWithTruncatedInfluences > 0)
            {
                EditorGUILayout.HelpBox(
                    string.Format(
                        "{0} vertices lost influences (largest dropped {1:0.######}). Set max bone influences to 0 on the mesh options to keep them all.",
                        report.VerticesWithTruncatedInfluences,
                        report.LargestDroppedWeight),
                    MessageType.Warning);
            }

            if (report.VerticesWithUnsortedNativeWeights > 0)
            {
                EditorGUILayout.HelpBox(
                    string.Format("{0} vertices arrived with unsorted native weights and were sorted.", report.VerticesWithUnsortedNativeWeights),
                    MessageType.Warning);
            }
        }

        private static void DrawPhenotype(AnnyCharacter character)
        {
            string[] labels = character.PhenotypeLabels;
            if (labels == null || labels.Length == 0)
            {
                return;
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Phenotype", EditorStyles.boldLabel);

            bool changed = false;
            EditorGUI.BeginChangeCheck();
            foreach (string label in labels)
            {
                float value = EditorGUILayout.Slider(label, character.GetPhenotype(label), 0f, 1f);
                if (!Mathf.Approximately(value, character.GetPhenotype(label)))
                {
                    Undo.RecordObject(character, "Anny phenotype");
                    character.SetPhenotype(label, value);
                    changed = true;
                }
            }

            if (EditorGUI.EndChangeCheck() && changed)
            {
                // Editing a slider is the same update path the runtime uses, so the viewport shows
                // exactly what the runtime would show.
                character.Apply();
                EditorUtility.SetDirty(character);
                SceneView.RepaintAll();
            }
        }

        private void DrawPresets(AnnyCharacter character)
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Presets", EditorStyles.boldLabel);
            preset = (AnnyPreset)EditorGUILayout.ObjectField("Preset", preset, typeof(AnnyPreset), false);

            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUI.DisabledScope(preset == null))
                {
                    if (GUILayout.Button("Apply to this character"))
                    {
                        preset.ApplyTo(character);
                        Generate(character);
                    }
                }

                if (GUILayout.Button("Capture current settings..."))
                {
                    CapturePreset(character);
                }
            }
        }

        private void CapturePreset(AnnyCharacter character)
        {
            string path = EditorUtility.SaveFilePanelInProject(
                "Save Anny preset",
                character.name + " Preset",
                "asset",
                "Presets are ordinary Unity assets and travel with the project.");

            if (string.IsNullOrEmpty(path))
            {
                return;
            }

            AnnyPreset created = ScriptableObject.CreateInstance<AnnyPreset>();
            created.CaptureFrom(character);
            AssetDatabase.CreateAsset(created, path);
            AssetDatabase.SaveAssets();
            preset = created;
        }

        private void DrawBake(AnnyCharacter character)
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Bake", EditorStyles.boldLabel);

            string payloadPath = character.modelAsset != null && character.modelAsset.payload != null
                ? AssetDatabase.GetAssetPath(character.modelAsset.payload)
                : null;

            if (string.IsNullOrEmpty(payloadPath))
            {
                EditorGUILayout.HelpBox(
                    "Assign a model asset whose payload is an imported .bytes asset, then bake to ordinary Unity assets.",
                    MessageType.Info);
                return;
            }

            if (GUILayout.Button("Bake model, mesh, material and prefab"))
            {
                try
                {
                    AnnyBaker.Result result = AnnyBaker.Bake(payloadPath, meshOptions: character.meshOptions);
                    Selection.activeObject = result.Prefab;
                    Debug.Log("Anny: baked into " + result.Folder, result.Prefab);
                }
                catch (System.Exception error)
                {
                    Debug.LogError("Anny: bake failed: " + error.Message, character);
                }
            }
        }
    }
}
