using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Anny
{
    /// <summary>How an <see cref="AnnyCharacter"/> gets its posed surface into Unity.</summary>
    public enum AnnyUpdateMode
    {
        /// <summary>
        /// Bones drive Unity's own skinning. Cheap for animation, and the surface is produced by
        /// Unity's linear blend skinning rather than by Anny, so it agrees with the native result
        /// only to a measured tolerance.
        /// </summary>
        Skinned,

        /// <summary>
        /// Anny poses the vertices and they are uploaded to the mesh each update. Slower, but the
        /// surface is the native result itself and can be compared bit for bit.
        /// </summary>
        Exact,
    }

    /// <summary>One named phenotype or local-change slider and the value applied to it.</summary>
    [Serializable]
    public struct AnnySliderEntry
    {
        public string name;

        [Range(-1f, 1f)]
        public float value;
    }

    /// <summary>
    /// An Anny character living in a Unity scene: model, mesh, skeleton, renderer and the update
    /// path that keeps them in step.
    /// </summary>
    [AddComponentMenu("Anny/Anny Character")]
    public sealed class AnnyCharacter : MonoBehaviour
    {
        [Header("Model")]
        [Tooltip("Prepared model payload. Assigned by the Anny baker, or by hand from a .bytes asset.")]
        public AnnyModelAsset modelAsset;

        [Tooltip("Compatibility loading switch. Unity's live mesh runtime is always f32; when disabled the payload is opened wide and converted once before use.")]
        public bool singlePrecision = true;

        [Tooltip("Skinned drives Unity skinning from the bones; Exact uploads native posed vertices.")]
        public AnnyUpdateMode mode = AnnyUpdateMode.Skinned;

        public AnnyMeshOptions meshOptions = new AnnyMeshOptions();

        [Header("Parameters")]
        [Tooltip("Phenotype and local-change sliders. Names come from the model description.")]
        public List<AnnySliderEntry> phenotype = new List<AnnySliderEntry>();

        [Tooltip("Reuse a native pose session for pose-only updates instead of re-evaluating.")]
        public bool usePoseSession = true;

        [Header("Behaviour")]
        public bool generateOnAwake = true;

        [Tooltip("Material applied to the generated renderer. Created from the active pipeline when unset.")]
        public Material material;

        /// <summary>The model actually generating this character.</summary>
        public AnnyModelF32 Runtime { get; private set; }

        public AnnyRig Rig { get; private set; }

        public Mesh GeneratedMesh { get; private set; }

        public AnnyMeshReport Report { get; private set; }

        /// <summary>Mesh vertex to source vertex map, when UV corners were expanded.</summary>
        public int[] CornerSource { get; private set; }

        public SkinnedMeshRenderer Skinned { get; private set; }

        public MeshRenderer MeshRenderer { get; private set; }

        public AnnyPoseSessionF32 Session { get; private set; }

        /// <summary>Phenotype labels the generated model exposed, in native order.</summary>
        public string[] PhenotypeLabels { get; private set; }

        /// <summary>
        /// Updates applied since this character last generated: <see cref="Generate"/> counts as the
        /// first one and every <see cref="Apply"/> or session pose update adds exactly one, so a
        /// caller can tell a re-evaluation from a reused pose session by the delta.
        /// </summary>
        public int Evaluations { get; private set; }

        /// <summary>
        /// A copy of the vertex array from the most recent evaluation, in Anny coordinates and
        /// unexpanded: one triple per model vertex, whatever the mesh's corner expansion did. This
        /// is what the mesh was built from, so a caller can compare the two.
        /// </summary>
        public float[] LastVertices()
        {
            return lastVertices;
        }

        /// <summary>Wall-clock milliseconds spent in the most recent native evaluation.</summary>
        public double LastEvaluateMs { get; private set; }

        private AnnyOutputF32 lastOutput;
        private float[] lastVertices;
        private MeshFilter filter;

        private void Awake()
        {
            // A character configured after it was added (pooling, instantiation, tests) has no model
            // yet; that is a normal state, not an error, so Awake stays quiet. Calling Generate by
            // hand with no model still reports the mistake.
            if (generateOnAwake && modelAsset != null)
            {
                Generate();
            }
        }

        private void OnDestroy()
        {
            if (Session != null)
            {
                Session.Dispose();
                Session = null;
            }

            if (lastOutput != null)
            {
                lastOutput.Dispose();
                lastOutput = null;
            }

            if (Runtime != null)
            {
                Runtime.Dispose();
                Runtime = null;
            }

            if (Rig != null && Rig.Root != null)
            {
                GameObject rigObject = Rig.Root.gameObject;
                if (Application.isPlaying)
                {
                    Destroy(rigObject);
                }
                else
                {
                    DestroyImmediate(rigObject);
                }
            }
            Rig = null;

            if (GeneratedMesh != null)
            {
                if (Application.isPlaying)
                {
                    Destroy(GeneratedMesh);
                }
                else
                {
                    DestroyImmediate(GeneratedMesh);
                }

                GeneratedMesh = null;
            }
        }

        /// <summary>
        /// Loads the model and builds the mesh, skeleton and renderer. Safe to call again to rebuild
        /// after changing <see cref="meshOptions"/>.
        /// </summary>
        public void Generate()
        {
            if (modelAsset == null)
            {
                throw new InvalidOperationException(name + " has no model asset assigned");
            }

            if (Runtime != null)
            {
                OnDestroy();
            }

            AnnyModelF32 runtime;
            string[] labels;
            if (singlePrecision)
            {
                runtime = modelAsset.OpenRuntime();
                labels = modelAsset.Description != null ? modelAsset.Description.PhenotypeLabels() : null;
            }
            else
            {
                using (AnnyModel wide = modelAsset.OpenWide())
                {
                    runtime = wide.ToSinglePrecision();
                    labels = wide.Description.PhenotypeLabels();
                }
            }

            Runtime = runtime;
            PhenotypeLabels = labels;

            AnnyParameters parameters = BuildParameters();
            Stopwatch watch = Stopwatch.StartNew();
            lastOutput = runtime.Evaluate(parameters);
            watch.Stop();
            LastEvaluateMs = watch.Elapsed.TotalMilliseconds;
            Evaluations = 1;

            // A skinned character must carry the bind-pose geometry: the renderer's bones supply the
            // pose. Exact mode uploads the evaluated vertices and needs no skinning at all.
            meshOptions.UseBindPoseGeometry = mode == AnnyUpdateMode.Skinned;
            AnnyMeshResult built = AnnyMeshBuilder.Build(runtime, lastOutput, meshOptions);
            GeneratedMesh = built.Mesh;
            Report = built.Report;
            CornerSource = built.CornerSource;
            GeneratedMesh.name = name + "-anny";

            if (modelAsset.Description == null)
            {
                // A payload without a cached description is still usable: read it from the model.
                using (AnnyModel wide = modelAsset.OpenWide())
                {
                    modelAsset.CacheDescription(wide);
                }
            }

            Rig = AnnySkeleton.Build(modelAsset.Description, lastOutput, transform);
            InstallRenderer();

            if (usePoseSession)
            {
                Session = runtime.CreateSession(parameters);
            }

            // Finish the job: a generated character carries the evaluation it was built from, so the
            // mesh, the rig and the cached vertex array all agree before the first frame is drawn.
            PushOutput(lastOutput);

            WarnIfQualityDropsInfluences();
        }

        /// <summary>Parameters from the sliders on this component.</summary>
        public AnnyParameters BuildParameters()
        {
            AnnyParameters parameters = new AnnyParameters();
            AnnyDescription description = modelAsset != null ? modelAsset.Description : null;
            for (int i = 0; i < phenotype.Count; i++)
            {
                AnnySliderEntry entry = phenotype[i];
                if (string.IsNullOrEmpty(entry.name))
                {
                    continue;
                }

                if (description != null && description.IsLocalChange(entry.name))
                {
                    parameters.LocalChangesKwargs[entry.name] = entry.value;
                }
                else
                {
                    parameters.PhenotypeKwargs[entry.name] = entry.value;
                }
            }

            return parameters;
        }

        /// <summary>Sets one slider, creating it if the model exposes that name.</summary>
        public void SetPhenotype(string sliderName, float value)
        {
            for (int i = 0; i < phenotype.Count; i++)
            {
                if (phenotype[i].name == sliderName)
                {
                    AnnySliderEntry entry = phenotype[i];
                    entry.value = value;
                    phenotype[i] = entry;
                    return;
                }
            }

            phenotype.Add(new AnnySliderEntry { name = sliderName, value = value });
        }

        public float GetPhenotype(string sliderName)
        {
            for (int i = 0; i < phenotype.Count; i++)
            {
                if (phenotype[i].name == sliderName)
                {
                    return phenotype[i].value;
                }
            }

            return 0f;
        }

        /// <summary>Re-evaluates the model and pushes the result into the scene.</summary>
        public void Apply()
        {
            if (Runtime == null)
            {
                Generate();
                return;
            }

            AnnyParameters parameters = BuildParameters();
            Stopwatch watch = Stopwatch.StartNew();
            AnnyOutputF32 output = Runtime.Evaluate(parameters);
            watch.Stop();
            LastEvaluateMs = watch.Elapsed.TotalMilliseconds;
            Evaluations++;

            if (lastOutput != null)
            {
                lastOutput.Dispose();
            }

            lastOutput = output;

            // A pose session is pinned to the phenotype/local/face coefficients it was created
            // with. Any full parameter update therefore invalidates the previous session; rebuild
            // it only after the new evaluation succeeds so a failed Apply leaves the old state intact.
            if (Session != null)
            {
                Session.Dispose();
                Session = null;
            }
            if (usePoseSession)
            {
                Session = Runtime.CreateSession(parameters);
            }

            PushOutput(output);
        }

        /// <summary>
        /// Pose-only update through the native session. Falls back to a full evaluation when no
        /// session exists, which is the case when sliders changed since <see cref="Generate"/>.
        /// </summary>
        public void ApplyPose(string poseJson)
        {
            if (Runtime == null)
            {
                Generate();
                return;
            }

            if (Session == null || !usePoseSession)
            {
                Apply();
                return;
            }

            Stopwatch watch = Stopwatch.StartNew();
            Session.UpdatePoseJson(poseJson);
            watch.Stop();
            LastEvaluateMs = watch.Elapsed.TotalMilliseconds;
            Evaluations++;

            PushOutput(Session);
        }

        public void ApplyRestPose()
        {
            ApplyPose(null);
        }

        /// <summary>Updates the mesh and the bone hierarchy from an evaluation.</summary>
        public void PushOutput(IAnnyEvaluation evaluation)
        {
            AnnyTensorF32 posed = evaluation.Tensor(AnnyOutputF32.Vertices);
            lastVertices = posed.Data;

            if (mode == AnnyUpdateMode.Exact)
            {
                AnnyMeshBuilder.UpdatePosedVertices(GeneratedMesh, evaluation, CornerSource, true);
            }

            AnnySkeleton.ApplyPose(Rig, evaluation);

            // `Evaluations` is advanced by the entry points (Generate/Apply/ApplyPose) only, so one
            // update is one count no matter which path pushed it.
        }

        /// <summary>
        /// Unity's skinning uses at most <see cref="QualitySettings.skinWeights"/> bones per vertex,
        /// whatever the mesh carries. When the active quality level would drop influences the model
        /// assigns, the skinned result stops matching a native evaluation, and that is worth saying
        /// out loud rather than discovering as centimetres of drift.
        /// </summary>
        private void WarnIfQualityDropsInfluences()
        {
            if (mode != AnnyUpdateMode.Skinned || Report == null)
            {
                return;
            }

            int allowed;
            switch (QualitySettings.skinWeights)
            {
                case SkinWeights.OneBone:
                    allowed = 1;
                    break;
                case SkinWeights.TwoBones:
                    allowed = 2;
                    break;
                case SkinWeights.FourBones:
                    allowed = 4;
                    break;
                default:
                    allowed = int.MaxValue;
                    break;
            }

            if (Report.MaxInfluencesPerVertex > allowed)
            {
                UnityEngine.Debug.LogWarning(
                    name + ": the active quality level (\"" +
                    QualitySettings.names[QualitySettings.GetQualityLevel()] +
                    "\") allows " + allowed + " bone influences per vertex, but this model assigns " +
                    Report.MaxInfluencesPerVertex + ". The skinned mesh will not match a native " +
                    "evaluation; set QualitySettings.skinWeights to Unlimited, or use " +
                    "AnnyUpdateMode.Exact.", this);
            }
        }

        private void InstallRenderer()
        {
            if (mode == AnnyUpdateMode.Skinned)
            {
                filter = GetComponent<MeshFilter>();
                if (filter == null)
                {
                    filter = gameObject.AddComponent<MeshFilter>();
                }

                filter.sharedMesh = GeneratedMesh;
                Skinned = GetComponent<SkinnedMeshRenderer>();
                if (Skinned == null)
                {
                    Skinned = gameObject.AddComponent<SkinnedMeshRenderer>();
                }

                Skinned.sharedMesh = GeneratedMesh;
                Skinned.bones = Rig.Bones;
                Skinned.rootBone = Rig.Bones.Length > 0 ? Rig.Bones[0] : transform;
                Skinned.sharedMaterial = ResolveMaterial();
                Skinned.updateWhenOffscreen = true;
                Skinned.enabled = true;
                if (MeshRenderer != null)
                {
                    MeshRenderer.enabled = false;
                }
                return;
            }

            filter = GetComponent<MeshFilter>();
            if (filter == null)
            {
                filter = gameObject.AddComponent<MeshFilter>();
            }

            filter.sharedMesh = GeneratedMesh;
            MeshRenderer = GetComponent<MeshRenderer>();
            if (MeshRenderer == null)
            {
                MeshRenderer = gameObject.AddComponent<MeshRenderer>();
            }

            MeshRenderer.sharedMaterial = ResolveMaterial();
            MeshRenderer.enabled = true;
            if (Skinned != null)
            {
                Skinned.enabled = false;
            }
        }

        private Material ResolveMaterial()
        {
            if (material != null)
            {
                return material;
            }

            material = AnnyMaterials.CreateDefault();
            return material;
        }
    }
}