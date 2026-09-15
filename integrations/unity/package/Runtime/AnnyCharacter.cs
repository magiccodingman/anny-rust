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

        [Tooltip("Runtime precision. The f64 model is the authority; f32 is the shipped default.")]
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

        public int Evaluations { get; private set; }

        /// <summary>Wall-clock milliseconds spent in the most recent native evaluation.</summary>
        public double LastEvaluateMs { get; private set; }

        private AnnyOutputF32 lastOutput;
        private MeshFilter filter;

        private void Awake()
        {
            if (generateOnAwake)
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

            AnnyMeshResult built = AnnyMeshBuilder.Build(runtime, lastOutput, meshOptions);
            GeneratedMesh = built.Mesh;
            Report = built.Report;
            CornerSource = built.CornerSource;
            GeneratedMesh.name = name + "-anny";

            Rig = AnnySkeleton.Build(modelAsset.Description, lastOutput, transform);
            InstallRenderer();

            if (usePoseSession)
            {
                Session = runtime.CreateSession(parameters);
            }
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
            if (mode == AnnyUpdateMode.Exact)
            {
                AnnyMeshBuilder.UpdatePosedVertices(GeneratedMesh, evaluation, CornerSource, true);
            }

            AnnySkeleton.ApplyPose(Rig, evaluation);
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