using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Anny
{
    /// <summary>Typed view of the JSON returned by <c>anny_model_describe</c>.</summary>
    [Serializable]
    public sealed class AnnyDescription
    {
        public int vertices;
        public int faces;
        public int bones;
        public int blendshapes;
        /// <summary>Phenotype slider labels, in native order.</summary>
        public string[] PhenotypeLabels()
        {
            return phenotype_labels ?? new string[0];
        }

        /// <summary>Local-change slider labels, in native order.</summary>
        public string[] LocalChangeLabels()
        {
            return local_change_labels ?? new string[0];
        }

        /// <summary>Whether a slider name addresses a local change rather than a phenotype axis.</summary>
        public bool IsLocalChange(string sliderName)
        {
            string[] labels = local_change_labels;
            if (labels == null)
            {
                return false;
            }

            for (int i = 0; i < labels.Length; i++)
            {
                if (labels[i] == sliderName)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>Every slider label the model exposes: phenotype axes, then local changes.</summary>
        public string[] SliderLabels()
        {
            string[] first = PhenotypeLabels();
            string[] second = LocalChangeLabels();
            string[] all = new string[first.Length + second.Length];
            System.Array.Copy(first, all, first.Length);
            System.Array.Copy(second, 0, all, first.Length, second.Length);
            return all;
        }

        public string[] bone_labels;
        public int[] bone_parents;
        public string[] phenotype_labels;
        public string[] local_change_labels;
        public string[] facial_action_labels;
        public string[] blend_shape_labels;
        public string[] blendshape_labels;

        /// <summary>
        /// Morph-target labels under either spelling the description may use, so callers do not
        /// have to care which key the native side emitted.
        /// </summary>
        public string[] BlendShapeLabels
        {
            get { return blend_shape_labels ?? blendshape_labels; }
        }

        /// <summary>Index of a bone label, or -1 when it is absent.</summary>
        public int BoneIndex(string label)
        {
            if (bone_labels == null)
            {
                return -1;
            }

            for (int i = 0; i < bone_labels.Length; i++)
            {
                if (bone_labels[i] == label)
                {
                    return i;
                }
            }

            return -1;
        }
    }

    /// <summary>
    /// Owns one native Anny model and its independent output/session handles. A model is
    /// immutable after creation, so several evaluations may be in flight at once; only
    /// <see cref="Dispose"/> has to be synchronized with them.
    /// </summary>
    public sealed class AnnyModel : IDisposable
    {
        private readonly AnnyModelHandle handle;
        private readonly object gate = new object();
        private bool disposed;
        private AnnyDescription description;

        private AnnyModel(AnnyModelHandle model)
        {
            handle = model;
        }

        /// <summary>The raw owning handle, for callers that need to reach the ABI directly.</summary>
        public AnnyModelHandle Handle
        {
            get { return handle; }
        }

        /// <summary>Element counts and labels, parsed from the native description once.</summary>
        public AnnyDescription Description
        {
            get
            {
                if (description == null)
                {
                    description = JsonUtility.FromJson<AnnyDescription>(DescribeJson());
                }

                return description;
            }
        }

        /// <summary>Loads a prepared model payload (safetensors) written by the CLI's <c>prepare</c>.</summary>
        public static AnnyModel Load(string path, string configJson = null)
        {
            ThrowIfBlankPath(path, "path");
            AnnyNative.RequireAbi();
            try
            {
                AnnyNative.Check(AnnyNative.ModelLoad(path, configJson, out IntPtr pointer));
                return new AnnyModel(new AnnyModelHandle(pointer));
            }
            catch (AnnyNativeException error)
            {
                // The native message is the OS error without the path; a caller chasing a missing
                // file needs to know which one it asked for.
                throw new AnnyNativeException(
                    error.Status, error.Message + " (while loading " + path + ")");
            }
        }

        /// <summary>Loads a prepared model payload from memory.</summary>
        public static AnnyModel FromBytes(byte[] bytes, string configJson = null)
        {
            if (bytes == null || bytes.Length == 0)
            {
                throw new ArgumentException("bytes must hold a prepared anny model", nameof(bytes));
            }

            AnnyNative.RequireAbi();
            AnnyNative.Check(AnnyNative.ModelFromBytes(bytes, (UIntPtr)(ulong)bytes.Length, configJson, out IntPtr pointer));
            return new AnnyModel(new AnnyModelHandle(pointer));
        }

        /// <summary>Builds a model from an asset store directory, reusing a cache directory when given.</summary>
        public static AnnyModel Build(string assetsDirectory, string configJson = null, string cacheDirectory = null)
        {
            ThrowIfBlankPath(assetsDirectory, "assetsDirectory");
            AnnyNative.RequireAbi();
            IntPtr pointer;
            if (cacheDirectory == null)
            {
                AnnyNative.Check(AnnyNative.ModelBuild(assetsDirectory, configJson, out pointer));
            }
            else
            {
                AnnyNative.Check(AnnyNative.ModelBuildCached(assetsDirectory, configJson, cacheDirectory, out pointer));
            }

            return new AnnyModel(new AnnyModelHandle(pointer));
        }

        /// <summary>The native description JSON, including the resolved config this layer does not mirror.</summary>
        public string DescribeJson()
        {
            lock (gate)
            {
                ThrowIfDisposed();
                AnnyNative.Check(AnnyNative.ModelDescribe(handle, out IntPtr text));
                return TakeString(text);
            }
        }

        /// <summary>Evaluates the model with the given parameters, returning an independent output.</summary>
        public AnnyOutput Evaluate(AnnyParameters parameters = null)
        {
            ThrowIfDisposed();
            string json = parameters == null ? "{}" : parameters.ToJson();
            AnnyNative.Check(AnnyNative.ModelEvaluate(handle, json, out IntPtr pointer));
            return new AnnyOutput(new AnnyOutputHandle(pointer));
        }

        /// <summary>Copies one static model array, e.g. <c>faces</c> or <c>vertex_bone_weights</c>.</summary>
        public AnnyTensor Tensor(string name)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelTensor(handle, name, out AnnyTensorView view));
            return AnnyTensor.FromView(name, view);
        }

        /// <summary>True when the model carries the named static array.</summary>
        public bool HasTensor(string name)
        {
            ThrowIfDisposed();
            return AnnyNative.ModelTensor(handle, name, out AnnyTensorView _) == AnnyNative.StatusOk;
        }

        /// <summary>Starts a reusable pose session; the fixed half of the evaluation is computed once.</summary>
        public AnnyPoseSession CreateSession(AnnyParameters parameters = null)
        {
            ThrowIfDisposed();
            string json = parameters == null ? "{}" : parameters.ToJson();
            AnnyNative.Check(AnnyNative.SessionNew(handle, json, out IntPtr pointer));
            return new AnnyPoseSession(this, new AnnySessionHandle(pointer));
        }

        /// <summary>Exports a standalone GLB (Y-up, f32 attributes) for the given parameters.</summary>
        public byte[] ExportGlb(AnnyParameters parameters = null, string optionsJson = null)
        {
            ThrowIfDisposed();
            string json = parameters == null ? "{}" : parameters.ToJson();
            AnnyNative.Check(AnnyNative.ModelExportGlb(handle, json, optionsJson, out IntPtr pointer));
            using (AnnyBytesHandle bytes = new AnnyBytesHandle(pointer))
            {
                return CopyBytes(bytes);
            }
        }

        /// <summary>The model's own serialized payload, byte-reproducible across processes.</summary>
        public byte[] PreparedBytes()
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelPreparedBytes(handle, out IntPtr pointer));
            using (AnnyBytesHandle bytes = new AnnyBytesHandle(pointer))
            {
                return CopyBytes(bytes);
            }
        }

        /// <summary>Runs a documented authoring request; see docs/AUTHORING.md.</summary>
        public string Query(string requestJson)
        {
            lock (gate)
            {
                ThrowIfDisposed();
                AnnyNative.Check(AnnyNative.ModelQuery(handle, requestJson, out IntPtr text));
                return TakeString(text);
            }
        }

        /// <summary>Applies documented model transform operations, returning a new independent model.</summary>
        public AnnyModel Transform(string operationsJson)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelTransform(handle, operationsJson, out IntPtr pointer));
            return new AnnyModel(new AnnyModelHandle(pointer));
        }

        /// <summary>Re-expresses a pose from this model onto another; returns the request JSON.</summary>
        public string TransferPose(AnnyModel target, AnnyParameters parameters, string mode = "local_ref")
        {
            if (target == null)
            {
                throw new ArgumentNullException(nameof(target));
            }

            ThrowIfDisposed();
            target.ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelTransferPose(handle, target.handle, parameters.ToJson(), mode, out IntPtr text));
            return TakeString(text);
        }

        /// <summary>Converts once into the single-precision runtime used for per-frame Unity work.</summary>
        public AnnyModelF32 ToSinglePrecision()
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelToF32(handle, out IntPtr pointer));
            return new AnnyModelF32(new AnnyModelF32Handle(pointer), Description);
        }

        public void Dispose()
        {
            lock (gate)
            {
                if (disposed)
                {
                    return;
                }

                disposed = true;
                handle.Dispose();
            }
        }

        internal void ThrowIfDisposed()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(AnnyModel));
            }
        }

        internal static byte[] CopyBytes(AnnyBytesHandle bytes)
        {
            int count = checked((int)AnnyNative.BytesLength(bytes).ToUInt64());
            byte[] result = new byte[count];
            if (count > 0)
            {
                Marshal.Copy(AnnyNative.BytesData(bytes), result, 0, count);
            }

            return result;
        }

        private static string TakeString(IntPtr pointer)
        {
            try
            {
                return pointer == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(pointer) ?? string.Empty;
            }
            finally
            {
                if (pointer != IntPtr.Zero)
                {
                    AnnyNative.StringFree(pointer);
                }
            }
        }

        private static void ThrowIfBlankPath(string value, string name)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new ArgumentException(name + " must be a non-empty path", name);
            }
        }
    }
}