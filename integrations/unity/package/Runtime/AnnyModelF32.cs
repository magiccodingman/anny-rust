using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Anny
{
    /// <summary>
    /// One evaluated result from an f32 model. Coordinates and conventions match the f64 output;
    /// only the stored precision differs.
    /// </summary>
    public sealed class AnnyOutputF32 : IDisposable, IAnnyEvaluation
    {
        public const string Vertices = "vertices";
        public const string RestVertices = "rest_vertices";
        public const string BonePoses = "bone_poses";
        public const string RestBonePoses = "rest_bone_poses";
        public const string BoneHeads = "bone_heads";
        public const string BoneTails = "bone_tails";

        private readonly AnnyOutputF32Handle handle;
        private bool disposed;

        public AnnyOutputF32(AnnyOutputF32Handle output)
        {
            handle = output;
        }

        public AnnyOutputF32Handle Handle
        {
            get { return handle; }
        }

        public AnnyTensorF32 Tensor(string name)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.OutputF32Tensor(handle, name, out AnnyTensorViewF32 view));
            return AnnyTensorF32.FromView(name, view);
        }

        public bool HasTensor(string name)
        {
            ThrowIfDisposed();
            return AnnyNative.OutputF32Tensor(handle, name, out AnnyTensorViewF32 _) == AnnyNative.StatusOk;
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            handle.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(AnnyOutputF32));
            }
        }
    }

    /// <summary>
    /// A single-precision model: the model that a Unity build should carry when memory bandwidth
    /// matters more than the authored precision. It is produced from an f64 model by
    /// <see cref="AnnyModel.ToSinglePrecision"/> or loaded from pre-converted bytes.
    /// </summary>
    public sealed class AnnyModelF32 : IDisposable
    {
        private readonly AnnyModelF32Handle handle;
        private bool disposed;

        internal AnnyModelF32(AnnyModelF32Handle model, AnnyDescription description)
        {
            handle = model;
            Description = description;
        }

        /// <summary>The raw owning handle, for callers that need to reach the ABI directly.</summary>
        public AnnyModelF32Handle Handle
        {
            get { return handle; }
        }

        /// <summary>
        /// Typed model description. Present when this model came from an f64 model, which is where
        /// the description text is produced; null when it was loaded directly from f32 bytes.
        /// </summary>
        public AnnyDescription Description { get; private set; }

        /// <summary>Loads a prepared single-precision model from a file.</summary>
        public static AnnyModelF32 Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("no prepared f32 model at " + path, path);
            }

            return FromBytes(File.ReadAllBytes(path));
        }

        /// <summary>Loads a prepared single-precision model from memory.</summary>
        public static AnnyModelF32 FromBytes(byte[] payload)
        {
            if (payload == null)
            {
                throw new ArgumentNullException(nameof(payload));
            }

            AnnyNative.RequireAbi();
            AnnyNative.Check(AnnyNative.ModelF32FromBytes(
                payload, (UIntPtr)(ulong)payload.Length, null, out IntPtr model));
            return new AnnyModelF32(new AnnyModelF32Handle(model), null);
        }

        /// <summary>Evaluates this model's rest pose.</summary>
        public AnnyOutputF32 Evaluate()
        {
            return Evaluate(null);
        }

        /// <summary>Evaluates this model at the given parameters.</summary>
        public AnnyOutputF32 Evaluate(AnnyParameters parameters)
        {
            ThrowIfDisposed();
            string json = parameters != null ? parameters.ToJson() : "{}";
            AnnyNative.Check(AnnyNative.ModelF32Evaluate(handle, json, out IntPtr output));
            return new AnnyOutputF32(new AnnyOutputF32Handle(output));
        }

        /// <summary>Copies one named model array.</summary>
        public AnnyTensorF32 Tensor(string name)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelF32Tensor(handle, name, out AnnyTensorViewF32 view));
            return AnnyTensorF32.FromView(name, view);
        }

        /// <summary>True when the model carries the named array.</summary>
        public bool HasTensor(string name)
        {
            ThrowIfDisposed();
            return AnnyNative.ModelF32Tensor(handle, name, out AnnyTensorViewF32 _) == AnnyNative.StatusOk;
        }

        /// <summary>Serializes the prepared single-precision payload.</summary>
        public byte[] PreparedBytes()
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.ModelF32PreparedBytes(handle, out IntPtr bytes));
            return AnnyModel.CopyBytes(new AnnyBytesHandle(bytes));
        }

        /// <summary>
        /// Creates a pose session, which evaluates the parameter-dependent coefficients once and
        /// then re-poses without repeating that work.
        /// </summary>
        public AnnyPoseSessionF32 CreateSession(AnnyParameters parameters)
        {
            ThrowIfDisposed();
            string json = parameters != null ? parameters.ToJson() : "{}";
            AnnyNative.Check(AnnyNative.SessionF32New(handle, json, out IntPtr session));
            return new AnnyPoseSessionF32(this, new AnnySessionF32Handle(session));
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            handle.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(AnnyModelF32));
            }
        }
    }

    /// <summary>A reusable pose session over an f32 model.</summary>
    public sealed class AnnyPoseSessionF32 : IDisposable, IAnnyEvaluation
    {
        private readonly AnnyModelF32 owner;
        private readonly AnnySessionF32Handle handle;
        private bool disposed;

        internal AnnyPoseSessionF32(AnnyModelF32 owner, AnnySessionF32Handle session)
        {
            this.owner = owner;
            handle = session;
        }

        public AnnyModelF32 Model
        {
            get { return owner; }
        }

        public AnnySessionF32Handle Handle
        {
            get { return handle; }
        }

        public void UpdatePoseJson(string poseJson)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.SessionF32Update(handle, poseJson));
        }

        public void UpdatePose(AnnyTensor pose)
        {
            UpdatePoseJson(pose == null ? "null" : AnnyRepresentation.TensorToJson(pose));
        }

        public void UpdateNamedPose(System.Collections.Generic.Dictionary<string, UnityEngine.Matrix4x4> poses)
        {
            UpdatePoseJson(poses == null ? "null" : AnnyRepresentation.NamedPosesToJson(poses));
        }

        public void UpdateRest()
        {
            UpdatePoseJson("null");
        }

        public AnnyTensorF32 Tensor(string name)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.SessionF32Tensor(handle, name, out AnnyTensorViewF32 view));
            return AnnyTensorF32.FromView(name, view);
        }

        public bool HasTensor(string name)
        {
            ThrowIfDisposed();
            return AnnyNative.SessionF32Tensor(handle, name, out AnnyTensorViewF32 _) == AnnyNative.StatusOk;
        }

        public AnnyTensorF32 Coefficients()
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.SessionF32Coefficients(handle, out AnnyTensorViewF32 view));
            return AnnyTensorF32.FromView("coefficients", view);
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            handle.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(AnnyPoseSessionF32));
            }
        }
    }
}