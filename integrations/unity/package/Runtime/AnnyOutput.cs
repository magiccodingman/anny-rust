using System;
using System.Runtime.InteropServices;

namespace Anny
{
    /// <summary>
    /// One evaluated result. The handle owns its arrays and is independent of the model that
    /// produced it, so it may outlive the model and may be read on another thread.
    /// </summary>
    public sealed class AnnyOutput : IDisposable
    {
        /// <summary>Rest-pose vertex positions, shape <c>[B, vertices, 3]</c>.</summary>
        public const string Vertices = "vertices";

        /// <summary>Posed vertex positions, shape <c>[B, vertices, 3]</c>.</summary>
        public const string RestVertices = "rest_vertices";

        /// <summary>Bone matrices, shape <c>[B, bones, 4, 4]</c>, row-major.</summary>
        public const string BonePoses = "bone_poses";

        /// <summary>Bind-pose bone matrices, shape <c>[B, bones, 4, 4]</c>, row-major.</summary>
        public const string RestBonePoses = "rest_bone_poses";

        /// <summary>Bone heads, present only when <c>return_bone_ends</c> was requested.</summary>
        public const string BoneHeads = "bone_heads";

        /// <summary>Bone tails, present only when <c>return_bone_ends</c> was requested.</summary>
        public const string BoneTails = "bone_tails";

        private readonly AnnyOutputHandle handle;
        private bool disposed;

        public AnnyOutput(AnnyOutputHandle output)
        {
            handle = output;
        }

        /// <summary>The raw owning handle, for callers that need to reach the ABI directly.</summary>
        public AnnyOutputHandle Handle
        {
            get { return handle; }
        }

        /// <summary>Copies one named array of this result.</summary>
        public AnnyTensor Tensor(string name)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.OutputTensor(handle, name, out AnnyTensorView view));
            return AnnyTensor.FromView(name, view);
        }

        /// <summary>True when this result carries the named array.</summary>
        public bool HasTensor(string name)
        {
            ThrowIfDisposed();
            return AnnyNative.OutputTensor(handle, name, out AnnyTensorView _) == AnnyNative.StatusOk;
        }

        /// <summary>The posed vertex positions of batch 0 as a flat <c>[vertices * 3]</c> array.</summary>
        public double[] VertexData(int batch = 0)
        {
            return Tensor(Vertices).Data;
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
                throw new ObjectDisposedException(nameof(AnnyOutput));
            }
        }
    }

    /// <summary>
    /// A reusable pose session. Creating it evaluates the phenotype/local-change/facial
    /// coefficients and the rest model once, so every update pays only for the pose-dependent
    /// half; the result is numerically identical to a full evaluation of the same parameters.
    /// The session keeps its own native reference to the model, so the model handle may be
    /// disposed first.
    /// </summary>
    public sealed class AnnyPoseSession : IDisposable
    {
        private readonly AnnyModel owner;
        private readonly AnnySessionHandle handle;
        private bool disposed;

        internal AnnyPoseSession(AnnyModel owner, AnnySessionHandle session)
        {
            this.owner = owner;
            handle = session;
        }

        /// <summary>The model this session was created from; kept alive while the session lives.</summary>
        public AnnyModel Model
        {
            get { return owner; }
        }

        /// <summary>The raw owning handle, for callers that need to reach the ABI directly.</summary>
        public AnnySessionHandle Handle
        {
            get { return handle; }
        }

        /// <summary>Re-poses with a bare pose tensor of shape <c>[B, bones, 4, 4]</c> or <c>[bones, 4, 4]</c>.</summary>
        public void UpdatePose(AnnyTensor pose)
        {
            if (pose == null)
            {
                UpdateRest();
                return;
            }

            UpdatePoseJson(AnnyRepresentation.TensorToJson(pose));
        }

        /// <summary>
        /// Re-poses with a raw pose JSON value: either a nested tensor of shape
        /// <c>[B, bones, 4, 4]</c>, or an object naming bones (<c>{"bone": [4,4]}</c>), or
        /// <c>null</c> for the rest pose.
        /// </summary>
        public void UpdatePoseJson(string poseJson)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.SessionUpdate(handle, poseJson));
        }

        /// <summary>Updates only the named bones, leaving the rest at identity.</summary>
        public void UpdateNamedPose(System.Collections.Generic.Dictionary<string, UnityEngine.Matrix4x4> poses)
        {
            if (poses == null)
            {
                UpdateRest();
                return;
            }

            UpdatePoseJson(AnnyRepresentation.NamedPosesToJson(poses));
        }

        /// <summary>Returns the session to the rest pose.</summary>
        public void UpdateRest()
        {
            UpdatePoseJson("null");
        }

        /// <summary>
        /// Copies one array of the most recent pose. Copies taken before the next update stay
        /// valid, because every access makes a managed copy.
        /// </summary>
        public AnnyTensor Tensor(string name)
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.SessionTensor(handle, name, out AnnyTensorView view));
            return AnnyTensor.FromView(name, view);
        }

        /// <summary>True when the session's current output carries the named array.</summary>
        public bool HasTensor(string name)
        {
            ThrowIfDisposed();
            return AnnyNative.SessionTensor(handle, name, out AnnyTensorView _) == AnnyNative.StatusOk;
        }

        /// <summary>Copies the coefficients the session was created with.</summary>
        public AnnyTensor Coefficients()
        {
            ThrowIfDisposed();
            AnnyNative.Check(AnnyNative.SessionCoefficients(handle, out AnnyTensorView view));
            return AnnyTensor.FromView("coefficients", view);
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
                throw new ObjectDisposedException(nameof(AnnyPoseSession));
            }
        }
    }
}