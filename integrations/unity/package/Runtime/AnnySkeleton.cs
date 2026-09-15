using System.Collections.Generic;
using UnityEngine;

namespace Anny
{
    /// <summary>A built bone hierarchy: the transforms, their parent indices and their labels.</summary>
    public sealed class AnnyRig
    {
        public Transform Root;
        public Transform[] Bones;
        public int[] Parents;
        public string[] Labels;

        public int Count
        {
            get { return Bones == null ? 0 : Bones.Length; }
        }

        /// <summary>Index of a bone label, or -1 when it is absent.</summary>
        public int IndexOf(string label)
        {
            for (int b = 0; b < Labels.Length; b++)
            {
                if (Labels[b] == label)
                {
                    return b;
                }
            }

            return -1;
        }

        /// <summary>World-space position of a labelled joint.</summary>
        public Vector3 Position(string label)
        {
            int index = IndexOf(label);
            if (index < 0)
            {
                throw new KeyNotFoundException("no bone labelled " + label);
            }

            return Bones[index].position;
        }
    }

    /// <summary>
    /// Drives a Unity transform hierarchy from Anny's bone matrices.
    /// <para>
    /// The native arrays are <em>global</em> bone matrices: <c>rest_bone_poses</c> is the bind pose
    /// and <c>bone_poses</c> is the current pose, both row-major and in Anny's Z-up space. Unity
    /// wants local transforms, so each bone's local frame is the parent's global matrix inverted
    /// and multiplied by its own. Anny bone matrices are rigid (rotation and translation only),
    /// so decomposing straight to position and rotation is exact.
    /// </para>
    /// </summary>
    public static class AnnySkeleton
    {
        /// <summary>Builds a bone hierarchy at the model's bind pose.</summary>
        public static AnnyRig Build(
            AnnyDescription description, IAnnyEvaluation bindPose, Transform parent, string rootName = "anny_rig")
        {
            if (description == null)
            {
                throw new System.ArgumentNullException(nameof(description));
            }

            AnnyTensorF32 rest = bindPose.Tensor(AnnyOutputF32.RestBonePoses);
            int boneCount = description.bones;
            if (rest.Shape[1] != boneCount)
            {
                throw new System.InvalidOperationException(
                    "the bind pose carries " + rest.Shape[1] + " bones but the model describes " + boneCount);
            }

            double[] wideRest = new double[boneCount * 16];
            for (int i = 0; i < wideRest.Length; i++)
            {
                wideRest[i] = rest.Data[i];
            }

            GameObject root = new GameObject(rootName);
            root.transform.SetParent(parent, false);

            AnnyRig rig = new AnnyRig
            {
                Root = root.transform,
                Bones = new Transform[boneCount],
                Parents = new int[boneCount],
                Labels = description.bone_labels,
            };

            for (int b = 0; b < boneCount; b++)
            {
                rig.Bones[b] = new GameObject(description.bone_labels[b]).transform;
            }

            for (int b = 0; b < boneCount; b++)
            {
                int parentIndex = description.bone_parents != null && b < description.bone_parents.Length
                    ? description.bone_parents[b]
                    : -1;
                rig.Parents[b] = parentIndex >= 0 && parentIndex < boneCount ? parentIndex : -1;
                Transform host = rig.Parents[b] >= 0 ? rig.Bones[rig.Parents[b]] : root.transform;
                rig.Bones[b].SetParent(host, false);
            }

            ApplyGlobalMatrices(rig, wideRest, boneCount);
            return rig;
        }

        /// <summary>Moves the hierarchy onto the posed bone matrices of an f64 output.</summary>
        public static void ApplyPose(AnnyRig rig, AnnyOutput output)
        {
            if (rig == null)
            {
                throw new System.ArgumentNullException(nameof(rig));
            }

            AnnyTensor pose = output.Tensor(AnnyOutput.BonePoses);
            ApplyGlobalMatrices(rig, pose.Data, Mathf.Min(rig.Count, pose.Shape[1]));
        }

        /// <summary>Moves the hierarchy onto the posed bone matrices of an f32 output.</summary>
        public static void ApplyPose(AnnyRig rig, AnnyOutputF32 output)
        {
            if (rig == null)
            {
                throw new System.ArgumentNullException(nameof(rig));
            }

            AnnyTensorF32 pose = output.Tensor(AnnyOutputF32.BonePoses);
            int count = Mathf.Min(rig.Count, pose.Shape[1]);
            double[] wide = new double[count * 16];
            for (int i = 0; i < wide.Length; i++)
            {
                wide[i] = pose.Data[i];
            }

            ApplyGlobalMatrices(rig, wide, count);
        }

        /// <summary>
        /// Same as the typed overloads, for any evaluation source — including a pose session, whose
        /// bone matrices are borrowed from the session.
        /// </summary>
        public static void ApplyPose(AnnyRig rig, IAnnyEvaluation evaluation)
        {
            if (rig == null)
            {
                throw new System.ArgumentNullException(nameof(rig));
            }

            AnnyTensorF32 pose = evaluation.Tensor(AnnyOutputF32.BonePoses);
            int count = Mathf.Min(rig.Count, pose.Shape[1]);
            double[] wide = new double[count * 16];
            for (int i = 0; i < wide.Length; i++)
            {
                wide[i] = pose.Data[i];
            }

            ApplyGlobalMatrices(rig, wide, count);
        }

        /// <summary>
        /// Converts a batch of global bone matrices into Unity local transforms. Bones are stored
        /// parents-before-children, so one pass suffices.
        /// </summary>
        public static void ApplyGlobalMatrices(AnnyRig rig, double[] matrices, int count)
        {
            if (rig == null)
            {
                throw new System.ArgumentNullException(nameof(rig));
            }

            if (matrices == null)
            {
                throw new System.ArgumentNullException(nameof(matrices));
            }

            count = Mathf.Min(count, rig.Count);
            Matrix4x4[] global = new Matrix4x4[count];
            for (int b = 0; b < count; b++)
            {
                global[b] = AnnyRepresentation.ToUnityMatrix(matrices, b * 16);
            }

            for (int b = 0; b < count; b++)
            {
                Matrix4x4 local = global[b];
                int parentIndex = rig.Parents[b];
                if (parentIndex >= 0 && parentIndex < count)
                {
                    local = global[parentIndex].inverse * global[b];
                }

                rig.Bones[b].localPosition = local.GetColumn(3);
                rig.Bones[b].localRotation = local.rotation;
                rig.Bones[b].localScale = Vector3.one;
            }
        }

        /// <summary>
        /// Reads the current Unity pose back as named world matrices, in Anny space, which is the
        /// form the native pose parser accepts for the <c>world</c> parameterization.
        /// </summary>
        public static Dictionary<string, Matrix4x4> CaptureAnnyWorldPose(AnnyRig rig)
        {
            Dictionary<string, Matrix4x4> captured = new Dictionary<string, Matrix4x4>(rig.Count);
            for (int b = 0; b < rig.Count; b++)
            {
                Matrix4x4 unityWorld = rig.Bones[b].localToWorldMatrix;
                Matrix4x4 anny = AnnyRepresentation.ToNativeMatrixContainer(unityWorld);
                captured[rig.Labels[b]] = anny;
            }

            return captured;
        }
    }
}