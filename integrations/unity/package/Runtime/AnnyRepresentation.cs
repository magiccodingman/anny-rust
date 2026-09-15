using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace Anny
{
    /// <summary>
    /// Converts between the native Anny representation and Unity's.
    /// <para>
    /// Anny coordinates are Z-up right-handed meters; Unity is Y-up left-handed meters. The
    /// mapping is <c>unity = (x, z, -y)</c>, a reflection, so triangle winding must be reversed
    /// to keep faces outward-facing. The same basis change conjugated onto a matrix — <c>C M Cᵀ</c>
    /// for the orthogonal <c>C</c> above — is how bone matrices move between the two spaces.
    /// </para>
    /// </summary>
    public static class AnnyRepresentation
    {
        /// <summary>Parameterization names accepted by the native pose parser.</summary>
        public static class Parameterization
        {
            public const string World = "world";
            public const string LocalBoneWorld = "local_bone_world";
            public const string LocalBone = "local_bone";
            public const string LocalRef = "local_ref";
            public const string WorldOrient = "world_orient";
        }

        /// <summary>Maps one native Z-up position into Unity space.</summary>
        public static Vector3 ToUnityPosition(double x, double y, double z)
        {
            return new Vector3((float)x, (float)z, (float)-y);
        }

        /// <summary>Maps one Unity position back into native Z-up space.</summary>
        public static Vector3 ToNativePosition(Vector3 unity)
        {
            return new Vector3(unity.x, -unity.z, unity.y);
        }

        /// <summary>
        /// Builds the Unity matrix for a native row-major 4x4 starting at <paramref name="offset"/>,
        /// applying the Anny-to-Unity basis change on both sides.
        /// </summary>
        public static Matrix4x4 ToUnityMatrix(double[] rowMajor, int offset = 0)
        {
            if (rowMajor == null)
            {
                throw new System.ArgumentNullException(nameof(rowMajor));
            }

            if (offset < 0 || offset + 16 > rowMajor.Length)
            {
                throw new System.ArgumentException(
                    "offset " + offset + " does not leave 16 elements in the matrix buffer", nameof(offset));
            }

            // Source matrix M in Anny space, row-major.
            double m00 = rowMajor[offset], m01 = rowMajor[offset + 1], m02 = rowMajor[offset + 2], m03 = rowMajor[offset + 3];
            double m10 = rowMajor[offset + 4], m11 = rowMajor[offset + 5], m12 = rowMajor[offset + 6], m13 = rowMajor[offset + 7];
            double m20 = rowMajor[offset + 8], m21 = rowMajor[offset + 9], m22 = rowMajor[offset + 10], m23 = rowMajor[offset + 11];
            double m30 = rowMajor[offset + 12], m31 = rowMajor[offset + 13], m32 = rowMajor[offset + 14], m33 = rowMajor[offset + 15];

            // C M Cᵀ with C = row0 (1,0,0), row1 (0,0,1), row2 (0,-1,0):
            //   result row 0 = M row 0
            //   result row 1 = M row 2
            //   result row 2 = -M row 1
            // and the same permutation applied to the columns.
            Matrix4x4 result = new Matrix4x4();
            result[0, 0] = (float)m00; result[0, 1] = (float)m02; result[0, 2] = (float)-m01; result[0, 3] = (float)m03;
            result[1, 0] = (float)m20; result[1, 1] = (float)m22; result[1, 2] = (float)-m21; result[1, 3] = (float)m23;
            result[2, 0] = (float)-m10; result[2, 1] = (float)-m12; result[2, 2] = (float)m11; result[2, 3] = (float)-m13;
            result[3, 0] = (float)m30; result[3, 1] = (float)m32; result[3, 2] = (float)-m31; result[3, 3] = (float)m33;
            return result;
        }

        /// <summary>Maps one native Z-up direction (no translation) into Unity space.</summary>
        public static Vector3 ToUnityDirection(double x, double y, double z)
        {
            return new Vector3((float)x, (float)z, (float)-y);
        }

        /// <summary>Converts a Unity matrix back into native row-major Z-up space.</summary>
        public static double[] ToNativeMatrix(Matrix4x4 unity)
        {
            // Same reflection, applied again: Cᵀ M C. The matrix is its own inverse basis swap.
            double[] result = new double[16];
            double m00 = unity[0, 0], m01 = unity[0, 1], m02 = unity[0, 2], m03 = unity[0, 3];
            double m10 = unity[1, 0], m11 = unity[1, 1], m12 = unity[1, 2], m13 = unity[1, 3];
            double m20 = unity[2, 0], m21 = unity[2, 1], m22 = unity[2, 2], m23 = unity[2, 3];
            double m30 = unity[3, 0], m31 = unity[3, 1], m32 = unity[3, 2], m33 = unity[3, 3];
            result[0] = m00; result[1] = -m02; result[2] = m01; result[3] = m03;
            result[4] = -m20; result[5] = m22; result[6] = -m21; result[7] = -m23;
            result[8] = m10; result[9] = -m12; result[10] = m11; result[11] = m13;
            result[12] = m30; result[13] = -m32; result[14] = m31; result[15] = m33;
            return result;
        }

        /// <summary>
        /// Same conversion as <see cref="ToNativeMatrix"/>, kept as a <see cref="Matrix4x4"/> so it
        /// can be handed to <see cref="NamedPosesToJson"/>. Remember this value is laid out in Anny
        /// space; treat it as data, not as something to apply to a Unity transform.
        /// </summary>
        public static Matrix4x4 ToNativeMatrixContainer(Matrix4x4 unity)
        {
            double[] native = ToNativeMatrix(unity);
            Matrix4x4 result = new Matrix4x4();
            for (int row = 0; row < 4; row++)
            {
                for (int column = 0; column < 4; column++)
                {
                    result[row, column] = (float)native[(row * 4) + column];
                }
            }

            return result;
        }

        /// <summary>Serializes a tensor as nested row-major JSON arrays.</summary>
        public static string TensorToJson(AnnyTensor tensor)
        {
            StringBuilder builder = new StringBuilder(64);
            AnnyJsonWriter.AppendTensor(builder, tensor);
            return builder.ToString();
        }

        /// <summary>Serializes named bone matrices as the object form the native pose parser accepts.</summary>
        public static string NamedPosesToJson(Dictionary<string, Matrix4x4> poses)
        {
            StringBuilder builder = new StringBuilder(256);
            builder.Append('{');
            bool first = true;
            foreach (KeyValuePair<string, Matrix4x4> entry in poses)
            {
                if (!first)
                {
                    builder.Append(',');
                }

                first = false;
                AnnyJsonWriter.AppendString(builder, entry.Key);
                builder.Append(':');
                Matrix4x4 m = entry.Value;
                builder.Append('[');
                for (int row = 0; row < 4; row++)
                {
                    if (row > 0)
                    {
                        builder.Append(',');
                    }

                    builder.Append('[');
                    for (int column = 0; column < 4; column++)
                    {
                        if (column > 0)
                        {
                            builder.Append(',');
                        }

                        AnnyJsonWriter.AppendNumber(builder, m[row, column]);
                    }

                    builder.Append(']');
                }

                builder.Append(']');
            }

            builder.Append('}');
            return builder.ToString();
        }
    }
}