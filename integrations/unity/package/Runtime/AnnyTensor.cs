using System;
using System.Runtime.InteropServices;

namespace Anny
{
    /// <summary>
    /// A managed copy of one f64 native tensor. Copies are taken eagerly because native views
    /// are borrowed: they are invalidated when the owning output/session handle is freed or,
    /// for session tensors, by the next session update.
    /// </summary>
    public sealed class AnnyTensor
    {
        public AnnyTensor(string name, int[] shape, AnnyTensorKind kind, double[] data)
        {
            Name = name;
            Shape = shape;
            Kind = kind;
            Data = data;
        }

        public string Name { get; private set; }

        /// <summary>Row-major element storage, in the tensor's own precision.</summary>
        public double[] Data { get; private set; }

        /// <summary>Row-major extents, outermost first.</summary>
        public int[] Shape { get; private set; }

        public AnnyTensorKind Kind { get; private set; }

        /// <summary>Total element count (product of <see cref="Shape"/>).</summary>
        public int Length
        {
            get { return Data.Length; }
        }

        public int Rank
        {
            get { return Shape.Length; }
        }

        public double this[int index]
        {
            get { return Data[index]; }
        }

        /// <summary>Locates a row-major element without bounds arithmetic at the call site.</summary>
        public int Offset(int i0, int i1)
        {
            return (i0 * Shape[1]) + i1;
        }

        public int Offset(int i0, int i1, int i2)
        {
            return ((i0 * Shape[1]) + i1) * Shape[2] + i2;
        }

        public double At(int i0, int i1)
        {
            return Data[Offset(i0, i1)];
        }

        public double At(int i0, int i1, int i2)
        {
            return Data[Offset(i0, i1, i2)];
        }

        /// <summary>Converts to floats, optionally reusing a caller-owned buffer of the same length.</summary>
        public float[] ToFloats(float[] reuse = null)
        {
            float[] target = reuse != null && reuse.Length == Data.Length ? reuse : new float[Data.Length];
            for (int i = 0; i < Data.Length; i++)
            {
                target[i] = (float)Data[i];
            }

            return target;
        }

        /// <summary>Converts exact-integer index data, rejecting anything the native kind forbids.</summary>
        public int[] ToIndices(int expectedWidth = 0)
        {
            if (Kind == AnnyTensorKind.Float)
            {
                throw new InvalidOperationException(Name + " is a float tensor, not an index tensor.");
            }

            if (expectedWidth > 0 && (Shape.Length != 2 || Shape[1] != expectedWidth))
            {
                throw new InvalidOperationException(
                    Name + " has shape " + Describe() + " but " + expectedWidth + " columns were expected.");
            }

            int[] result = new int[Data.Length];
            for (int i = 0; i < Data.Length; i++)
            {
                double v = Data[i];
                if (v < 0 || v > int.MaxValue || v != Math.Truncate(v))
                {
                    throw new InvalidOperationException(
                        Name + " contains " + v + " at " + i + ", which is not a nonnegative integer index.");
                }

                result[i] = (int)v;
            }

            return result;
        }

        public string Describe()
        {
            string shape = string.Empty;
            for (int i = 0; i < Shape.Length; i++)
            {
                shape += i == 0 ? Shape[i].ToString() : "x" + Shape[i];
            }

            return Name + " [" + shape + "] " + Kind;
        }

        /// <summary>Copies a borrowed native f64 view into managed storage.</summary>
        public static AnnyTensor FromView(string name, AnnyTensorView view)
        {
            int rank = checked((int)view.Rank.ToUInt64());
            int[] shape = new int[rank];
            int elements = rank == 0 ? 0 : 1;
            for (int i = 0; i < rank; i++)
            {
                // shape elements are size_t, so the stride is the pointer width, not 4.
                long extent = IntPtr.Size == 8
                    ? Marshal.ReadInt64(view.Shape, i * 8)
                    : Marshal.ReadInt32(view.Shape, i * 4);
                if (extent < 0 || extent > int.MaxValue)
                {
                    throw new InvalidOperationException(name + " has a non-representable extent at " + i);
                }

                shape[i] = (int)extent;
                elements *= (int)extent;
            }

            int reported = checked((int)view.Length.ToUInt64());
            if (reported != elements)
            {
                throw new InvalidOperationException(
                    name + " reports " + reported + " elements but its shape needs " + elements + ".");
            }

            double[] data = new double[elements];
            if (elements > 0)
            {
                Marshal.Copy(view.Data, data, 0, elements);
            }

            return new AnnyTensor(name, shape, (AnnyTensorKind)view.Kind, data);
        }
    }

    /// <summary>A managed copy of one f32 native tensor, which is the form Unity meshes want.</summary>
    public sealed class AnnyTensorF32
    {
        public AnnyTensorF32(string name, int[] shape, AnnyTensorKind kind, float[] data)
        {
            Name = name;
            Shape = shape;
            Kind = kind;
            Data = data;
        }

        public string Name { get; private set; }

        /// <summary>Row-major element storage, in the tensor's own precision.</summary>
        public float[] Data { get; private set; }

        public int[] Shape { get; private set; }

        public AnnyTensorKind Kind { get; private set; }

        public int Length
        {
            get { return Data.Length; }
        }

        public int Rank
        {
            get { return Shape.Length; }
        }

        public float this[int index]
        {
            get { return Data[index]; }
        }

        public int Offset(int i0, int i1)
        {
            return (i0 * Shape[1]) + i1;
        }

        public int Offset(int i0, int i1, int i2)
        {
            return ((i0 * Shape[1]) + i1) * Shape[2] + i2;
        }

        public float At(int i0, int i1)
        {
            return Data[Offset(i0, i1)];
        }

        public float At(int i0, int i1, int i2)
        {
            return Data[Offset(i0, i1, i2)];
        }

        public int[] ToIndices(int expectedWidth = 0)
        {
            if (Kind == AnnyTensorKind.Float)
            {
                throw new InvalidOperationException(Name + " is a float tensor, not an index tensor.");
            }

            if (expectedWidth > 0 && (Shape.Length != 2 || Shape[1] != expectedWidth))
            {
                throw new InvalidOperationException(
                    Name + " has shape " + Describe() + " but " + expectedWidth + " columns were expected.");
            }

            int[] result = new int[Data.Length];
            for (int i = 0; i < Data.Length; i++)
            {
                float v = Data[i];
                if (v < 0 || v > int.MaxValue || v != (float)Math.Truncate(v))
                {
                    throw new InvalidOperationException(
                        Name + " contains " + v + " at " + i + ", which is not a nonnegative integer index.");
                }

                result[i] = (int)v;
            }

            return result;
        }

        public string Describe()
        {
            string shape = string.Empty;
            for (int i = 0; i < Shape.Length; i++)
            {
                shape += i == 0 ? Shape[i].ToString() : "x" + Shape[i];
            }

            return Name + " [" + shape + "] " + Kind;
        }

        /// <summary>Copies a borrowed native f32 view into managed storage.</summary>
        public static AnnyTensorF32 FromView(string name, AnnyTensorViewF32 view)
        {
            int rank = checked((int)view.Rank.ToUInt64());
            int[] shape = new int[rank];
            int elements = rank == 0 ? 0 : 1;
            for (int i = 0; i < rank; i++)
            {
                // shape elements are size_t, so the stride is the pointer width, not 4.
                long extent = IntPtr.Size == 8
                    ? Marshal.ReadInt64(view.Shape, i * 8)
                    : Marshal.ReadInt32(view.Shape, i * 4);
                if (extent < 0 || extent > int.MaxValue)
                {
                    throw new InvalidOperationException(name + " has a non-representable extent at " + i);
                }

                shape[i] = (int)extent;
                elements *= (int)extent;
            }

            int reported = checked((int)view.Length.ToUInt64());
            if (reported != elements)
            {
                throw new InvalidOperationException(
                    name + " reports " + reported + " elements but its shape needs " + elements + ".");
            }

            float[] data = new float[elements];
            if (elements > 0)
            {
                Marshal.Copy(view.Data, data, 0, elements);
            }

            return new AnnyTensorF32(name, shape, (AnnyTensorKind)view.Kind, data);
        }
    }
}