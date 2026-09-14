using System.Runtime.InteropServices;

namespace AnnyExample;

public sealed record AnnyMeshF32(float[] Vertices, int[] Faces, int FaceSize);

/// <summary>A native f32 model with independently owned data. All returned arrays are managed copies.</summary>
public sealed class AnnySinglePrecisionModel : IDisposable
{
    private readonly ModelHandleF32 model;
    private readonly object gate = new();
    private bool disposed;
    internal AnnySinglePrecisionModel(IntPtr pointer) => model = new ModelHandleF32(pointer);
    public AnnySinglePrecisionModel(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Check(NativeF32.FromBytes(bytes, checked((nuint)bytes.LongLength), null, out var pointer));
        model = new ModelHandleF32(pointer);
    }
    public AnnyMeshF32 Generate(string parametersJson = "{}")
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(NativeF32.Evaluate(model, parametersJson, out var pointer));
            using var output = new OutputHandleF32(pointer);
            Check(NativeF32.OutputTensor(output, "vertices", out var vertices));
            Check(NativeF32.ModelTensor(model, "faces", out var faces));
            var indices = Copy(faces).Select(x =>
            {
                if (!float.IsFinite(x) || x < 0 || x >= 2147483648f || x != MathF.Truncate(x))
                    throw new InvalidDataException("Invalid native index.");
                return checked((int)x);
            }).ToArray();
            int width = checked((int)Marshal.ReadIntPtr(faces.Shape, IntPtr.Size));
            return new AnnyMeshF32(Copy(vertices), indices, width);
        }
    }
    public byte[] SavePrepared()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(NativeF32.Prepared(model, out var pointer));
            using var bytes = new BytesHandle(pointer);
            int count = checked((int)Native.BytesLength(bytes));
            var result = new byte[count];
            if (count != 0) Marshal.Copy(Native.BytesData(bytes), result, 0, count);
            return result;
        }
    }
    private static float[] Copy(TensorView view)
    {
        int count = checked((int)view.Length);
        var result = new float[count];
        if (count != 0) Marshal.Copy(view.Data, result, 0, count);
        return result;
    }
    private static void Check(int status)
    {
        if (status != 0) throw new InvalidOperationException(Marshal.PtrToStringUTF8(Native.LastError()) ?? "Native f32 error.");
    }
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            model.Dispose();
        }
    }
}
internal sealed class ModelHandleF32 : SafeHandle
{
    internal ModelHandleF32(IntPtr value) : base(IntPtr.Zero, true) => SetHandle(value);
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle() { NativeF32.FreeModel(handle); return true; }
}
internal sealed class OutputHandleF32 : SafeHandle
{
    internal OutputHandleF32(IntPtr value) : base(IntPtr.Zero, true) => SetHandle(value);
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle() { NativeF32.FreeOutput(handle); return true; }
}
internal static class NativeF32
{
    private const string Library = "anny_capi";
    [DllImport(Library, EntryPoint="anny_model_to_f32", CallingConvention=CallingConvention.Cdecl)]
    internal static extern int Convert(ModelHandle model, out IntPtr result);
    [DllImport(Library, EntryPoint="anny_model_f32_from_bytes", CallingConvention=CallingConvention.Cdecl)]
    internal static extern int FromBytes(byte[] bytes, nuint count, [MarshalAs(UnmanagedType.LPUTF8Str)] string? config, out IntPtr result);
    [DllImport(Library, EntryPoint="anny_model_f32_evaluate", CallingConvention=CallingConvention.Cdecl)]
    internal static extern int Evaluate(ModelHandleF32 model, [MarshalAs(UnmanagedType.LPUTF8Str)] string parameters, out IntPtr result);
    [DllImport(Library, EntryPoint="anny_model_f32_tensor", CallingConvention=CallingConvention.Cdecl)]
    internal static extern int ModelTensor(ModelHandleF32 model, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out TensorView view);
    [DllImport(Library, EntryPoint="anny_output_f32_tensor", CallingConvention=CallingConvention.Cdecl)]
    internal static extern int OutputTensor(OutputHandleF32 output, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out TensorView view);
    [DllImport(Library, EntryPoint="anny_model_f32_prepared_bytes", CallingConvention=CallingConvention.Cdecl)]
    internal static extern int Prepared(ModelHandleF32 model, out IntPtr bytes);
    [DllImport(Library, EntryPoint="anny_model_f32_free", CallingConvention=CallingConvention.Cdecl)]
    internal static extern void FreeModel(IntPtr model);
    [DllImport(Library, EntryPoint="anny_output_f32_free", CallingConvention=CallingConvention.Cdecl)]
    internal static extern void FreeOutput(IntPtr output);
}
