using System.Runtime.InteropServices;

namespace AnnyExample;

public sealed record AnnyMesh(double[] Vertices, int[] Faces, int FaceSize);

/// <summary>Owns one native model. All returned arrays are managed copies.</summary>
public sealed class AnnyModel : IDisposable
{
    private static long nextId;
    private readonly long id = Interlocked.Increment(ref nextId);
    private readonly ModelHandle model;
    private AnnyModel(ModelHandle handle) => model = handle;
    private readonly object gate = new();
    private bool disposed;

    public AnnyModel(string preparedModelPath)
    {
        Check(Native.Load(preparedModelPath, null, out var pointer));
        model = new ModelHandle(pointer);
    }

    public static AnnyModel FromAssets(string path, string configJson = "{}", string? cacheDirectory = null)
    {
        Check(Native.BuildCached(path, configJson, cacheDirectory, out var pointer));
        return new AnnyModel(new ModelHandle(pointer));
    }

    public string Describe()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.Describe(model, out var value));
            try { return Marshal.PtrToStringUTF8(value) ?? "{}"; }
            finally { Native.FreeString(value); }
        }
    }

    public AnnyMesh Generate(string parametersJson = "{}")
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.Evaluate(model, parametersJson, out var pointer));
            using var output = new OutputHandle(pointer);
            Check(Native.OutputTensor(output, "vertices", out var vertices));
            Check(Native.ModelTensor(model, "faces", out var faces));
            var vertexData = Copy(vertices);
            var faceData = Copy(faces).Select(x =>
            {
                if (x < 0 || x > int.MaxValue || x != Math.Truncate(x))
                    throw new InvalidDataException("Native mesh has an invalid index.");
                return checked((int)x);
            }).ToArray();
            var faceSize = checked((int)Marshal.ReadIntPtr(faces.Shape, IntPtr.Size));
            return new AnnyMesh(vertexData, faceData, faceSize);
        }
    }

    /// <summary>Returns a standalone GLB; options can request a rigged LBS mesh.</summary>
    public byte[] ExportGlb(string parametersJson = "{}", string optionsJson = "{}")
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.ExportGlb(model, parametersJson, optionsJson, out var pointer));
            using var bytes = new BytesHandle(pointer);
            var count = checked((int)Native.BytesLength(bytes));
            var result = new byte[count];
            if (count != 0) Marshal.Copy(Native.BytesData(bytes), result, 0, count);
            return result;
        }
    }

    /// <summary>Runs a documented secondary request without a Python bridge.</summary>
    public string Query(string requestJson)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.Query(model, requestJson, out var value));
            try { return Marshal.PtrToStringUTF8(value) ?? "{}"; }
            finally { Native.FreeString(value); }
        }
    }
    public AnnyModel Transform(string operationsJson)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.Transform(model, operationsJson, out var pointer));
            return new AnnyModel(new ModelHandle(pointer));
        }
    }
    public byte[] SavePrepared()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.Prepared(model, out var pointer));
            using var bytes = new BytesHandle(pointer);
            var count = checked((int)Native.BytesLength(bytes));
            var result = new byte[count];
            if (count != 0) Marshal.Copy(Native.BytesData(bytes), result, 0, count);
            return result;
        }
    }
    public string TransferPoseTo(AnnyModel target, string parametersJson = "{}", string mode = "local-ref")
    {
        ArgumentNullException.ThrowIfNull(target);
        // Fixed lock order prevents opposing transfers from deadlocking.
        var first = id <= target.id ? this : target;
        var second = id <= target.id ? target : this;
        lock (first.gate) lock (second.gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            ObjectDisposedException.ThrowIf(target.disposed, target);
            Check(Native.Transfer(model, target.model, parametersJson, mode, out var value));
            try { return Marshal.PtrToStringUTF8(value) ?? "{}"; }
            finally { Native.FreeString(value); }
        }
    }

    private static double[] Copy(TensorView view)
    {
        var count = checked((int)view.Length);
        var result = new double[count];
        if (count != 0) Marshal.Copy(view.Data, result, 0, count);
        return result;
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

    private static void Check(int status)
    {
        if (status != 0)
            throw new InvalidOperationException(Marshal.PtrToStringUTF8(Native.LastError()) ?? "Native Anny error.");
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct TensorView
{
    internal IntPtr Data;
    internal nuint Length;
    internal IntPtr Shape;
    internal nuint Rank;
    internal uint Kind;
}

internal sealed class ModelHandle : SafeHandle
{
    internal ModelHandle(IntPtr pointer) : base(IntPtr.Zero, true) => SetHandle(pointer);
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle() { Native.FreeModel(handle); return true; }
}
internal sealed class OutputHandle : SafeHandle
{
    internal OutputHandle(IntPtr pointer) : base(IntPtr.Zero, true) => SetHandle(pointer);
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle() { Native.FreeOutput(handle); return true; }
}

internal sealed class BytesHandle : SafeHandle
{
    internal BytesHandle(IntPtr pointer) : base(IntPtr.Zero, true) => SetHandle(pointer);
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle() { Native.FreeBytes(handle); return true; }
}

internal static class Native
{
    private const string Library = "anny_capi";
    [DllImport(Library, EntryPoint = "anny_model_build_cached", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int BuildCached([MarshalAs(UnmanagedType.LPUTF8Str)] string assets, [MarshalAs(UnmanagedType.LPUTF8Str)] string config, [MarshalAs(UnmanagedType.LPUTF8Str)] string? cache, out IntPtr result);
    [DllImport(Library, EntryPoint = "anny_model_query", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Query(ModelHandle model, [MarshalAs(UnmanagedType.LPUTF8Str)] string request, out IntPtr text);
    [DllImport(Library, EntryPoint = "anny_model_transform", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Transform(ModelHandle model, [MarshalAs(UnmanagedType.LPUTF8Str)] string operations, out IntPtr result);
    [DllImport(Library, EntryPoint = "anny_model_prepared_bytes", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Prepared(ModelHandle model, out IntPtr result);
    [DllImport(Library, EntryPoint = "anny_model_transfer_pose", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Transfer(ModelHandle source, ModelHandle target, [MarshalAs(UnmanagedType.LPUTF8Str)] string parameters, [MarshalAs(UnmanagedType.LPUTF8Str)] string mode, out IntPtr result);

    [DllImport(Library, EntryPoint = "anny_model_export_glb", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ExportGlb(ModelHandle model, [MarshalAs(UnmanagedType.LPUTF8Str)] string parameters, [MarshalAs(UnmanagedType.LPUTF8Str)] string options, out IntPtr bytes);
    [DllImport(Library, EntryPoint = "anny_bytes_data", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr BytesData(BytesHandle bytes);
    [DllImport(Library, EntryPoint = "anny_bytes_len", CallingConvention = CallingConvention.Cdecl)]
    internal static extern nuint BytesLength(BytesHandle bytes);
    [DllImport(Library, EntryPoint = "anny_bytes_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeBytes(IntPtr bytes);
    [DllImport(Library, EntryPoint = "anny_last_error", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr LastError();
    [DllImport(Library, EntryPoint = "anny_model_load", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Load([MarshalAs(UnmanagedType.LPUTF8Str)] string path, [MarshalAs(UnmanagedType.LPUTF8Str)] string? config, out IntPtr model);
    [DllImport(Library, EntryPoint = "anny_model_evaluate", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Evaluate(ModelHandle model, [MarshalAs(UnmanagedType.LPUTF8Str)] string parameters, out IntPtr output);
    [DllImport(Library, EntryPoint = "anny_model_describe", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Describe(ModelHandle model, out IntPtr text);
    [DllImport(Library, EntryPoint = "anny_model_tensor", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ModelTensor(ModelHandle model, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out TensorView view);
    [DllImport(Library, EntryPoint = "anny_output_tensor", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int OutputTensor(OutputHandle output, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out TensorView view);
    [DllImport(Library, EntryPoint = "anny_model_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeModel(IntPtr model);
    [DllImport(Library, EntryPoint = "anny_output_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeOutput(IntPtr output);
    [DllImport(Library, EntryPoint = "anny_string_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeString(IntPtr text);
}
