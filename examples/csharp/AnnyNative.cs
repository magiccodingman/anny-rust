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

    /// <summary>Copies static data once into an independent f32 native runtime.</summary>
    public AnnySinglePrecisionModel ToSinglePrecision()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(NativeF32.Convert(model, out var pointer));
            return new AnnySinglePrecisionModel(pointer);
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

    /// <summary>
    /// Starts a reusable pose session: the coefficient and rest-model halves are evaluated once, so
    /// every <see cref="AnnyPoseSession.Update"/> pays only for the pose. The session keeps its own
    /// native reference to the model, so disposing this model first is safe.
    /// </summary>
    public AnnyPoseSession CreateSession(string parametersJson = "{}")
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Check(Native.SessionNew(model, parametersJson, out var pointer));
            return new AnnyPoseSession(this, new SessionHandle(pointer));
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

    internal static double[] Copy(TensorView view)
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

    internal static void Check(int status)
    {
        if (status != 0)
            throw new InvalidOperationException(Marshal.PtrToStringUTF8(Native.LastError()) ?? "Native Anny error.");
    }
}

/// <summary>
/// Owns one native pose session over a model. Not thread safe: serialise calls yourself. It holds the
/// managed model, so the model cannot be collected (and the native model cannot be freed) while the
/// session is alive.
/// </summary>
public sealed class AnnyPoseSession : IDisposable
{
    private readonly AnnyModel owner;
    private readonly SessionHandle session;
    private readonly object gate = new();
    private bool disposed;

    internal AnnyPoseSession(AnnyModel owner, SessionHandle session)
    {
        this.owner = owner;
        this.session = session;
    }

    /// <summary>Re-poses the session. Arrays copied before this call remain valid.</summary>
    public void Update(string poseJson = "{}")
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            AnnyModel.Check(Native.SessionUpdate(session, poseJson));
        }
    }

    /// <summary>Copies one array of the most recent pose (the rest model before any update).</summary>
    public double[] Tensor(string name)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            AnnyModel.Check(Native.SessionTensor(session, name, out var view));
            return AnnyModel.Copy(view);
        }
    }

    /// <summary>Copies the coefficients the session was created with.</summary>
    public double[] Coefficients()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            AnnyModel.Check(Native.SessionCoefficients(session, out var view));
            return AnnyModel.Copy(view);
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            session.Dispose();
        }
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

internal sealed class SessionHandle : SafeHandle
{
    internal SessionHandle(IntPtr pointer) : base(IntPtr.Zero, true) => SetHandle(pointer);
    public override bool IsInvalid => handle == IntPtr.Zero;
    protected override bool ReleaseHandle() { Native.FreeSession(handle); return true; }
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
    [DllImport(Library, EntryPoint = "anny_session_new", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SessionNew(ModelHandle model, [MarshalAs(UnmanagedType.LPUTF8Str)] string parameters, out IntPtr session);
    [DllImport(Library, EntryPoint = "anny_session_update", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SessionUpdate(SessionHandle session, [MarshalAs(UnmanagedType.LPUTF8Str)] string pose);
    [DllImport(Library, EntryPoint = "anny_session_tensor", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SessionTensor(SessionHandle session, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out TensorView view);
    [DllImport(Library, EntryPoint = "anny_session_coefficients", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SessionCoefficients(SessionHandle session, out TensorView view);
    [DllImport(Library, EntryPoint = "anny_session_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeSession(IntPtr session);
    [DllImport(Library, EntryPoint = "anny_string_free", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FreeString(IntPtr text);
}
