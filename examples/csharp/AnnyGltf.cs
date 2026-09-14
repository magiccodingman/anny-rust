using System.Runtime.InteropServices;
namespace AnnyExample;

/// <summary>Python-free glTF document operations; inputs and outputs are owned managed bytes.</summary>
public static class AnnyGltf
{
    public static byte[] Edit(byte[] bytes, string operationsJson = "[]")
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentNullException.ThrowIfNull(operationsJson);
        Check(EditNative(bytes, checked((nuint)bytes.Length), operationsJson, out var pointer));
        using var result = new BytesHandle(pointer);
        var count = checked((int)Native.BytesLength(result));
        var output = new byte[count];
        if (count != 0) Marshal.Copy(Native.BytesData(result), output, 0, count);
        return output;
    }
    public static string Query(byte[] bytes, string requestJson = "{\"operation\":\"describe\"}")
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentNullException.ThrowIfNull(requestJson);
        Check(QueryNative(bytes, checked((nuint)bytes.Length), requestJson, out var pointer));
        try { return Marshal.PtrToStringUTF8(pointer) ?? "{}"; }
        finally { Native.FreeString(pointer); }
    }
    private static void Check(int status)
    {
        if (status != 0) throw new InvalidOperationException(Marshal.PtrToStringUTF8(Native.LastError()) ?? "Native glTF error.");
    }
    [DllImport("anny_capi", EntryPoint = "anny_gltf_edit", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EditNative([In] byte[] bytes, nuint length, [MarshalAs(UnmanagedType.LPUTF8Str)] string operations, out IntPtr result);
    [DllImport("anny_capi", EntryPoint = "anny_gltf_query", CallingConvention = CallingConvention.Cdecl)]
    private static extern int QueryNative([In] byte[] bytes, nuint length, [MarshalAs(UnmanagedType.LPUTF8Str)] string request, out IntPtr result);
}
