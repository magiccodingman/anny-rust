using System;
using System.Runtime.InteropServices;

namespace Anny
{
    /// <summary>Semantic kind of a tensor, as reported by the native ABI.</summary>
    public enum AnnyTensorKind
    {
        Float = 0,
        Index = 1,
        Bool = 2,
    }

    /// <summary>Borrowed f64 tensor view. Valid only until the owning handle is freed.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AnnyTensorView
    {
        public IntPtr Data;
        public UIntPtr Length;
        public IntPtr Shape;
        public UIntPtr Rank;
        public uint Kind;
    }

    /// <summary>Borrowed f32 tensor view. Valid only until the owning handle is freed.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AnnyTensorViewF32
    {
        public IntPtr Data;
        public UIntPtr Length;
        public IntPtr Shape;
        public UIntPtr Rank;
        public uint Kind;
    }

    /// <summary>Error raised when a native anny call returns a nonzero status.</summary>
    public sealed class AnnyNativeException : Exception
    {
        public AnnyNativeException(int status, string message)
            : base(message)
        {
            Status = status;
        }

        public int Status { get; private set; }
    }

    /// <summary>
    /// Raw ABI-v1 surface of the Anny-Rust native library. Every entry point below is a
    /// one-to-one mapping of <c>include/anny.h</c>; the managed wrappers in this package own
    /// the lifetime rules. Strings cross the boundary as UTF-8; views are borrowed, read-only
    /// and invalidated when the handle that produced them is freed.
    /// </summary>
    public static class AnnyNative
    {
        /// <summary>
        /// Native library basename. Ships as libanny.so / anny.dll / libanny.dylib. On WebGL the
        /// functions are linked into the player from <c>Plugins/WebGL/libanny.a</c>, so they resolve
        /// through Emscripten's <c>__Internal</c> module instead of by loading a library.
        /// </summary>
#if UNITY_WEBGL && !UNITY_EDITOR
        public const string Library = "__Internal";
#else
        public const string Library = "anny";
#endif

        public const int StatusOk = 0;
        public const int StatusError = 1;
        public const int StatusPanic = 2;

        /// <summary>ABI version this managed layer was written against.</summary>
        public const uint ExpectedAbi = 1;

        [DllImport(Library, EntryPoint = "anny_abi_version", CallingConvention = CallingConvention.Cdecl)]
        public static extern uint AbiVersion();

        [DllImport(Library, EntryPoint = "anny_last_error", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr LastErrorPointer();

        // ---- double-precision model API ------------------------------------------------

        [DllImport(Library, EntryPoint = "anny_model_load", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelLoad(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson,
            out IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_from_bytes", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelFromBytes(
            byte[] bytes,
            UIntPtr length,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson,
            out IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_build", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelBuild(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string assets,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson,
            out IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_build_cached", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelBuildCached(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string assets,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string cacheDirectory,
            out IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ModelFree(IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_evaluate", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelEvaluate(
            AnnyModelHandle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string parametersJson,
            out IntPtr output);

        [DllImport(Library, EntryPoint = "anny_model_tensor", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelTensor(
            AnnyModelHandle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            out AnnyTensorView view);

        [DllImport(Library, EntryPoint = "anny_model_describe", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelDescribe(AnnyModelHandle model, out IntPtr text);

        [DllImport(Library, EntryPoint = "anny_model_query", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelQuery(
            AnnyModelHandle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
            out IntPtr text);

        [DllImport(Library, EntryPoint = "anny_model_transform", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelTransform(
            AnnyModelHandle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string operationsJson,
            out IntPtr result);

        [DllImport(Library, EntryPoint = "anny_model_prepared_bytes", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelPreparedBytes(AnnyModelHandle model, out IntPtr bytes);

        [DllImport(Library, EntryPoint = "anny_model_transfer_pose", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelTransferPose(
            AnnyModelHandle source,
            AnnyModelHandle target,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string parametersJson,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string mode,
            out IntPtr text);

        [DllImport(Library, EntryPoint = "anny_model_export_glb", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelExportGlb(
            AnnyModelHandle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string parametersJson,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string optionsJson,
            out IntPtr bytes);

        // ---- double-precision output API ----------------------------------------------

        [DllImport(Library, EntryPoint = "anny_output_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void OutputFree(IntPtr output);

        [DllImport(Library, EntryPoint = "anny_output_tensor", CallingConvention = CallingConvention.Cdecl)]
        public static extern int OutputTensor(
            AnnyOutputHandle output,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            out AnnyTensorView view);

        // ---- pose sessions ------------------------------------------------------------

        [DllImport(Library, EntryPoint = "anny_session_new", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionNew(
            AnnyModelHandle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string parametersJson,
            out IntPtr session);

        [DllImport(Library, EntryPoint = "anny_session_update", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionUpdate(
            AnnySessionHandle session,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string poseJson);

        [DllImport(Library, EntryPoint = "anny_session_tensor", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionTensor(
            AnnySessionHandle session,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            out AnnyTensorView view);

        [DllImport(Library, EntryPoint = "anny_session_coefficients", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionCoefficients(AnnySessionHandle session, out AnnyTensorView view);

        [DllImport(Library, EntryPoint = "anny_session_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void SessionFree(IntPtr session);

        // ---- single-precision model API -----------------------------------------------

        [DllImport(Library, EntryPoint = "anny_model_to_f32", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelToF32(AnnyModelHandle source, out IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_f32_from_bytes", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelF32FromBytes(
            byte[] bytes,
            UIntPtr length,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson,
            out IntPtr model);

        [DllImport(Library, EntryPoint = "anny_model_f32_evaluate", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelF32Evaluate(
            AnnyModelF32Handle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string parametersJson,
            out IntPtr output);

        [DllImport(Library, EntryPoint = "anny_model_f32_tensor", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelF32Tensor(
            AnnyModelF32Handle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            out AnnyTensorViewF32 view);

        [DllImport(Library, EntryPoint = "anny_model_f32_prepared_bytes", CallingConvention = CallingConvention.Cdecl)]
        public static extern int ModelF32PreparedBytes(AnnyModelF32Handle model, out IntPtr bytes);

        [DllImport(Library, EntryPoint = "anny_model_f32_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void ModelF32Free(IntPtr model);

        [DllImport(Library, EntryPoint = "anny_output_f32_tensor", CallingConvention = CallingConvention.Cdecl)]
        public static extern int OutputF32Tensor(
            AnnyOutputF32Handle output,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            out AnnyTensorViewF32 view);

        [DllImport(Library, EntryPoint = "anny_output_f32_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void OutputF32Free(IntPtr output);

        [DllImport(Library, EntryPoint = "anny_session_f32_new", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionF32New(
            AnnyModelF32Handle model,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string parametersJson,
            out IntPtr session);

        [DllImport(Library, EntryPoint = "anny_session_f32_update", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionF32Update(
            AnnySessionF32Handle session,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string poseJson);

        [DllImport(Library, EntryPoint = "anny_session_f32_tensor", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionF32Tensor(
            AnnySessionF32Handle session,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            out AnnyTensorViewF32 view);

        [DllImport(Library, EntryPoint = "anny_session_f32_coefficients", CallingConvention = CallingConvention.Cdecl)]
        public static extern int SessionF32Coefficients(AnnySessionF32Handle session, out AnnyTensorViewF32 view);

        [DllImport(Library, EntryPoint = "anny_session_f32_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void SessionF32Free(IntPtr session);

        // ---- neutral buffer + strings -------------------------------------------------

        [DllImport(Library, EntryPoint = "anny_bytes_data", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr BytesData(AnnyBytesHandle bytes);

        [DllImport(Library, EntryPoint = "anny_bytes_len", CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr BytesLength(AnnyBytesHandle bytes);

        [DllImport(Library, EntryPoint = "anny_bytes_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void BytesFree(IntPtr bytes);

        [DllImport(Library, EntryPoint = "anny_string_free", CallingConvention = CallingConvention.Cdecl)]
        public static extern void StringFree(IntPtr text);

        [DllImport(Library, EntryPoint = "anny_gltf_edit", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GltfEdit(
            byte[] bytes,
            UIntPtr length,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string operationsJson,
            out IntPtr result);

        [DllImport(Library, EntryPoint = "anny_gltf_query", CallingConvention = CallingConvention.Cdecl)]
        public static extern int GltfQuery(
            byte[] bytes,
            UIntPtr length,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
            out IntPtr text);

        /// <summary>Reads the borrowed last-error text; must be copied before the next native call.</summary>
        public static string LastError()
        {
            IntPtr pointer = LastErrorPointer();
            return pointer == IntPtr.Zero ? string.Empty : Marshal.PtrToStringUTF8(pointer) ?? string.Empty;
        }

        /// <summary>Throws when a native status is nonzero, capturing the message immediately.</summary>
        public static void Check(int status)
        {
            if (status == StatusOk)
            {
                return;
            }

            throw new AnnyNativeException(status, LastError());
        }

        /// <summary>Verifies that the loaded native library speaks the ABI this layer expects.</summary>
        public static void RequireAbi()
        {
            uint version;
            try
            {
                version = AbiVersion();
            }
            catch (DllNotFoundException e)
            {
                throw new AnnyNativeException(
                    StatusError,
                    "The native anny library was not found. Expected libanny.so (Linux), anny.dll " +
                    "(Windows) or libanny.dylib (macOS) inside the package Plugins folder; rebuild it " +
                    "with anny-rust's tools/unity/build-native.sh for this platform. " + e.Message);
            }

            if (version != ExpectedAbi)
            {
                throw new AnnyNativeException(
                    StatusError,
                    "anny native ABI " + version + " does not match the managed layer's expected ABI " + ExpectedAbi + ".");
            }
        }
    }
}