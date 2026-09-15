using System;
using System.Runtime.InteropServices;

namespace Anny
{
    /// <summary>
    /// Owns a native <c>AnnyModel</c>. Release runs exactly once, including from the finalizer,
    /// so an aborted exception path cannot leak the model.
    /// </summary>
    public sealed class AnnyModelHandle : SafeHandle
    {
        public AnnyModelHandle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnyModelHandle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.ModelFree(handle);
            return true;
        }
    }

    /// <summary>Owns a native <c>AnnyOutput</c>. Independent of the model that produced it.</summary>
    public sealed class AnnyOutputHandle : SafeHandle
    {
        public AnnyOutputHandle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnyOutputHandle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.OutputFree(handle);
            return true;
        }
    }

    /// <summary>Owns a native <c>AnnySession</c>. Holds its own reference to the model.</summary>
    public sealed class AnnySessionHandle : SafeHandle
    {
        public AnnySessionHandle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnySessionHandle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.SessionFree(handle);
            return true;
        }
    }

    /// <summary>Owns a native neutral byte buffer (GLB or prepared payload).</summary>
    public sealed class AnnyBytesHandle : SafeHandle
    {
        public AnnyBytesHandle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnyBytesHandle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.BytesFree(handle);
            return true;
        }
    }

    /// <summary>Owns a native single-precision <c>AnnyModelF32</c>.</summary>
    public sealed class AnnyModelF32Handle : SafeHandle
    {
        public AnnyModelF32Handle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnyModelF32Handle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.ModelF32Free(handle);
            return true;
        }
    }

    /// <summary>Owns a native single-precision <c>AnnyOutputF32</c>.</summary>
    public sealed class AnnyOutputF32Handle : SafeHandle
    {
        public AnnyOutputF32Handle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnyOutputF32Handle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.OutputF32Free(handle);
            return true;
        }
    }

    /// <summary>Owns a native single-precision <c>AnnySessionF32</c>.</summary>
    public sealed class AnnySessionF32Handle : SafeHandle
    {
        public AnnySessionF32Handle()
            : base(IntPtr.Zero, true)
        {
        }

        public AnnySessionF32Handle(IntPtr pointer)
            : base(IntPtr.Zero, true)
        {
            SetHandle(pointer);
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            AnnyNative.SessionF32Free(handle);
            return true;
        }
    }
}