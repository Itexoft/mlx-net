// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

[StructLayout(LayoutKind.Sequential)]
public unsafe struct MlxIoVTable
{
    public delegate* unmanaged[Cdecl]<void*, byte> is_open;
    public delegate* unmanaged[Cdecl]<void*, byte> good;
    public delegate* unmanaged[Cdecl]<void*, nuint> tell;
    public delegate* unmanaged[Cdecl]<void*, long, int, void> seek;
    public delegate* unmanaged[Cdecl]<void*, sbyte*, nuint, void> read;
    public delegate* unmanaged[Cdecl]<void*, sbyte*, nuint, nuint, void> read_at_offset;
    public delegate* unmanaged[Cdecl]<void*, sbyte*, nuint, void> write;
    public delegate* unmanaged[Cdecl]<void*, sbyte*> label;
    public delegate* unmanaged[Cdecl]<void*, void> free;
}

public static unsafe partial class MlxIoTypes
{
    /// <summary>Creates a new I/O reader object (for reading array data from a file or memory buffer).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_new")]
    public static partial MlxIoReader IoReaderNew(
        void* desc,
        MlxIoVTable vtable
    );

    /// <summary>Retrieves the underlying file descriptor or handle from an MLX I/O reader object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_descriptor")]
    public static partial int IoReaderDescriptor(
        out void* desc_,
        MlxIoReader io
    );

    /// <summary>Returns a string description of the I/O reader (e.g. file path or memory source info).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_tostring")]
    public static partial int IoReaderToString(
        out MlxStringHandle str_,
        MlxIoReader io
    );

    /// <summary>Frees an MLX I/O reader object and closes any associated resource (e.g. file).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_free")]
    public static partial int IoReaderFree(
        MlxIoReader io
    );

    /// <summary>Creates a new I/O writer object for writing array data (to a file or other output stream).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_new")]
    public static partial MlxIoWriter IoWriterNew(
        void* desc,
        MlxIoVTable vtable
    );

    /// <summary>Retrieves the underlying file descriptor/handle from an MLX I/O writer object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_descriptor")]
    public static partial int IoWriterDescriptor(
        out void* desc_,
        MlxIoWriter io
    );

    /// <summary>Returns a string description of the I/O writer (e.g. output target info).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_tostring")]
    public static partial int IoWriterToString(
        out MlxStringHandle str_,
        MlxIoWriter io
    );

    /// <summary>Frees an MLX I/O writer object and closes any associated output resource.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_free")]
    public static partial int IoWriterFree(
        MlxIoWriter io
    );
}