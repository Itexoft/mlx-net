// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

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
    public static partial MlxIoReader IoReaderNew(void* desc, MlxIoVTable vtable);

    /// <summary>Retrieves the underlying file descriptor or handle from an MLX I/O reader object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_descriptor")]
    public static partial int IoReaderDescriptor(out void* desc, MlxIoReader io);

    /// <summary>Returns a string description of the I/O reader (e.g. file path or memory source info).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_tostring")]
    public static partial int IoReaderToString(out MlxStringHandle str, MlxIoReader io);

    /// <summary>Frees an MLX I/O reader object and closes any associated resource (e.g. file).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_reader_free")]
    public static partial int IoReaderFree(MlxIoReader io);

    /// <summary>Creates a new I/O writer object for writing array data (to a file or other output stream).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_new")]
    public static partial MlxIoWriter IoWriterNew(void* desc, MlxIoVTable vtable);

    /// <summary>Retrieves the underlying file descriptor/handle from an MLX I/O writer object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_descriptor")]
    public static partial int IoWriterDescriptor(out void* desc, MlxIoWriter io);

    /// <summary>Returns a string description of the I/O writer (e.g. output target info).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_tostring")]
    public static partial int IoWriterToString(out MlxStringHandle str, MlxIoWriter io);

    /// <summary>Frees an MLX I/O writer object and closes any associated output resource.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_writer_free")]
    public static partial int IoWriterFree(MlxIoWriter io);

    /// <summary>Creates an empty GGUF container.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_new")]
    public static partial MlxIoGguf IoGgufNew();

    /// <summary>Releases a GGUF container.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_free")]
    public static partial int IoGgufFree(MlxIoGguf io);

    /// <summary>Returns all GGUF tensor and metadata keys.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_get_keys")]
    public static partial int IoGgufGetKeys(out MlxVectorStringHandle keys, MlxIoGguf io);

    /// <summary>Returns a GGUF tensor entry by key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_get_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufGetArray(out MlxArrayHandle arr, MlxIoGguf io, [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Returns a GGUF metadata tensor entry by key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_get_metadata_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufGetMetadataArray(out MlxArrayHandle arr, MlxIoGguf io, [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Returns a GGUF metadata string pointer by key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_get_metadata_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufGetMetadataString(out MlxStringHandle str, MlxIoGguf io, [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Returns a GGUF metadata string vector by key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_get_metadata_vector_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufGetMetadataVectorString(
        out MlxVectorStringHandle values,
        MlxIoGguf io,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Indicates whether GGUF metadata contains a tensor entry with the specified key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_has_metadata_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufHasMetadataArray(
        [MarshalAs(UnmanagedType.I1)] out bool flag,
        MlxIoGguf io,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Indicates whether GGUF metadata contains a string entry with the specified key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_has_metadata_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufHasMetadataString(
        [MarshalAs(UnmanagedType.I1)] out bool flag,
        MlxIoGguf io,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Indicates whether GGUF metadata contains a string-vector entry with the specified key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_has_metadata_vector_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufHasMetadataVectorString(
        [MarshalAs(UnmanagedType.I1)] out bool flag,
        MlxIoGguf io,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key);

    /// <summary>Sets a GGUF tensor entry.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_set_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufSetArray(MlxIoGguf io, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, MlxArrayHandle arr);

    /// <summary>Sets a GGUF metadata tensor entry.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_set_metadata_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufSetMetadataArray(MlxIoGguf io, [MarshalAs(UnmanagedType.LPUTF8Str)] string key, MlxArrayHandle arr);

    /// <summary>Sets a GGUF metadata string entry.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_set_metadata_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufSetMetadataString(
        MlxIoGguf io,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string value);

    /// <summary>Sets a GGUF metadata string-vector entry.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_io_gguf_set_metadata_vector_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IoGgufSetMetadataVectorString(
        MlxIoGguf io,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key,
        MlxVectorStringHandle values);
}
