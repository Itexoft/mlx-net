// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxIo
{
    /// <summary>Loads array(s) using a given I/O reader instead of a filename.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_load_reader")]
    public static partial int LoadReader(out MlxArrayHandle res, MlxIoReader inStream, MlxStreamHandle s);

    /// <summary>Loads array(s) from a file, autodetecting format and returning the array or dictionary of arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_load", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Load(out MlxArrayHandle res, [MarshalAs(UnmanagedType.LPUTF8Str)] string file, MlxStreamHandle s);

    /// <summary>Loads a GGUF container including tensor and metadata entries.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_load_gguf", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int LoadGguf(out MlxIoGguf gguf, [MarshalAs(UnmanagedType.LPUTF8Str)] string file, MlxStreamHandle s);

    /// <summary>Loads array(s) from a SafeTensors format using an I/O reader source.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_load_safetensors_reader")]
    public static partial int LoadSafetensorsReader(
        out MlxMapStringToArrayHandle res0,
        out MlxMapStringToStringHandle res1,
        MlxIoReader inStream,
        MlxStreamHandle s);

    /// <summary>Loads array(s) from a SafeTensors file including tensor data and metadata.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_load_safetensors", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int LoadSafetensors(
        out MlxMapStringToArrayHandle res0,
        out MlxMapStringToStringHandle res1,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string file,
        MlxStreamHandle s);

    /// <summary>Saves an array to a .npy file using the provided I/O writer object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_save_writer")]
    public static partial int SaveWriter(MlxIoWriter outStream, MlxArrayHandle a);

    /// <summary>Saves an array to a binary file in NumPy .npy format.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_save", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Save([MarshalAs(UnmanagedType.LPUTF8Str)] string file, MlxArrayHandle a);

    /// <summary>Saves a GGUF container with tensor and metadata entries.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_save_gguf", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SaveGguf([MarshalAs(UnmanagedType.LPUTF8Str)] string file, MlxIoGguf gguf);

    /// <summary>Saves array(s) in safetensors format using a provided writer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_save_safetensors_writer")]
    public static partial int SaveSafetensorsWriter(MlxIoWriter inStream, MlxMapStringToArrayHandle param, MlxMapStringToStringHandle metadata);

    /// <summary>Saves one or more arrays with optional metadata to a file in safetensors format.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_save_safetensors", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SaveSafetensors(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string file,
        MlxMapStringToArrayHandle param,
        MlxMapStringToStringHandle metadata);
}
