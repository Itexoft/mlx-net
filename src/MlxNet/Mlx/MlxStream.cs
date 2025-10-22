// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxStream
{
    /// <summary>
    /// Creates a new execution stream on the default device.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_new")]
    public static partial MlxStreamHandle New();

    /// <summary>
    /// Creates a new execution stream on a specified device.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_new_device")]
    public static partial MlxStreamHandle NewDevice(
        MlxDeviceHandle dev
    );

    /// <summary>
    /// Sets the current default stream to a given stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_set")]
    public static partial int Set(
        ref MlxStreamHandle stream,
        MlxStreamHandle src
    );

    /// <summary>
    /// Frees a stream object and releases its resources.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_free")]
    public static partial int Free(
        MlxStreamHandle stream
    );

    /// <summary>
    /// Returns a string representation of the stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_tostring")]
    public static partial int ToString(
        out MlxStringHandle str,
        MlxStreamHandle stream
    );

    /// <summary>
    /// Checks if two stream handles refer to the same execution stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_equal")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool Equal(
        MlxStreamHandle lhs,
        MlxStreamHandle rhs
    );

    /// <summary>
    /// Returns the device associated with a given stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_get_device")]
    public static partial int GetDevice(
        out MlxDeviceHandle dev,
        MlxStreamHandle stream
    );

    /// <summary>
    /// Returns the index or identifier of the stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stream_get_index")]
    public static partial int GetIndex(
        out int index,
        MlxStreamHandle stream
    );

    /// <summary>
    /// Blocks until all pending computations on the stream are complete.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_synchronize")]
    public static partial int Synchronize(
        MlxStreamHandle stream
    );

    /// <summary>
    /// Returns the current default stream being used for operations.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_get_default_stream")]
    public static partial int GetDefaultStream(
        out MlxStreamHandle stream,
        MlxDeviceHandle dev
    );

    /// <summary>
    /// Sets the default stream to the specified stream for future operations.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_default_stream")]
    public static partial int SetDefaultStream(
        MlxStreamHandle stream
    );

    /// <summary>
    /// Returns a handle to the default CPU execution stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_default_cpu_stream_new")]
    public static partial MlxStreamHandle DefaultCpuStreamNew();

    /// <summary>
    /// Returns a handle to the default GPU execution stream.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_default_gpu_stream_new")]
    public static partial MlxStreamHandle DefaultGpuStreamNew();
}