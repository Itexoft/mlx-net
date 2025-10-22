// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxMetal
{
    /// <summary>Returns information about the Metal GPU devices available (e.g. names, features of the GPU).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_metal_device_info")]
    public static partial MlxMetalDeviceInfoT DeviceInfo();

    /// <summary>Returns True if Apple Metal (GPU acceleration) is available on this system.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_metal_is_available")]
    public static partial int IsAvailable(
        out byte res
    );

    /// <summary>Starts capturing GPU commands (using Metal’s capture tool) for debugging or profiling GPU execution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_metal_start_capture", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StartCapture(
        string path
    );

    /// <summary>Stops the Metal GPU command capture that was previously started.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_metal_stop_capture")]
    public static partial int StopCapture();
}