// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxCuda
{
    /// <summary>Indicates whether CUDA support is available in the current runtime.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cuda_is_available")]
    public static partial int IsAvailable([MarshalAs(UnmanagedType.I1)] out bool available);
}
