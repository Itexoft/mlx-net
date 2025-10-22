// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxMemory
{
    /// <summary>
    /// Clears internal caches used by the MLX framework.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_clear_cache")]
    public static partial int ClearCache();

    /// <summary>
    /// Returns the current amount of actively allocated memory in bytes.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_get_active_memory")]
    public static partial int GetActiveMemory(
        out nuint res
    );

    /// <summary>
    /// Returns the current size of cached memory in bytes.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_get_cache_memory")]
    public static partial int GetCacheMemory(
        out nuint res
    );

    /// <summary>
    /// Returns the current memory limit in bytes set for MLX allocations.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_get_memory_limit")]
    public static partial int GetMemoryLimit(
        out nuint res
    );

    /// <summary>
    /// Returns the peak memory usage in bytes recorded during runtime.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_get_peak_memory")]
    public static partial int GetPeakMemory(
        out nuint res
    );

    /// <summary>
    /// Resets the recorded peak memory usage counter.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_reset_peak_memory")]
    public static partial int ResetPeakMemory();

    /// <summary>
    /// Sets a limit on the cache size for MLX's allocator or kernel caches.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_cache_limit")]
    public static partial int SetCacheLimit(
        out nuint res,
        nuint limit
    );

    /// <summary>
    /// Sets a limit on total memory that MLX can allocate for arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_memory_limit")]
    public static partial int SetMemoryLimit(
        out nuint res,
        nuint limit
    );

    /// <summary>
    /// Sets a limit on wired (pinned) memory that MLX can use.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_wired_limit")]
    public static partial int SetWiredLimit(
        out nuint res,
        nuint limit
    );
}