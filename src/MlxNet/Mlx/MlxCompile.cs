// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public enum MlxCompileMode
{
    MLX_COMPILE_MODE_DISABLED,
    MLX_COMPILE_MODE_NO_SIMPLIFY,
    MLX_COMPILE_MODE_NO_FUSE,
    MLX_COMPILE_MODE_ENABLED
}

public static unsafe partial class MlxCompile
{
    /// <summary>Returns a compiled (just-in-time optimized) version of the given function or closure for faster repeated execution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_compile")]
    public static partial int Compile(
        out MlxClosureHandle res,
        MlxClosureHandle fun,
        [MarshalAs(UnmanagedType.I1)] bool shapeless
    );

    /// <summary>Internally compiles a function or computation graph for performance (lower-level call used by _mlx_compile).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_detail_compile")]
    public static partial int DetailCompile(
        out MlxClosureHandle res,
        MlxClosureHandle fun,
        nuint fun_id,
        [MarshalAs(UnmanagedType.I1)] bool shapeless,
        ulong* constants,
        nuint constants_num
    );

    /// <summary>Clears the cache of compiled functions/kernels (forcing recompilation next time).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_detail_compile_clear_cache")]
    public static partial int DetailCompileClearCache();

    /// <summary>Removes a specific compiled function from the cache (invalidating it).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_detail_compile_erase")]
    public static partial int DetailCompileErase(
        nuint fun_id
    );

    /// <summary>Globally disables JIT compilation of operations (forces eager execution or interpretation).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_disable_compile")]
    public static partial int DisableCompile();

    /// <summary>Globally enables JIT compilation for eligible operations (allowing functions to be optimized/compiled).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_enable_compile")]
    public static partial int EnableCompile();

    /// <summary>Sets the compilation mode (e.g. whether to prefer compilation/jit or interpretation, possibly with options like optimization level).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_compile_mode")]
    public static partial int SetCompileMode(
        MlxCompileMode mode
    );
}