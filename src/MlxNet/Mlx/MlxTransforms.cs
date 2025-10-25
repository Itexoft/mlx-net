// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxTransforms
{
    /// <summary>Asynchronously evaluates the array’s computation without blocking.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_async_eval")]
    public static partial int AsyncEval(
        MlxVectorArrayHandle outputs
    );

    /// <summary>Marks a point in computation for gradient checkpointing.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_checkpoint")]
    public static partial int Checkpoint(
        out MlxClosureHandle res,
        MlxClosureHandle fun
    );

    /// <summary>Defines a function with custom gradient and vectorization rules.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_custom_function")]
    public static partial int CustomFunction(
        out MlxClosureHandle res,
        MlxClosureHandle fun,
        MlxClosureCustomHandle fun_vjp,
        MlxClosureCustomJvp fun_jvp,
        MlxClosureCustomVmap fun_vmap
    );

    /// <summary>Sets up a custom VJP for a function, overriding default autodiff.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_custom_vjp")]
    public static partial int CustomVjp(
        out MlxClosureHandle res,
        MlxClosureHandle fun,
        MlxClosureCustomHandle fun_vjp
    );

    /// <summary>Forces evaluation of pending computations for the given outputs.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_eval")]
    public static partial int Eval(
        MlxVectorArrayHandle outputs
    );

    /// <summary>Computes the Jacobian-vector product of a function in forward mode.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_jvp")]
    public static partial int Jvp(
        out MlxVectorArrayHandle res_0,
        out MlxVectorArrayHandle res_1,
        MlxClosureHandle fun,
        MlxVectorArrayHandle primals,
        MlxVectorArrayHandle tangents
    );

    /// <summary>Creates a closure that returns both a function’s value and its gradient.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_value_and_grad")]
    public static partial int ValueAndGrad(
        out MlxClosureValueAndGradHandle res,
        MlxClosureHandle fun,
        int* argnums,
        nuint argnums_num
    );

    /// <summary>Computes the vector-Jacobian product of a function in reverse mode.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vjp")]
    public static partial int Vjp(
        out MlxVectorArrayHandle res_0,
        out MlxVectorArrayHandle res_1,
        MlxClosureHandle fun,
        MlxVectorArrayHandle primals,
        MlxVectorArrayHandle cotangents
    );

    /// <summary>Replaces traced inputs/outputs during vmap transformation (internal helper).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_detail_vmap_replace")]
    public static partial int DetailVmapReplace(
        out MlxVectorArrayHandle res,
        MlxVectorArrayHandle inputs,
        MlxVectorArrayHandle s_inputs,
        MlxVectorArrayHandle s_outputs,
        int* in_axes,
        nuint in_axes_num,
        int* out_axes,
        nuint out_axes_num
    );

    /// <summary>Traces a function for vectorization, returning transformed outputs and metadata.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_detail_vmap_trace")]
    public static partial int DetailVmapTrace(
        out MlxVectorArrayHandle res_0,
        out MlxVectorArrayHandle res_1,
        MlxClosureHandle fun,
        MlxVectorArrayHandle inputs,
        int* in_axes,
        nuint in_axes_num
    );
}
