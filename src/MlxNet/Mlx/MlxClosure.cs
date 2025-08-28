// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxClosure
{
    /// <summary>Creates a new closure (function object) from a given function pointer or MLX function, allowing it to be passed around or invoked later.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_new")]
    public static partial MlxClosureHandle New();

    /// <summary>Releases/frees a closure and its associated resources.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_free")]
    public static partial int Free(MlxClosureHandle cls);

    /// <summary>Specifies the function pointer or implementation for a newly created closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_new_func")]
    public static partial MlxClosureHandle NewFunc(delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, int> fun);

    /// <summary>Attaches extra payload data (captured environment or constants) to a closure’s function.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_new_func_payload")]
    public static partial MlxClosureHandle NewFuncPayload(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, void*, int> fun,
        void* payload,
        delegate* unmanaged[Cdecl]<void*, void> dtor);

    /// <summary>Finalizes the closure creation by setting its attributes or registering it within the runtime so it can be invoked.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_set")]
    public static partial int Set(ref MlxClosureHandle cls, MlxClosureHandle src);

    /// <summary>Applies (executes) a previously created MLX closure (function object) with given arguments.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_apply")]
    public static partial int Apply(out MlxVectorArrayHandle res, MlxClosureHandle cls, MlxVectorArrayHandle input);

    /// <summary>Creates a new closure optimized for a unary function (single-argument function).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_new_unary")]
    public static partial MlxClosureHandle NewUnary(delegate* unmanaged[Cdecl]<MlxArrayHandle*, MlxArrayHandle, int> fun);

    /// <summary>Creates a new closure that can be called with keyword arguments (i.e. wraps a function with named parameters).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_kwargs_new")]
    public static partial MlxClosureKwargsHandle KwargsNew();

    /// <summary>Frees a closure created to handle keyword arguments.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_kwargs_free")]
    public static partial int KwargsFree(MlxClosureKwargsHandle cls);

    /// <summary>Sets the underlying function for a kwargs-closure (the function that accepts specific named parameters).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_kwargs_new_func")]
    public static partial MlxClosureKwargsHandle KwargsNewFunc(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, MlxMapStringToArrayHandle, int> fun);

    /// <summary>Attaches payload data for the function of a kwargs-closure (context or captured variables).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_kwargs_new_func_payload")]
    public static partial MlxClosureKwargsHandle KwargsNewFuncPayload(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, MlxMapStringToArrayHandle, void*, int> fun,
        void* payload,
        delegate* unmanaged[Cdecl]<void*, void> dtor);

    /// <summary>Finalizes or sets additional properties for the keyword-argument accepting closure (such as default values or schema).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_kwargs_set")]
    public static partial int KwargsSet(ref MlxClosureKwargsHandle cls, MlxClosureKwargsHandle src);

    /// <summary>Applies a closure that accepts keyword arguments, using a dictionary of argument names to values.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_kwargs_apply")]
    public static partial int KwargsApply(
        out MlxVectorArrayHandle res,
        MlxClosureKwargsHandle cls,
        MlxVectorArrayHandle input_0,
        MlxMapStringToArrayHandle input_1);

    /// <summary>Creates a new closure that will compute both a function’s result and its gradient.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_value_and_grad_new")]
    public static partial MlxClosureValueAndGradHandle ValueAndGradNew();

    /// <summary>Frees a closure that computes value and gradient (releases its resources).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_value_and_grad_free")]
    public static partial int ValueAndGradFree(MlxClosureValueAndGradHandle cls);

    /// <summary>Sets the underlying function for the value-and-grad closure (the function whose value and grad will be computed).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_value_and_grad_new_func")]
    public static partial MlxClosureValueAndGradHandle ValueAndGradNewFunc(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle*, MlxVectorArrayHandle, int> fun);

    /// <summary>Attaches payload (e.g. parameters that remain constant) to the value-and-grad closure’s function.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_value_and_grad_new_func_payload")]
    public static partial MlxClosureValueAndGradHandle ValueAndGradNewFuncPayload(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle*, MlxVectorArrayHandle, void*, int> fun,
        void* payload,
        delegate* unmanaged[Cdecl]<void*, void> dtor);

    /// <summary>Finalizes configuration of the value-and-grad closure (e.g. specifies which argument to differentiate with respect to).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_value_and_grad_set")]
    public static partial int ValueAndGradSet(ref MlxClosureValueAndGradHandle cls, MlxClosureValueAndGradHandle src);

    /// <summary>Applies a special closure that computes both value and gradient, returning the function’s output and its gradient.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_value_and_grad_apply")]
    public static partial int ValueAndGradApply(
        out MlxVectorArrayHandle res_0,
        out MlxVectorArrayHandle res_1,
        MlxClosureValueAndGradHandle cls,
        MlxVectorArrayHandle input);

    /// <summary>Creates a new closure for a user-defined operation with custom gradient and vectorization behavior.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_new")]
    public static partial MlxClosureCustomHandle CustomNew();

    /// <summary>Releases resources associated with a custom operation closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_free")]
    public static partial int CustomFree(MlxClosureCustomHandle cls);

    /// <summary>Sets the forward function for the custom operation closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_new_func")]
    public static partial MlxClosureCustomHandle CustomNewFunc(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, MlxVectorArrayHandle, MlxVectorArrayHandle, int> fun);

    /// <summary>Attaches user payload (captured data) to the custom operation closure’s forward function.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_new_func_payload")]
    public static partial MlxClosureCustomHandle CustomNewFuncPayload(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, MlxVectorArrayHandle, MlxVectorArrayHandle, void*, int> fun,
        void* payload,
        delegate* unmanaged[Cdecl]<void*, void> dtor);

    /// <summary>Finalizes the custom operation closure, optionally registering custom gradient (VJP) or vectorization (vmap) rules.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_set")]
    public static partial int CustomSet(ref MlxClosureCustomHandle cls, MlxClosureCustomHandle src);

    /// <summary>Applies a custom operation closure (one with user-defined gradient or vmap) with given arguments.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_apply")]
    public static partial int CustomApply(
        out MlxVectorArrayHandle res,
        MlxClosureCustomHandle cls,
        MlxVectorArrayHandle input_0,
        MlxVectorArrayHandle input_1,
        MlxVectorArrayHandle input_2);

    /// <summary>Creates a new closure for a function with a custom JVP (forward-mode differentiation) definition.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_jvp_new")]
    public static partial MlxClosureCustomJvp CustomJvpNew();

    /// <summary>Frees the resources for a custom-JVP closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_jvp_free")]
    public static partial int CustomJvpFree(MlxClosureCustomJvp cls);

    /// <summary>Sets the primary (primal) function for the custom-JVP closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_jvp_new_func")]
    public static partial MlxClosureCustomJvp CustomJvpNewFunc(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, MlxVectorArrayHandle, int*, nuint, int> fun);

    /// <summary>Attaches user-defined payload data to the custom-JVP closure’s primary function.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_jvp_new_func_payload")]
    public static partial MlxClosureCustomJvp CustomJvpNewFuncPayload(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorArrayHandle, MlxVectorArrayHandle, int*, nuint, void*, int> fun,
        void* payload,
        delegate* unmanaged[Cdecl]<void*, void> dtor);

    /// <summary>Defines the custom Jacobian-vector product function for the closure (the rule for forward-mode differentiation).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_jvp_set")]
    public static partial int CustomJvpSet(ref MlxClosureCustomJvp cls, MlxClosureCustomJvp src);

    /// <summary>Applies a closure that has a custom JVP rule to given inputs (computing forward-mode AD result).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_jvp_apply")]
    public static partial int CustomJvpApply(
        out MlxVectorArrayHandle res,
        MlxClosureCustomJvp cls,
        MlxVectorArrayHandle input_0,
        MlxVectorArrayHandle input_1,
        int* input_2,
        nuint input_2_num);

    /// <summary>Creates a new closure for a function with a custom vmap (vectorization) definition.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_vmap_new")]
    public static partial MlxClosureCustomVmap CustomVmapNew();

    /// <summary>Frees the resources of a custom-vmap closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_vmap_free")]
    public static partial int CustomVmapFree(MlxClosureCustomVmap cls);

    /// <summary>Sets the base scalar function for the custom-vmap closure.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_vmap_new_func")]
    public static partial MlxClosureCustomVmap CustomVmapNewFunc(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorIntHandle*, MlxVectorArrayHandle, int*, nuint, int> fun);

    /// <summary>Attaches payload data to the custom-vmap closure’s base function.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_vmap_new_func_payload")]
    public static partial MlxClosureCustomVmap CustomVmapNewFuncPayload(
        delegate* unmanaged[Cdecl]<MlxVectorArrayHandle*, MlxVectorIntHandle*, MlxVectorArrayHandle, int*, nuint, void*, int> fun,
        void* payload,
        delegate* unmanaged[Cdecl]<void*, void> dtor);

    /// <summary>Defines the custom vectorization mapping function for the closure (how to map it across batch dimensions).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_vmap_set")]
    public static partial int CustomVmapSet(ref MlxClosureCustomVmap cls, MlxClosureCustomVmap src);

    /// <summary>Applies a closure with a custom vectorization (vmap) rule to batched inputs (computes the vectorized result).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_closure_custom_vmap_apply")]
    public static partial int CustomVmapApply(
        out MlxVectorArrayHandle res_0,
        out MlxVectorIntHandle res_1,
        MlxClosureCustomVmap cls,
        MlxVectorArrayHandle input_0,
        int* input_1,
        nuint input_1_num);
}