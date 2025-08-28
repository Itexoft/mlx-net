// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxExport
{
    /// <summary>Exports a given MLX function or computation graph to an external representation or file (for use in other contexts).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_export_function", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Function(
        string file,
        MlxClosureHandle fun,
        MlxVectorArrayHandle args,
        [MarshalAs(UnmanagedType.I1)] bool shapeless
    );

    /// <summary>Similar to _mlx_export_function, but handles functions that take keyword arguments (preserving argument names in the export).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_export_function_kwargs", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int FunctionKwargs(
        string file,
        MlxClosureKwargsHandle fun,
        MlxVectorArrayHandle args,
        MlxMapStringToArrayHandle kwargs,
        [MarshalAs(UnmanagedType.I1)] bool shapeless
    );

    /// <summary>Creates a new function exporter, which can record a function’s operations (for saving or inspecting a computation graph).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_function_exporter_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxFunctionExporter FunctionExporterNew(
        string file,
        MlxClosureHandle fun,
        [MarshalAs(UnmanagedType.I1)] bool shapeless
    );

    /// <summary>Frees the resources associated with a function exporter object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_function_exporter_free")]
    public static partial int FunctionExporterFree(
        MlxFunctionExporter xfunc
    );

    /// <summary>Applies (runs) a function under an exporter to capture its execution as an exportable representation.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_function_exporter_apply")]
    public static partial int FunctionExporterApply(
        MlxFunctionExporter xfunc,
        MlxVectorArrayHandle args
    );

    /// <summary>Applies a function (with keyword arguments) under an exporter, capturing its execution for export.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_function_exporter_apply_kwargs")]
    public static partial int FunctionExporterApplyKwargs(
        MlxFunctionExporter xfunc,
        MlxVectorArrayHandle args,
        MlxMapStringToArrayHandle kwargs
    );

    /// <summary>Loads or creates an MLX function from an imported (saved) format (e.g. from a file containing a serialized function).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_imported_function_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxImportedFunction ImportedFunctionNew(
        string file
    );

    /// <summary>Releases the resources associated with an imported function.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_imported_function_free")]
    public static partial int ImportedFunctionFree(
        MlxImportedFunction xfunc
    );

    /// <summary>Applies a previously imported function (from an external representation) with given arguments, executing its computation.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_imported_function_apply")]
    public static partial int ImportedFunctionApply(
        out MlxVectorArrayHandle res,
        MlxImportedFunction xfunc,
        MlxVectorArrayHandle args
    );

    /// <summary>Applies an imported function using keyword arguments for its parameters.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_imported_function_apply_kwargs")]
    public static partial int ImportedFunctionApplyKwargs(
        out MlxVectorArrayHandle res,
        MlxImportedFunction xfunc,
        MlxVectorArrayHandle args,
        MlxMapStringToArrayHandle kwargs
    );
}