// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxFast
{
    /// <summary>Efficiently converts affine-quantized integers (given scale and zero-point) to floats (applies x*scale + zero to a quantized array).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_affine_dequantize")]
    public static partial int AffineDequantize(
        out MlxArrayHandle res,
        MlxArrayHandle w,
        MlxArrayHandle scales,
        MlxArrayHandle biases,
        int group_size,
        int bits,
        MlxStreamHandle s
    );

    /// <summary>Efficiently quantizes float values into integers using an affine mapping (with given scale and zero-point).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_affine_quantize")]
    public static partial int AffineQuantize(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        out MlxArrayHandle res_2,
        MlxArrayHandle w,
        int group_size,
        int bits,
        MlxStreamHandle s
    );

    /// <summary>Performs layer normalization on an array with a fast, optimized implementation.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_layer_norm")]
    public static partial int LayerNorm(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxArrayHandle weight,
        MlxArrayHandle bias,
        float eps,
        MlxStreamHandle s
    );

    /// <summary>Creates a new configuration object for a Metal kernel (to specify grid size, thread groups, and arguments).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_new")]
    public static partial MlxFastMetalKernelConfig MetalKernelConfigNew();

    /// <summary>Frees/destroys a Metal kernel configuration object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_free")]
    public static partial void MetalKernelConfigFree(
        MlxFastMetalKernelConfig cls
    );

    /// <summary>Adds an output argument (buffer) specification to a Metal kernel’s configuration.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_add_output_arg")]
    public static partial int MetalKernelConfigAddOutputArg(
        MlxFastMetalKernelConfig cls,
        int* shape,
        nuint size,
        MlxDType dtype
    );

    /// <summary>Sets the grid dimensions (total threads or threadgroups) for executing the Metal kernel.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_set_grid")]
    public static partial int MetalKernelConfigSetGrid(
        MlxFastMetalKernelConfig cls,
        int grid1,
        int grid2,
        int grid3
    );

    /// <summary>Sets the threadgroup size (threads per group) for the Metal kernel execution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_set_thread_group")]
    public static partial int MetalKernelConfigSetThreadGroup(
        MlxFastMetalKernelConfig cls,
        int thread1,
        int thread2,
        int thread3
    );

    /// <summary>Sets an initial value (if needed by the kernel, e.g. an initial accumulator or constant) in the kernel configuration.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_set_init_value")]
    public static partial int MetalKernelConfigSetInitValue(
        MlxFastMetalKernelConfig cls,
        float value
    );

    /// <summary>Enables or disables verbose logging for the Metal kernel execution (for debugging).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_config_set_verbose")]
    public static partial int MetalKernelConfigSetVerbose(
        MlxFastMetalKernelConfig cls,
        [MarshalAs(UnmanagedType.I1)] bool verbose
    );

    /// <summary>Sets a data-type template parameter for a Metal kernel configuration (choosing a type in shader code).</summary>
    [LibraryImport(
        Common.Lib,
        EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_dtype",
        StringMarshalling = StringMarshalling.Utf8)]
    public static partial int MetalKernelConfigAddTemplateArgDType(
        MlxFastMetalKernelConfig cls,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
        MlxDType dtype
    );

    /// <summary>Sets an integer template parameter for a Metal kernel configuration.</summary>
    [LibraryImport(
        Common.Lib,
        EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_int",
        StringMarshalling = StringMarshalling.Utf8)]
    public static partial int MetalKernelConfigAddTemplateArgInt(
        MlxFastMetalKernelConfig cls,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
        int value
    );

    /// <summary>Sets a boolean template parameter for a Metal kernel configuration (to specialize shader code).</summary>
    [LibraryImport(
        Common.Lib,
        EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_bool",
        StringMarshalling = StringMarshalling.Utf8)]
    public static partial int MetalKernelConfigAddTemplateArgBool(
        MlxFastMetalKernelConfig cls,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
        [MarshalAs(UnmanagedType.I1)] bool value
    );

    /// <summary>Compiles a new Metal compute kernel from source or predefined library and returns a handle to it.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxFastMetalKernel MetalKernelNew(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string name,
        MlxVectorStringHandle input_names,
        MlxVectorStringHandle output_names,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string source,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string header,
        [MarshalAs(UnmanagedType.I1)] bool ensure_row_contiguous,
        [MarshalAs(UnmanagedType.I1)] bool atomic_outputs
    );

    /// <summary>Releases the compiled Metal kernel object.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_free")]
    public static partial void MetalKernelFree(
        MlxFastMetalKernel cls
    );

    /// <summary>Launches (executes) a custom Metal shader/kernel with a given configuration on the GPU.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_metal_kernel_apply")]
    public static partial int MetalKernelApply(
        out MlxVectorArrayHandle outputs,
        MlxFastMetalKernel cls,
        MlxVectorArrayHandle inputs,
        MlxFastMetalKernelConfig config,
        MlxStreamHandle stream
    );

    /// <summary>Computes Root Mean Square normalization on the input (normalized by RMS of elements) using an optimized routine.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_rms_norm")]
    public static partial int RmsNorm(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxArrayHandle weight,
        float eps,
        MlxStreamHandle s
    );

    /// <summary>Applies rotary positional embedding (RoPE) to input sequences in a fast, vectorized manner (commonly used in transformer models).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_rope")]
    public static partial int Rope(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        int dims,
        [MarshalAs(UnmanagedType.I1)] bool traditional,
        MlxOptionalFloat @base,
        float scale,
        int offset,
        MlxArrayHandle freqs,
        MlxStreamHandle s
    );

    /// <summary>Computes scaled dot-product attention (queries * keys with scaling and softmax, applied to values) efficiently.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fast_scaled_dot_product_attention", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int ScaledDotProductAttention(
        out MlxArrayHandle res,
        MlxArrayHandle queries,
        MlxArrayHandle keys,
        MlxArrayHandle values,
        float scale,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string mask_mode,
        MlxVectorArrayHandle mask_arrs,
        MlxStreamHandle s
    );
}