// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxOps
{
    /// <summary>Computes the element-wise absolute value of the input array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_abs")]
    public static partial int Abs(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Adds two arrays element-wise, with NumPy-style broadcasting for mismatched shapes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_add")]
    public static partial int Add(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Performs matrix multiplication of two matrices (or batches of matrices), then adds the result to a third matrix, optionally with scaling factors.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_addmm")]
    public static partial int Addmm(
        out MlxArrayHandle res,
        MlxArrayHandle c,
        MlxArrayHandle a,
        MlxArrayHandle b,
        float alpha,
        float beta,
        MlxStreamHandle s
    );

    /// <summary>Returns an array of booleans, each indicating if all elements are True along the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_all_axes")]
    public static partial int AllAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the logical AND of elements along a given axis of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_all_axis")]
    public static partial int AllAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns True if all elements of the array are non-zero or True.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_all")]
    public static partial int All(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns True if two arrays are element-wise equal within a tolerance.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_allclose")]
    public static partial int Allclose(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        double rtol,
        double atol,
        [MarshalAs(UnmanagedType.I1)] bool equal_nan,
        MlxStreamHandle s
    );

    /// <summary>Returns an array of booleans indicating whether any element is True along the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_any_axes")]
    public static partial int AnyAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Checks if any element along a given axis is True.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_any_axis")]
    public static partial int AnyAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns True if any element of the array is non-zero or True.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_any")]
    public static partial int Any(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Creates a 1-D array of evenly spaced values within a given interval.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arange")]
    public static partial int Arange(
        out MlxArrayHandle res,
        double start,
        double stop,
        double step,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise arccosine (inverse cosine) of the input in radians.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arccos")]
    public static partial int Arccos(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise inverse hyperbolic cosine of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arccosh")]
    public static partial int Arccosh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise arcsine (inverse sine) of the input in radians.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arcsin")]
    public static partial int Arcsin(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise inverse hyperbolic sine of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arcsinh")]
    public static partial int Arcsinh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise arctangent of the input in radians.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arctan")]
    public static partial int Arctan(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes arctan(y/x) for corresponding elements of two arrays, preserving quadrant information.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arctan2")]
    public static partial int Arctan2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise inverse hyperbolic tangent of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_arctanh")]
    public static partial int Arctanh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Returns the indices of the maximum values along the specified axis of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argmax_axis")]
    public static partial int ArgmaxAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns the index of the maximum element in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argmax")]
    public static partial int Argmax(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns the indices of the minimum values along the specified axis of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argmin_axis")]
    public static partial int ArgminAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns the index of the minimum element in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argmin")]
    public static partial int Argmin(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Returns indices that would partition the array along a specified axis around the given kth element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argpartition_axis")]
    public static partial int ArgpartitionAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int kth,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Returns the indices that would partially sort the array such that the element at the given kth position is in sorted order.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argpartition")]
    public static partial int Argpartition(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int kth,
        MlxStreamHandle s
    );

    /// <summary>Returns the indices that would sort the array along a specified axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argsort_axis")]
    public static partial int ArgsortAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Returns the indices that would sort the array if it were flattened.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_argsort")]
    public static partial int Argsort(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Returns True if two arrays have the same shape and identical elements.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_equal")]
    public static partial int ArrayEqual(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        [MarshalAs(UnmanagedType.I1)] bool equal_nan,
        MlxStreamHandle s
    );

    /// <summary>Creates a new view of the array with the given shape and strides (without copying data).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_as_strided")]
    public static partial int AsStrided(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* shape,
        nuint shape_num,
        long* strides,
        nuint strides_num,
        nuint offset,
        MlxStreamHandle s
    );

    /// <summary>Casts (converts) an array to a new dtype, returning a new array of that type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_astype")]
    public static partial int Astype(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Ensures the input is at least 1-dimensional, converting scalars to 1D arrays and passing through arrays of higher dim unchanged.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_atleast_1d")]
    public static partial int Atleast1d(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Promotes the input to at least 2-dimensional form (e.g. turns 1D array into row vector).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_atleast_2d")]
    public static partial int Atleast2d(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Promotes the input to at least 3-dimensional form (e.g. adds extra dimensions of size 1 as needed).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_atleast_3d")]
    public static partial int Atleast3d(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the bitwise AND of two arrays element-wise.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_bitwise_and")]
    public static partial int BitwiseAnd(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the bitwise NOT (bitwise inversion) of an array element-wise.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_bitwise_invert")]
    public static partial int BitwiseInvert(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the bitwise OR of two arrays element-wise.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_bitwise_or")]
    public static partial int BitwiseOr(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the bitwise XOR of two arrays element-wise.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_bitwise_xor")]
    public static partial int BitwiseXor(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Performs a block-masked matrix multiplication (matrix multiplication where certain blocks or elements are masked out/ignored).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_block_masked_mm")]
    public static partial int BlockMaskedMm(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        int block_size,
        MlxArrayHandle mask_out,
        MlxArrayHandle mask_lhs,
        MlxArrayHandle mask_rhs,
        MlxStreamHandle s
    );

    /// <summary>Broadcasts a list of arrays against each other to make them compatible in shape (returns copies/views of each array in a common shape).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_broadcast_arrays")]
    public static partial int BroadcastArrays(
        out MlxVectorArrayHandle res,
        MlxVectorArrayHandle inputs,
        MlxStreamHandle s
    );

    /// <summary>Broadcasts a single array to a new shape (replicating its data as needed).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_broadcast_to")]
    public static partial int BroadcastTo(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* shape,
        nuint shape_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the ceiling of each element (rounds each value up to the nearest integer).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_ceil")]
    public static partial int Ceil(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Clips array values to a specified interval, setting values below a min to the min and above a max to the max.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_clip")]
    public static partial int Clip(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle a_min,
        MlxArrayHandle a_max,
        MlxStreamHandle s
    );

    /// <summary>Concatenates multiple arrays along a specified existing axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_concatenate_axis")]
    public static partial int ConcatenateAxis(
        out MlxArrayHandle res,
        MlxVectorArrayHandle arrays,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Concatenates a sequence of arrays along a new axis 0 (stacks them vertically by default).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_concatenate")]
    public static partial int Concatenate(
        out MlxArrayHandle res,
        MlxVectorArrayHandle arrays,
        MlxStreamHandle s
    );

    /// <summary>Returns the complex conjugate of each element in a complex array (flips the sign of the imaginary parts).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conjugate")]
    public static partial int Conjugate(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Returns a contiguous copy of the array in memory (ensuring row-major contiguous layout).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_contiguous")]
    public static partial int Contiguous(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool allow_col_major,
        MlxStreamHandle s
    );

    /// <summary>Performs a one-dimensional convolution between an input and a filter (with optional stride, padding as applicable).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv1d")]
    public static partial int Conv1d(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int stride,
        int padding,
        int dilation,
        int groups,
        MlxStreamHandle s
    );

    /// <summary>Performs a two-dimensional convolution (e.g. image convolution with a 2D kernel).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv2d")]
    public static partial int Conv2d(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int stride_0,
        int stride_1,
        int padding_0,
        int padding_1,
        int dilation_0,
        int dilation_1,
        int groups,
        MlxStreamHandle s
    );

    /// <summary>Performs a three-dimensional convolution on volumetric data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv3d")]
    public static partial int Conv3d(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int stride_0,
        int stride_1,
        int stride_2,
        int padding_0,
        int padding_1,
        int padding_2,
        int dilation_0,
        int dilation_1,
        int dilation_2,
        int groups,
        MlxStreamHandle s
    );

    /// <summary>Performs an N-dimensional convolution or cross-correlation with general specified parameters (dimension-agnostic conv).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv_general")]
    public static partial int ConvGeneral(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int* stride,
        nuint stride_num,
        int* padding_lo,
        nuint padding_lo_num,
        int* padding_hi,
        nuint padding_hi_num,
        int* kernel_dilation,
        nuint kernel_dilation_num,
        int* input_dilation,
        nuint input_dilation_num,
        int groups,
        [MarshalAs(UnmanagedType.I1)] bool flip,
        MlxStreamHandle s
    );

    /// <summary>Performs a one-dimensional transposed convolution (deconvolution) operation.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv_transpose1d")]
    public static partial int ConvTranspose1d(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int stride,
        int padding,
        int dilation,
        int output_padding,
        int groups,
        MlxStreamHandle s
    );

    /// <summary>Performs a two-dimensional transposed convolution (deconvolution).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv_transpose2d")]
    public static partial int ConvTranspose2d(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int stride_0,
        int stride_1,
        int padding_0,
        int padding_1,
        int dilation_0,
        int dilation_1,
        int output_padding_0,
        int output_padding_1,
        int groups,
        MlxStreamHandle s
    );

    /// <summary>Performs a three-dimensional transposed convolution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_conv_transpose3d")]
    public static partial int ConvTranspose3d(
        out MlxArrayHandle res,
        MlxArrayHandle input,
        MlxArrayHandle weight,
        int stride_0,
        int stride_1,
        int stride_2,
        int padding_0,
        int padding_1,
        int padding_2,
        int dilation_0,
        int dilation_1,
        int dilation_2,
        int output_padding_0,
        int output_padding_1,
        int output_padding_2,
        int groups,
        MlxStreamHandle s
    );

    /// <summary>Creates a deep copy of the given array (allocating new memory).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_copy")]
    public static partial int Copy(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise cosine of the input (input in radians).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cos")]
    public static partial int Cos(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise hyperbolic cosine of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cosh")]
    public static partial int Cosh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the cumulative maximum of array elements along a given axis (each element of result is the max up to that position).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cummax")]
    public static partial int Cummax(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool reverse,
        [MarshalAs(UnmanagedType.I1)] bool inclusive,
        MlxStreamHandle s
    );

    /// <summary>Computes the cumulative minimum of array elements along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cummin")]
    public static partial int Cummin(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool reverse,
        [MarshalAs(UnmanagedType.I1)] bool inclusive,
        MlxStreamHandle s
    );

    /// <summary>Computes the cumulative product of array elements along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cumprod")]
    public static partial int Cumprod(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool reverse,
        [MarshalAs(UnmanagedType.I1)] bool inclusive,
        MlxStreamHandle s
    );

    /// <summary>Computes the cumulative sum of array elements along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_cumsum")]
    public static partial int Cumsum(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool reverse,
        [MarshalAs(UnmanagedType.I1)] bool inclusive,
        MlxStreamHandle s
    );

    /// <summary>Converts each element from radians to degrees (element-wise).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_degrees")]
    public static partial int Degrees(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Creates a dependency between arrays: ensures that one array’s computation is sequenced after another (without using its values).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_depends")]
    public static partial int Depends(
        out MlxVectorArrayHandle res,
        MlxVectorArrayHandle inputs,
        MlxVectorArrayHandle dependencies
    );

    /// <summary>Converts a quantized array (with given scale/zero-point or similar parameters) back to floating-point values.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_dequantize")]
    public static partial int Dequantize(
        out MlxArrayHandle res,
        MlxArrayHandle w,
        MlxArrayHandle scales,
        MlxArrayHandle biases,
        int group_size,
        int bits,
        MlxStreamHandle s
    );

    /// <summary>Constructs a diagonal matrix from a 1-D array (places the input on the diagonal of a new matrix).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_diag")]
    public static partial int Diag(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int k,
        MlxStreamHandle s
    );

    /// <summary>Extracts the diagonal elements of a 2-D array (or a specified diagonal of a higher-dimensional array).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_diagonal")]
    public static partial int Diagonal(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int offset,
        int axis1,
        int axis2,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise division of one array by another or by a scalar.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_divide")]
    public static partial int Divide(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes element-wise quotient and remainder of division.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_divmod")]
    public static partial int Divmod(
        out MlxVectorArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Performs Einstein summation on the given operands.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_einsum", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Einsum(
        out MlxArrayHandle res,
        string subscripts,
        MlxVectorArrayHandle operands,
        MlxStreamHandle s
    );

    /// <summary>Returns a boolean array indicating where two arrays are exactly equal.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_equal")]
    public static partial int Equal(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the Gaussian error function of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_erf")]
    public static partial int Erf(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse error function of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_erfinv")]
    public static partial int Erfinv(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise exponential.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_exp")]
    public static partial int Exp(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Expands the array’s shape by inserting new axes at specified positions.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_expand_dims_axes")]
    public static partial int ExpandDimsAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Inserts a new axis of length 1 at the given position.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_expand_dims")]
    public static partial int ExpandDims(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes exp(x) - 1 with improved precision for small x.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_expm1")]
    public static partial int Expm1(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Creates a 2-D identity matrix with ones on the diagonal.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_eye")]
    public static partial int Eye(
        out MlxArrayHandle res,
        int n,
        int m,
        int k,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Flattens the array into 1-D between the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_flatten")]
    public static partial int Flatten(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int start_axis,
        int end_axis,
        MlxStreamHandle s
    );

    /// <summary>Rounds each element down to the nearest integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_floor")]
    public static partial int Floor(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Performs element-wise integer division rounding toward negative infinity.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_floor_divide")]
    public static partial int FloorDivide(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Creates a new array of given shape filled with the provided value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_full")]
    public static partial int Full(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxArrayHandle vals,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Gathers slices from an array using indices along specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_gather")]
    public static partial int Gather(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxVectorArrayHandle indices,
        int* axes,
        nuint axes_num,
        int* slice_sizes,
        nuint slice_sizes_num,
        MlxStreamHandle s
    );

    /// <summary>Performs matrix multiplication and gathers selected rows or columns.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_gather_mm")]
    public static partial int GatherMm(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxArrayHandle lhs_indices,
        MlxArrayHandle rhs_indices,
        [MarshalAs(UnmanagedType.I1)] bool sorted_indices,
        MlxStreamHandle s
    );

    /// <summary>Performs matrix multiplication with quantized weights and gathers results.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_gather_qmm")]
    public static partial int GatherQmm(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxArrayHandle w,
        MlxArrayHandle scales,
        MlxArrayHandle biases,
        MlxArrayHandle lhs_indices,
        MlxArrayHandle rhs_indices,
        [MarshalAs(UnmanagedType.I1)] bool transpose,
        int group_size,
        int bits,
        [MarshalAs(UnmanagedType.I1)] bool sorted_indices,
        MlxStreamHandle s
    );

    /// <summary>Performs matrix multiplication independently on each segment.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_segmented_mm")]
    public static partial int SegmentedMm(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxArrayHandle segments,
        MlxStreamHandle s
    );

    /// <summary>Returns a boolean array where a is greater than b.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_greater")]
    public static partial int Greater(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Returns a boolean array where a is greater than or equal to b.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_greater_equal")]
    public static partial int GreaterEqual(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Applies the Walsh-Hadamard transform to the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_hadamard_transform")]
    public static partial int HadamardTransform(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxOptionalFloat scale,
        MlxStreamHandle s
    );

    /// <summary>Returns an identity matrix of given size.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_identity")]
    public static partial int Identity(
        out MlxArrayHandle res,
        int n,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Returns the imaginary component of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_imag")]
    public static partial int Imag(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the inner product of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_inner")]
    public static partial int Inner(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Returns True where two arrays are element-wise close within tolerances.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_isclose")]
    public static partial int Isclose(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        double rtol,
        double atol,
        [MarshalAs(UnmanagedType.I1)] bool equal_nan,
        MlxStreamHandle s
    );

    /// <summary>Indicates which elements are finite (not NaN or ±Inf).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_isfinite")]
    public static partial int Isfinite(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Indicates which elements are positive or negative infinity.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_isinf")]
    public static partial int Isinf(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Indicates which elements are NaN.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_isnan")]
    public static partial int Isnan(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Indicates which elements are negative infinity.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_isneginf")]
    public static partial int Isneginf(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Indicates which elements are positive infinity.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_isposinf")]
    public static partial int Isposinf(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the Kronecker product of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_kron")]
    public static partial int Kron(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Performs element-wise left bit shift.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_left_shift")]
    public static partial int LeftShift(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Returns a boolean array where a is less than b.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_less")]
    public static partial int Less(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Returns a boolean array where a is less than or equal to b.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_less_equal")]
    public static partial int LessEqual(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Returns evenly spaced numbers over a specified interval.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linspace")]
    public static partial int Linspace(
        out MlxArrayHandle res,
        double start,
        double stop,
        int num,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Computes the natural logarithm of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_log")]
    public static partial int Log(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the base-10 logarithm of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_log10")]
    public static partial int Log10(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes log(1 + x) for each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_log1p")]
    public static partial int Log1p(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the base-2 logarithm of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_log2")]
    public static partial int Log2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes log(exp(a) + exp(b)) element-wise in a stable manner.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logaddexp")]
    public static partial int Logaddexp(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes cumulative log-sum-exp along an axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logcumsumexp")]
    public static partial int Logcumsumexp(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool reverse,
        [MarshalAs(UnmanagedType.I1)] bool inclusive,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise logical AND of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logical_and")]
    public static partial int LogicalAnd(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the logical NOT of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logical_not")]
    public static partial int LogicalNot(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise logical OR of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logical_or")]
    public static partial int LogicalOr(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes log-sum-exp over the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logsumexp_axes")]
    public static partial int LogsumexpAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes log-sum-exp along a single axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logsumexp_axis")]
    public static partial int LogsumexpAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes log-sum-exp over all elements.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_logsumexp")]
    public static partial int Logsumexp(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Performs matrix multiplication of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_matmul")]
    public static partial int Matmul(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes maxima over specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_max_axes")]
    public static partial int MaxAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the maximum values along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_max_axis")]
    public static partial int MaxAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Finds the maximum value of all elements.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_max")]
    public static partial int Max(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Element-wise maximum of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_maximum")]
    public static partial int Maximum(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the mean over specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_mean_axes")]
    public static partial int MeanAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the mean along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_mean_axis")]
    public static partial int MeanAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the mean of all elements.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_mean")]
    public static partial int Mean(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Generates coordinate matrices from coordinate vectors.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_meshgrid", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Meshgrid(
        out MlxVectorArrayHandle res,
        MlxVectorArrayHandle arrays,
        [MarshalAs(UnmanagedType.I1)] bool sparse,
        string indexing,
        MlxStreamHandle s
    );

    /// <summary>Computes the minima over the specified axes of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_min_axes")]
    public static partial int MinAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the minimum values along a given axis of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_min_axis")]
    public static partial int MinAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the minimum value among all elements of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_min")]
    public static partial int Min(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise minimum of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_minimum")]
    public static partial int Minimum(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Moves an axis from one position to another in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_moveaxis")]
    public static partial int Moveaxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int source,
        int destination,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise product of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_multiply")]
    public static partial int Multiply(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Replaces NaN with zero and infinities with large finite numbers.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_nan_to_num")]
    public static partial int NanToNum(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        float nan,
        MlxOptionalFloat posinf,
        MlxOptionalFloat neginf,
        MlxStreamHandle s
    );

    /// <summary>Computes the numerical negation of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_negative")]
    public static partial int Negative(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Returns a boolean array of element-wise inequality.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_not_equal")]
    public static partial int NotEqual(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Returns the total number of elements in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_number_of_elements")]
    public static partial int NumberOfElements(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool inverted,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Creates a new array of given shape and dtype, filled with ones.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_ones")]
    public static partial int Ones(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Creates a new array of ones with the same shape and dtype as a reference array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_ones_like")]
    public static partial int OnesLike(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the outer product of two one-dimensional arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_outer")]
    public static partial int Outer(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Pads an array on each side according to the specified mode and pad widths.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_pad", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Pad(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        int* low_pad_size,
        nuint low_pad_size_num,
        int* high_pad_size,
        nuint high_pad_size_num,
        MlxArrayHandle pad_value,
        string mode,
        MlxStreamHandle s
    );

    /// <summary>Pads an array symmetrically by reflecting its edges.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_pad_symmetric", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int PadSymmetric(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int pad_width,
        MlxArrayHandle pad_value,
        string mode,
        MlxStreamHandle s
    );

    /// <summary>Partitions the array along a specific axis around a given k-th element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_partition_axis")]
    public static partial int PartitionAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int kth,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Rearranges elements so that those before the k-th position are smaller and those after are larger.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_partition")]
    public static partial int Partition(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int kth,
        MlxStreamHandle s
    );

    /// <summary>Raises one array to the power of another, element-wise.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_power")]
    public static partial int Power(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the product of elements over the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_prod_axes")]
    public static partial int ProdAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the product of elements along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_prod_axis")]
    public static partial int ProdAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the product of all elements in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_prod")]
    public static partial int Prod(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Writes values into an array along a given axis at specified indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_put_along_axis")]
    public static partial int PutAlongAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle indices,
        MlxArrayHandle values,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Quantizes a matrix using the specified bit-width and group size.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_quantize")]
    public static partial int Quantize(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        out MlxArrayHandle res_2,
        MlxArrayHandle w,
        int group_size,
        int bits,
        MlxStreamHandle s
    );

    /// <summary>Performs matrix multiplication where operands are quantized, using provided scales and biases.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_quantized_matmul")]
    public static partial int QuantizedMatmul(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxArrayHandle w,
        MlxArrayHandle scales,
        MlxArrayHandle biases,
        [MarshalAs(UnmanagedType.I1)] bool transpose,
        int group_size,
        int bits,
        MlxStreamHandle s
    );

    /// <summary>Converts each element from degrees to radians.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_radians")]
    public static partial int Radians(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Returns the real part of a complex array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_real")]
    public static partial int Real(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the reciprocal of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_reciprocal")]
    public static partial int Reciprocal(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise remainder of division.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_remainder")]
    public static partial int Remainder(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Repeats the elements of an array along a specified axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_repeat_axis")]
    public static partial int RepeatAxis(
        out MlxArrayHandle res,
        MlxArrayHandle arr,
        int repeats,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Repeats the elements of an array a specified number of times.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_repeat")]
    public static partial int Repeat(
        out MlxArrayHandle res,
        MlxArrayHandle arr,
        int repeats,
        MlxStreamHandle s
    );

    /// <summary>Gives a new shape to an array without changing its data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_reshape")]
    public static partial int Reshape(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* shape,
        nuint shape_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise right bit shift of an integer array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_right_shift")]
    public static partial int RightShift(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Rolls the elements of the array along one axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_roll_axis")]
    public static partial int RollAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* shift,
        nuint shift_num,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Rolls the elements of the array along multiple axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_roll_axes")]
    public static partial int RollAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* shift,
        nuint shift_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Rolls the array elements, shifting and wrapping around.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_roll")]
    public static partial int Roll(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* shift,
        nuint shift_num,
        MlxStreamHandle s
    );

    /// <summary>Rounds each element to the nearest integer or specified decimals.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_round")]
    public static partial int Round(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int decimals,
        MlxStreamHandle s
    );

    /// <summary>Computes the reciprocal square root of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_rsqrt")]
    public static partial int Rsqrt(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Writes values into an array at specified indices along given axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_scatter")]
    public static partial int Scatter(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxVectorArrayHandle indices,
        MlxArrayHandle updates,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Adds values into an array at specified indices, summing with existing data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_scatter_add")]
    public static partial int ScatterAdd(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxVectorArrayHandle indices,
        MlxArrayHandle updates,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Performs scatter-add along a specific axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_scatter_add_axis")]
    public static partial int ScatterAddAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle indices,
        MlxArrayHandle values,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Updates array elements with the maximum of existing and new values at given indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_scatter_max")]
    public static partial int ScatterMax(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxVectorArrayHandle indices,
        MlxArrayHandle updates,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Updates array elements with the minimum of existing and new values at given indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_scatter_min")]
    public static partial int ScatterMin(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxVectorArrayHandle indices,
        MlxArrayHandle updates,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Multiplies values into an array at specified indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_scatter_prod")]
    public static partial int ScatterProd(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxVectorArrayHandle indices,
        MlxArrayHandle updates,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the logistic sigmoid function element-wise.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sigmoid")]
    public static partial int Sigmoid(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the sign of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sign")]
    public static partial int Sign(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise sine of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sin")]
    public static partial int Sin(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise hyperbolic sine of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sinh")]
    public static partial int Sinh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Extracts a sub-array using static start indices and sizes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_slice")]
    public static partial int Slice(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* start,
        nuint start_num,
        int* stop,
        nuint stop_num,
        int* strides,
        nuint strides_num,
        MlxStreamHandle s
    );

    /// <summary>Extracts a sub-array using dynamic start indices and sizes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_slice_dynamic")]
    public static partial int SliceDynamic(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle start,
        int* axes,
        nuint axes_num,
        int* slice_size,
        nuint slice_size_num,
        MlxStreamHandle s
    );

    /// <summary>Writes an update array into a slice defined by static indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_slice_update")]
    public static partial int SliceUpdate(
        out MlxArrayHandle res,
        MlxArrayHandle src,
        MlxArrayHandle update,
        int* start,
        nuint start_num,
        int* stop,
        nuint stop_num,
        int* strides,
        nuint strides_num,
        MlxStreamHandle s
    );

    /// <summary>Writes an update array into positions defined by dynamic start indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_slice_update_dynamic")]
    public static partial int SliceUpdateDynamic(
        out MlxArrayHandle res,
        MlxArrayHandle src,
        MlxArrayHandle update,
        MlxArrayHandle start,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes softmax over multiple specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_softmax_axes")]
    public static partial int SoftmaxAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool precise,
        MlxStreamHandle s
    );

    /// <summary>Computes softmax along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_softmax_axis")]
    public static partial int SoftmaxAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool precise,
        MlxStreamHandle s
    );

    /// <summary>Computes the softmax of the array along the default axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_softmax")]
    public static partial int Softmax(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool precise,
        MlxStreamHandle s
    );

    /// <summary>Returns a sorted copy of the array along a specified axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sort_axis")]
    public static partial int SortAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Returns a sorted copy of the flattened array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sort")]
    public static partial int Sort(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Splits an array into multiple sub-arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_split")]
    public static partial int Split(
        out MlxVectorArrayHandle res,
        MlxArrayHandle a,
        int num_splits,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Splits an array into a specified number of equal sections along an axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_split_sections")]
    public static partial int SplitSections(
        out MlxVectorArrayHandle res,
        MlxArrayHandle a,
        int* indices,
        nuint indices_num,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the square root of each element.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sqrt")]
    public static partial int Sqrt(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise square of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_square")]
    public static partial int Square(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Removes specific axes of length one from the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_squeeze_axes")]
    public static partial int SqueezeAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Removes a single axis of length one from the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_squeeze_axis")]
    public static partial int SqueezeAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Removes all axes of length one from the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_squeeze")]
    public static partial int Squeeze(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Stacks multiple arrays along a specified axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stack_axis")]
    public static partial int StackAxis(
        out MlxArrayHandle res,
        MlxVectorArrayHandle arrays,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Stacks a sequence of arrays along a new axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stack")]
    public static partial int Stack(
        out MlxArrayHandle res,
        MlxVectorArrayHandle arrays,
        MlxStreamHandle s
    );

    /// <summary>Computes the standard deviation over the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_std_axes")]
    public static partial int StdAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        int ddof,
        MlxStreamHandle s
    );

    /// <summary>Computes the standard deviation along a particular axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_std_axis")]
    public static partial int StdAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        int ddof,
        MlxStreamHandle s
    );

    /// <summary>Computes the standard deviation of all elements in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_std")]
    public static partial int Std(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        int ddof,
        MlxStreamHandle s
    );

    /// <summary>Stops gradient computation for the given array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_stop_gradient")]
    public static partial int StopGradient(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise difference of two arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_subtract")]
    public static partial int Subtract(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Computes the sum over the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sum_axes")]
    public static partial int SumAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the sum along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sum_axis")]
    public static partial int SumAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the sum of all elements in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_sum")]
    public static partial int Sum(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Swaps two axes of an array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_swapaxes")]
    public static partial int Swapaxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis1,
        int axis2,
        MlxStreamHandle s
    );

    /// <summary>Takes elements from an array along the specified axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_take_axis")]
    public static partial int TakeAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle indices,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Takes elements from the array at the given indices.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_take")]
    public static partial int Take(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle indices,
        MlxStreamHandle s
    );

    /// <summary>Takes elements along a given axis at specified positions.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_take_along_axis")]
    public static partial int TakeAlongAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle indices,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise tangent of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tan")]
    public static partial int Tan(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the element-wise hyperbolic tangent of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tanh")]
    public static partial int Tanh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the tensor dot product along specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tensordot")]
    public static partial int Tensordot(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        int* axes_a,
        nuint axes_a_num,
        int* axes_b,
        nuint axes_b_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the tensordot of two arrays contracting along a single axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tensordot_axis")]
    public static partial int TensordotAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Constructs a new array by repeating the input along each axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tile")]
    public static partial int Tile(
        out MlxArrayHandle res,
        MlxArrayHandle arr,
        int* reps,
        nuint reps_num,
        MlxStreamHandle s
    );

    /// <summary>Returns the top-k largest elements along a specified axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_topk_axis")]
    public static partial int TopkAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int k,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Returns the top-k largest elements of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_topk")]
    public static partial int Topk(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int k,
        MlxStreamHandle s
    );

    /// <summary>Computes the trace of a matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_trace")]
    public static partial int Trace(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int offset,
        int axis1,
        int axis2,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Permutes the array’s axes to a given order.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_transpose_axes")]
    public static partial int TransposeAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Transposes the array dimensions.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_transpose")]
    public static partial int Transpose(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Constructs a matrix with ones on and below a given diagonal.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tri")]
    public static partial int Tri(
        out MlxArrayHandle res,
        int n,
        int m,
        int k,
        MlxDType type,
        MlxStreamHandle s
    );

    /// <summary>Returns the lower-triangular part of a matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_tril")]
    public static partial int Tril(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        int k,
        MlxStreamHandle s
    );

    /// <summary>Returns the upper-triangular part of a matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_triu")]
    public static partial int Triu(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        int k,
        MlxStreamHandle s
    );

    /// <summary>Expands a single dimension into multiple dimensions with given sizes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_unflatten")]
    public static partial int Unflatten(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        int* shape,
        nuint shape_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the variance over the specified axes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_var_axes")]
    public static partial int VarAxes(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axes,
        nuint axes_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        int ddof,
        MlxStreamHandle s
    );

    /// <summary>Computes the variance along a given axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_var_axis")]
    public static partial int VarAxis(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int axis,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        int ddof,
        MlxStreamHandle s
    );

    /// <summary>Computes the variance of all elements in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_var")]
    public static partial int Var(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        int ddof,
        MlxStreamHandle s
    );

    /// <summary>Creates a view of the array sharing the same data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_view")]
    public static partial int View(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Selects elements from two arrays based on a condition.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_where")]
    public static partial int Where(
        out MlxArrayHandle res,
        MlxArrayHandle condition,
        MlxArrayHandle x,
        MlxArrayHandle y,
        MlxStreamHandle s
    );

    /// <summary>Creates a new array of given shape and dtype, filled with zeros.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_zeros")]
    public static partial int Zeros(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxStreamHandle s
    );

    /// <summary>Creates a new array of zeros with the same shape and dtype as a reference array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_zeros_like")]
    public static partial int ZerosLike(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );
}