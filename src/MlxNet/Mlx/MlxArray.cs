// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public enum MlxDType
{
    MLX_BOOL,
    MLX_UINT8,
    MLX_UINT16,
    MLX_UINT32,
    MLX_UINT64,
    MLX_INT8,
    MLX_INT16,
    MLX_INT32,
    MLX_INT64,
    MLX_FLOAT16,
    MLX_FLOAT32,
    MLX_FLOAT64,
    MLX_BFLOAT16,
    MLX_COMPLEX64
}

public static unsafe partial class MlxArray
{
    /// <summary>Returns the size in bytes of a given data type (Dtype).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_dtype_size")]
    public static partial nuint DTypeSize(
        MlxDType dtype
    );

    /// <summary>Returns a string representation of the array’s contents.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_tostring")]
    public static partial int ToString(
        out MlxStringHandle str,
        MlxArrayHandle arr
    );

    /// <summary>Creates a new uninitialized array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new")]
    public static partial MlxArrayHandle New();

    /// <summary>Frees the memory and resources associated with an array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_free")]
    public static partial int Free(
        MlxArrayHandle arr
    );

    /// <summary>Creates a new boolean array of the specified shape.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_bool")]
    public static partial MlxArrayHandle NewBool(
        [MarshalAs(UnmanagedType.I1)] bool val
    );

    /// <summary>Creates a new integer array of the default integer dtype.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_int")]
    public static partial MlxArrayHandle NewInt(
        int val
    );

    /// <summary>Creates a new array of dtype float32.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_float32")]
    public static partial MlxArrayHandle NewFloat32(
        float val
    );

    /// <summary>Creates a new single-precision floating-point array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_float")]
    public static partial MlxArrayHandle NewFloat(
        float val
    );

    /// <summary>Creates a new array of dtype float64.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_float64")]
    public static partial MlxArrayHandle NewFloat64(
        double val
    );

    /// <summary>Creates a new double-precision floating-point array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_double")]
    public static partial MlxArrayHandle NewDouble(
        double val
    );

    /// <summary>Creates a new complex-number array using complex64 dtype.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_complex")]
    public static partial MlxArrayHandle NewComplex(
        float real_val,
        float imag_val
    );

    /// <summary>Creates a new array using provided data buffer with the specified shape and dtype.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_new_data")]
    public static partial MlxArrayHandle NewData(
        void* data,
        int* shape,
        int dim,
        MlxDType dtype
    );

    /// <summary>Sets all elements of the array to the value of another array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set")]
    public static partial int Set(
        ref MlxArrayHandle arr,
        MlxArrayHandle src
    );

    /// <summary>Fills the array with a given boolean value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_bool")]
    public static partial int SetBool(
        ref MlxArrayHandle arr,
        [MarshalAs(UnmanagedType.I1)] bool val
    );

    /// <summary>Fills the array with a given integer value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_int")]
    public static partial int SetInt(
        ref MlxArrayHandle arr,
        int val
    );

    /// <summary>Sets all elements of the array to the given float32 value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_float32")]
    public static partial int SetFloat32(
        ref MlxArrayHandle arr,
        float val
    );

    /// <summary>Fills the array with a given single-precision float value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_float")]
    public static partial int SetFloat(
        ref MlxArrayHandle arr,
        float val
    );

    /// <summary>Sets all elements of the array to the given float64 value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_float64")]
    public static partial int SetFloat64(
        ref MlxArrayHandle arr,
        double val
    );

    /// <summary>Fills the array with a given double-precision floating-point value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_double")]
    public static partial int SetDouble(
        ref MlxArrayHandle arr,
        double val
    );

    /// <summary>Fills the array with a given complex value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_complex")]
    public static partial int SetComplex(
        ref MlxArrayHandle arr,
        float real_val,
        float imag_val
    );

    /// <summary>Copies data from a buffer into the array, overwriting its contents.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_set_data")]
    public static partial int SetData(
        ref MlxArrayHandle arr,
        void* data,
        int* shape,
        int dim,
        MlxDType dtype
    );

    /// <summary>Returns the size in bytes of each element in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_itemsize")]
    public static partial nuint Itemsize(
        MlxArrayHandle arr
    );

    /// <summary>Returns the total number of elements in the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_size")]
    public static partial nuint Size(
        MlxArrayHandle arr
    );

    /// <summary>Returns the total number of bytes occupied by the array’s data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_nbytes")]
    public static partial nuint Nbytes(
        MlxArrayHandle arr
    );

    /// <summary>Returns the number of dimensions of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_ndim")]
    public static partial nuint Ndim(
        MlxArrayHandle arr
    );

    /// <summary>Returns the shape of the array as a sequence of dimension sizes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_shape")]
    public static partial int* Shape(
        MlxArrayHandle arr
    );

    /// <summary>Returns the stride of the array for each dimension.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_strides")]
    public static partial nuint* Strides(
        MlxArrayHandle arr
    );

    /// <summary>Returns the size of the specified dimension of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_dim")]
    public static partial int Dim(
        MlxArrayHandle arr,
        int dim
    );

    /// <summary>Returns the data type of the array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_dtype")]
    public static partial MlxDType DType(
        MlxArrayHandle arr
    );

    /// <summary>Forces evaluation of a lazy array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_eval")]
    public static partial int Eval(
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a boolean array as a bool.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_bool")]
    public static partial int ItemBool(
        out byte res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a uint8 array as an 8-bit unsigned integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_uint8")]
    public static partial int ItemUint8(
        out byte res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a uint16 array as a 16-bit unsigned integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_uint16")]
    public static partial int ItemUint16(
        out ushort res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a uint32 array as a 32-bit unsigned integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_uint32")]
    public static partial int ItemUint32(
        out uint res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a uint64 array as a 64-bit unsigned integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_uint64")]
    public static partial int ItemUint64(
        out ulong res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of an int8 array as an 8-bit integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_int8")]
    public static partial int ItemInt8(
        out sbyte res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of an int16 array as a 16-bit integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_int16")]
    public static partial int ItemInt16(
        out short res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of an int32 array as a 32-bit integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_int32")]
    public static partial int ItemInt32(
        out int res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of an int64 array as a 64-bit integer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_int64")]
    public static partial int ItemInt64(
        out long res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a float array as a 32-bit float value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_float32")]
    public static partial int ItemFloat32(
        out float res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a float array as a 64-bit float value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_float64")]
    public static partial int ItemFloat64(
        out double res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a complex array as a complex64 value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_complex64")]
    public static partial int ItemComplex64(
        out Complex64 res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of a half-precision float array as an FP16 value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_float16")]
    public static partial int ItemFloat16(
        out ushort res,
        MlxArrayHandle arr
    );

    /// <summary>Returns the single element of an array as a bfloat16 value.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_item_bfloat16")]
    public static partial int ItemBfloat16(
        out ushort res,
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to bool type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_bool")]
    public static partial byte* DataBool(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 8-bit unsigned integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_uint8")]
    public static partial byte* DataUint8(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 16-bit unsigned integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_uint16")]
    public static partial ushort* DataUint16(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 32-bit unsigned integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_uint32")]
    public static partial uint* DataUint32(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 64-bit unsigned integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_uint64")]
    public static partial ulong* DataUint64(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 8-bit signed integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_int8")]
    public static partial sbyte* DataInt8(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 16-bit signed integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_int16")]
    public static partial short* DataInt16(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 32-bit signed integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_int32")]
    public static partial int* DataInt32(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 64-bit signed integer type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_int64")]
    public static partial long* DataInt64(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 32-bit float type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_float32")]
    public static partial float* DataFloat32(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 64-bit float type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_float64")]
    public static partial double* DataFloat64(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to complex64 type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_complex64")]
    public static partial Complex64* DataComplex64(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to 16-bit float type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_float16")]
    public static partial ushort* DataFloat16(
        MlxArrayHandle arr
    );

    /// <summary>Returns a pointer to the array’s data cast to the bfloat16 type.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_array_data_bfloat16")]
    public static partial ushort* DataBfloat16(
        MlxArrayHandle arr
    );

    /// <summary>Checks whether the array value is available.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "_mlx_array_is_available")]
    public static partial int _mlx_array_is_available(
        out byte res,
        MlxArrayHandle arr
    );

    /// <summary>Blocks until the array computation completes.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "_mlx_array_wait")]
    public static partial int _mlx_array_wait(
        MlxArrayHandle arr
    );

    /// <summary>Checks if the array memory is contiguous.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "_mlx_array_is_contiguous")]
    public static partial int _mlx_array_is_contiguous(
        out byte res,
        MlxArrayHandle arr
    );

    /// <summary>Checks if the array is contiguous by rows.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "_mlx_array_is_row_contiguous")]
    public static partial int _mlx_array_is_row_contiguous(
        out byte res,
        MlxArrayHandle arr
    );

    /// <summary>Checks if the array is contiguous by columns.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "_mlx_array_is_col_contiguous")]
    public static partial int _mlx_array_is_col_contiguous(
        out byte res,
        MlxArrayHandle arr
    );
}