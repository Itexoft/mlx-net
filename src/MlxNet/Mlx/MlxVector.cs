// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxVector
{
    /// <summary>
    /// Creates a new dynamic vector to hold MLX array elements.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_new")]
    public static partial MlxVectorArrayHandle ArrayNew();

    /// <summary>
    /// Sets the element at a specific index in the vector-of-arrays to a given MLX array.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_set")]
    public static partial int ArraySet(
        ref MlxVectorArrayHandle vec,
        MlxVectorArrayHandle src
    );

    /// <summary>
    /// Frees a dynamic vector that holds MLX array objects.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_free")]
    public static partial int ArrayFree(
        MlxVectorArrayHandle vec
    );

    /// <summary>
    /// Creates a new vector-of-arrays initialized with a copy of an existing array of MLX arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_new_data")]
    public static partial MlxVectorArrayHandle ArrayNewData(
        MlxArrayHandle* data,
        nuint size
    );

    /// <summary>
    /// Creates a new vector-of-arrays containing a single given MLX array as the initial element.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_new_value")]
    public static partial MlxVectorArrayHandle ArrayNewValue(
        MlxArrayHandle val
    );

    /// <summary>
    /// Copies a sequence of MLX array handles into the vector starting at a given index.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_set_data")]
    public static partial int ArraySetData(
        ref MlxVectorArrayHandle vec,
        MlxArrayHandle* data,
        nuint size
    );

    /// <summary>
    /// Sets the element at a specified index in the vector to an MLX array.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_set_value")]
    public static partial int ArraySetValue(
        ref MlxVectorArrayHandle vec,
        MlxArrayHandle val
    );

    /// <summary>
    /// Appends multiple arrays from a C array of MLX array handles to the end of a vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_append_data")]
    public static partial int ArrayAppendData(
        MlxVectorArrayHandle vec,
        MlxArrayHandle* data,
        nuint size
    );

    /// <summary>
    /// Appends a single MLX array to the end of a vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_append_value")]
    public static partial int ArrayAppendValue(
        MlxVectorArrayHandle vec,
        MlxArrayHandle val
    );

    /// <summary>
    /// Returns the number of elements in the vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_size")]
    public static partial nuint ArraySize(
        MlxVectorArrayHandle vec
    );

    /// <summary>
    /// Retrieves the MLX array at a given index in the vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_array_get")]
    public static partial int ArrayGet(
        out MlxArrayHandle res,
        MlxVectorArrayHandle vec,
        nuint idx
    );

    /// <summary>
    /// Creates a new dynamic vector intended to hold multiple vector-of-array objects.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_new")]
    public static partial MlxVectorVectorArrayHandle VectorArrayNew();

    /// <summary>
    /// Sets the element at a specific index of the outer vector to a given vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_set")]
    public static partial int VectorArraySet(
        ref MlxVectorVectorArrayHandle vec,
        MlxVectorVectorArrayHandle src
    );

    /// <summary>
    /// Frees a dynamic vector that holds vector-of-array objects.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_free")]
    public static partial int VectorArrayFree(
        MlxVectorVectorArrayHandle vec
    );

    /// <summary>
    /// Initializes a new vector-of-vector-of-arrays with a copy of an existing array of vector-of-array objects.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_new_data")]
    public static partial MlxVectorVectorArrayHandle VectorArrayNewData(
        MlxVectorArrayHandle* data,
        nuint size
    );

    /// <summary>
    /// Creates a new vector-of-vector-of-arrays containing a single given vector-of-arrays as the initial element.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_new_value")]
    public static partial MlxVectorVectorArrayHandle VectorArrayNewValue(
        MlxVectorArrayHandle val
    );

    /// <summary>
    /// Copies an array of vector-of-array objects into the outer vector starting at a given index.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_set_data")]
    public static partial int VectorArraySetData(
        ref MlxVectorVectorArrayHandle vec,
        MlxVectorArrayHandle* data,
        nuint size
    );

    /// <summary>
    /// Replaces the element at a given index of the outer vector with a new vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_set_value")]
    public static partial int VectorArraySetValue(
        ref MlxVectorVectorArrayHandle vec,
        MlxVectorArrayHandle val
    );

    /// <summary>
    /// Appends multiple vector-of-array objects from a C array into a vector-of-vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_append_data")]
    public static partial int VectorArrayAppendData(
        MlxVectorVectorArrayHandle vec,
        MlxVectorArrayHandle* data,
        nuint size
    );

    /// <summary>
    /// Appends a single vector-of-arrays into a vector-of-vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_append_value")]
    public static partial int VectorArrayAppendValue(
        MlxVectorVectorArrayHandle vec,
        MlxVectorArrayHandle val
    );

    /// <summary>
    /// Returns the number of elements in the vector-of-vector-of-arrays.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_size")]
    public static partial nuint VectorArraySize(
        MlxVectorVectorArrayHandle vec
    );

    /// <summary>
    /// Retrieves the vector-of-arrays at a given index in the outer vector.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_vector_array_get")]
    public static partial int VectorArrayGet(
        out MlxVectorArrayHandle res,
        MlxVectorVectorArrayHandle vec,
        nuint idx
    );

    /// <summary>
    /// Creates a new dynamic vector for int values.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_new")]
    public static partial MlxVectorIntHandle IntNew();

    /// <summary>
    /// Sets the integer at a specific index in the vector to a new value.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_set")]
    public static partial int IntSet(
        ref MlxVectorIntHandle vec,
        MlxVectorIntHandle src
    );

    /// <summary>
    /// Frees a dynamic vector of integers.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_free")]
    public static partial int IntFree(
        MlxVectorIntHandle vec
    );

    /// <summary>
    /// Creates a new int vector initialized with a copy of an existing array of integers.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_new_data")]
    public static partial MlxVectorIntHandle IntNewData(
        int* data,
        nuint size
    );

    /// <summary>
    /// Creates a new int vector with a single integer as the initial content.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_new_value")]
    public static partial MlxVectorIntHandle IntNewValue(
        int val
    );

    /// <summary>
    /// Copies an array of integers into the vector starting at a given index.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_set_data")]
    public static partial int IntSetData(
        ref MlxVectorIntHandle vec,
        int* data,
        nuint size
    );

    /// <summary>
    /// Sets the element at a given index to an integer value.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_set_value")]
    public static partial int IntSetValue(
        ref MlxVectorIntHandle vec,
        int val
    );

    /// <summary>
    /// Appends multiple integers from a C array to the end of the vector.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_append_data")]
    public static partial int IntAppendData(
        MlxVectorIntHandle vec,
        int* data,
        nuint size
    );

    /// <summary>
    /// Appends a single integer value to the end of the vector.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_append_value")]
    public static partial int IntAppendValue(
        MlxVectorIntHandle vec,
        int val
    );

    /// <summary>
    /// Returns the number of elements in the int vector.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_size")]
    public static partial nuint IntSize(
        MlxVectorIntHandle vec
    );

    /// <summary>
    /// Retrieves the integer at a given index in the vector.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_int_get")]
    public static partial int IntGet(
        out int res,
        MlxVectorIntHandle vec,
        nuint idx
    );

    /// <summary>
    /// Creates a new dynamic vector for string values.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_new")]
    public static partial MlxVectorStringHandle StringNew();

    /// <summary>
    /// Sets the string at a specific index in the vector to a new string value.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_set")]
    public static partial int StringSet(
        ref MlxVectorStringHandle vec,
        MlxVectorStringHandle src
    );

    /// <summary>
    /// Frees a dynamic vector of strings.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_free")]
    public static partial int StringFree(
        MlxVectorStringHandle vec
    );

    /// <summary>
    /// Creates a new string vector initialized with a copy of an existing array of strings.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_new_data")]
    public static partial MlxVectorStringHandle StringNewData(
        nint* data,
        nuint size
    );

    /// <summary>
    /// Creates a new string vector containing a single initial string element.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_new_value", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxVectorStringHandle StringNewValue(
        string val
    );

    /// <summary>
    /// Copies an array of strings into the vector starting at a specified index.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_set_data")]
    public static partial int StringSetData(
        ref MlxVectorStringHandle vec,
        nint* data,
        nuint size
    );

    /// <summary>
    /// Replaces the string at a given index with a new string value.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_set_value", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StringSetValue(
        ref MlxVectorStringHandle vec,
        string val
    );

    /// <summary>
    /// Appends multiple strings from a C array to the end of the vector-of-strings.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_append_data")]
    public static partial int StringAppendData(
        MlxVectorStringHandle vec,
        nint* data,
        nuint size
    );

    /// <summary>
    /// Appends a single string to the end of the vector-of-strings.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_append_value", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StringAppendValue(
        MlxVectorStringHandle vec,
        string val
    );

    /// <summary>
    /// Returns the number of string elements in the vector.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_size")]
    public static partial nuint StringSize(
        MlxVectorStringHandle vec
    );

    /// <summary>
    /// Retrieves the string at a given index in the vector-of-strings.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_vector_string_get")]
    public static partial int StringGet(
        out nint res,
        MlxVectorStringHandle vec,
        nuint idx
    );
}