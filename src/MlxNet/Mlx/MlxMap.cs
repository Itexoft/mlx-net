// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxMap
{
    /// <summary>
    /// Creates a new empty map for string keys to array values.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_new")]
    public static partial MlxMapStringToArrayHandle StringToArrayNew();

    /// <summary>
    /// Sets or updates the array value for a given key in the map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_set")]
    public static partial int StringToArraySet(
        ref MlxMapStringToArrayHandle map,
        MlxMapStringToArrayHandle src
    );

    /// <summary>
    /// Frees a map that stores string keys to array values.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_free")]
    public static partial int StringToArrayFree(
        MlxMapStringToArrayHandle map
    );

    /// <summary>
    /// Inserts a new key-array pair into the map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_insert", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StringToArrayInsert(
        MlxMapStringToArrayHandle map,
        string key,
        MlxArrayHandle value
    );

    /// <summary>
    /// Retrieves the array value associated with a given key in the map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_get", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StringToArrayGet(
        out MlxArrayHandle value,
        MlxMapStringToArrayHandle map,
        string key
    );

    /// <summary>
    /// Creates a new iterator for a string-to-array map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_iterator_new")]
    public static partial MlxMapStringToArrayIteratorHandle StringToArrayIteratorNew(
        MlxMapStringToArrayHandle map
    );

    /// <summary>
    /// Frees an iterator over a string-to-array map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_iterator_free")]
    public static partial int StringToArrayIteratorFree(
        MlxMapStringToArrayIteratorHandle it
    );

    /// <summary>
    /// Advances the iterator and returns the next key and array value pair.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_array_iterator_next")]
    public static partial int StringToArrayIteratorNext(
        out nint key,
        out MlxArrayHandle value,
        MlxMapStringToArrayIteratorHandle it
    );

    /// <summary>
    /// Creates a new empty map for string keys to string values.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_new")]
    public static partial MlxMapStringToStringHandle StringToStringNew();

    /// <summary>
    /// Sets or updates the string value for a given key in the map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_set")]
    public static partial int StringToStringSet(
        ref MlxMapStringToStringHandle map,
        MlxMapStringToStringHandle src
    );

    /// <summary>
    /// Frees a map that stores string keys to string values.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_free")]
    public static partial int StringToStringFree(
        MlxMapStringToStringHandle map
    );

    /// <summary>
    /// Inserts a new key-string pair into the map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_insert", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StringToStringInsert(
        MlxMapStringToStringHandle map,
        string key,
        string value
    );

    /// <summary>
    /// Retrieves the string value associated with a given key in the map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_get", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int StringToStringGet(
        out nint value,
        MlxMapStringToStringHandle map,
        string key
    );

    /// <summary>
    /// Creates a new iterator for a string-to-string map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_iterator_new")]
    public static partial MlxMapStringToStringIteratorHandle StringToStringIteratorNew(
        MlxMapStringToStringHandle map
    );

    /// <summary>
    /// Frees an iterator over a string-to-string map.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_iterator_free")]
    public static partial int StringToStringIteratorFree(
        MlxMapStringToStringIteratorHandle it
    );

    /// <summary>
    /// Returns the next key and string value pair from the iterator.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_map_string_to_string_iterator_next")]
    public static partial int StringToStringIteratorNext(
        out nint key,
        out nint value,
        MlxMapStringToStringIteratorHandle it
    );
}