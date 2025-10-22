// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxString
{
    /// <summary>
    /// Creates a new, empty MLX string object.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_string_new")]
    public static partial MlxStringHandle New();

    /// <summary>
    /// Creates a new MLX string initialized with the given data.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_string_new_data", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxStringHandle NewData(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string str
    );

    /// <summary>
    /// Sets the contents of an MLX string to the given source string.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_string_set")]
    public static partial int Set(
        ref MlxStringHandle str,
        MlxStringHandle src
    );

    /// <summary>
    /// Returns a pointer to the internal character data of an MLX string.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_string_data")]
    public static partial nint Data(
        MlxStringHandle str
    );

    /// <summary>
    /// Frees an MLX string object and its allocated memory.
    /// </summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_string_free")]
    public static partial int Free(
        MlxStringHandle str
    );
}