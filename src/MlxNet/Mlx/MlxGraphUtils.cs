// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static partial class MlxGraphUtils
{
    /// <summary>Creates a node namer used for graph export and diagnostics.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_node_namer_new")]
    public static partial MlxNodeNamer NodeNamerNew();

    /// <summary>Releases a node namer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_node_namer_free")]
    public static partial int NodeNamerFree(
        MlxNodeNamer namer
    );

    /// <summary>Associates a name with the specified array node.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_node_namer_set_name", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int NodeNamerSetName(
        MlxNodeNamer namer,
        MlxArrayHandle arr,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string name
    );

    /// <summary>Returns the UTF-8 node name pointer for the specified array node.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_node_namer_get_name")]
    public static partial int NodeNamerGetName(
        out nint name,
        MlxNodeNamer namer,
        MlxArrayHandle arr
    );

    /// <summary>Exports the graph to DOT format using the provided FILE pointer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_export_to_dot")]
    public static partial int ExportToDot(
        nint os,
        MlxNodeNamer namer,
        MlxVectorArrayHandle outputs
    );

    /// <summary>Prints a textual graph representation using the provided FILE pointer.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_print_graph")]
    public static partial int PrintGraph(
        nint os,
        MlxNodeNamer namer,
        MlxVectorArrayHandle outputs
    );
}
