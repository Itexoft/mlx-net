// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxDistributedGroup
{
    /// <summary>Returns the rank (ID) of the current process within a given distributed group.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_group_rank")]
    public static partial int DistributedGroupRank(
        MlxDistributedGroupHandle group
    );

    /// <summary>Returns the total number of processes in the given distributed group.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_group_size")]
    public static partial int DistributedGroupSize(
        MlxDistributedGroupHandle group
    );

    /// <summary>Splits a distributed world communicator into subgroups (e.g. by color), returning a new group handle.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_group_split")]
    public static partial MlxDistributedGroupHandle DistributedGroupSplit(
        MlxDistributedGroupHandle group,
        int color,
        int key
    );

    /// <summary>Returns True if distributed communication features are available (e.g. MPI or similar is initialized).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_is_available")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool DistributedIsAvailable();

    /// <summary>Initializes the distributed communication environment (sets up backend, world size, etc.).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_init")]
    public static partial MlxDistributedGroupHandle DistributedInit(
        [MarshalAs(UnmanagedType.I1)] bool strict
    );
}