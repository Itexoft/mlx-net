// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxDistributed
{
    /// <summary>Gathers arrays from all processes in a distributed group and concatenates or collects them on each process.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_all_gather")]
    public static partial int AllGather(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );

    /// <summary>Performs an all-reduce operation computing the element-wise maximum across all processes in the group.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_all_max")]
    public static partial int AllMax(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );

    /// <summary>Performs an all-reduce operation computing the element-wise minimum across all processes in the group.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_all_min")]
    public static partial int AllMin(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );

    /// <summary>Performs an all-reduce summation across all processes (sums up arrays from all ranks, result available to all).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_all_sum")]
    public static partial int AllSum(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );

    /// <summary>Receives an array from another process (blocking until the data from the sending rank is received).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_recv")]
    public static partial int Recv(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        int src,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );

    /// <summary>Receives an array from another process, using a reference array to determine the shape and dtype of the incoming data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_recv_like")]
    public static partial int RecvLike(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        int src,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );

    /// <summary>Sends an array to another process (non-blocking send of array data to a specified rank).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_distributed_send")]
    public static partial int Send(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        int dst,
        MlxDistributedGroupHandle group,
        MlxStreamHandle s
    );
}