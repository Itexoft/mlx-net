// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.CompilerServices;

namespace Itexoft.Mlx.Nn;

internal static class TensorUtilities
{
    /// <summary>
    /// Returns <c>true</c> when the supplied handle does not reference a live MLX array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsNull(MlxArrayHandle handle) => handle.ctx == 0;

    /// <summary>
    /// Throws an <see cref="InvalidOperationException"/> when <paramref name="status"/> is non-zero.
    /// </summary>
    internal static void CheckStatus(int status, string operation)
    {
        if (status != 0)
            throw new InvalidOperationException($"MLX operation '{operation}' failed with status code {status}.");
    }

    /// <summary>
    /// Returns the default stream used for eager operations.
    /// </summary>
    internal static MlxStreamHandle DefaultStream()
        => MlxStream.DefaultCpuStreamNew();
}