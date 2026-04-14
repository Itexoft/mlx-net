// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Itexoft.Mlx.Nn;

internal static class TensorUtilities
{
    private static readonly Lock defaultStreamSync = new();
    private static MlxStreamHandle sCpuDefaultStream;
    private static MlxStreamHandle sGpuDefaultStream;

    /// <summary>
    /// Returns <c>true</c> when the supplied handle does not reference a live MLX array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsNull(MlxArrayHandle handle) => handle.ctx == 0;

    /// <summary>
    /// Throws an <see cref="InvalidOperationException" /> when <paramref name="status" /> is non-zero.
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
    {
        CheckStatus(MlxDevice.GetDefaultDevice(out var device), "get_default_device");

        try
        {
            CheckStatus(MlxDevice.GetType(out var type, device), "get_default_device_type");

            return type switch
            {
                MlxDeviceType.MlxCpu => GetCachedDefaultStream(device, ref sCpuDefaultStream),
                MlxDeviceType.MlxGpu => GetCachedDefaultStream(device, ref sGpuDefaultStream),
                _ => throw new InvalidOperationException($"Unsupported MLX device type '{type}'."),
            };
        }
        finally
        {
            if (device.ctx != 0)
                CheckStatus(MlxDevice.Free(device), "free_default_device");
        }
    }

    private static MlxStreamHandle GetCachedDefaultStream(MlxDeviceHandle device, ref MlxStreamHandle cachedStream)
    {
        lock (defaultStreamSync)
        {
            CheckStatus(MlxStream.GetDefaultStream(out var currentStream, device), "get_default_stream");

            try
            {
                if (cachedStream.ctx == 0)
                {
                    cachedStream = currentStream;
                    currentStream = default;
                }
                else
                    CheckStatus(MlxStream.Set(ref cachedStream, currentStream), "set_cached_default_stream");

                return cachedStream;
            }
            finally
            {
                if (currentStream.ctx != 0)
                    CheckStatus(MlxStream.Free(currentStream), "free_current_default_stream");
            }
        }
    }
}
