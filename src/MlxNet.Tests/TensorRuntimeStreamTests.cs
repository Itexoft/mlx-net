// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Threading;
using Itexoft.Mlx;
using Itexoft.Tensors.Internal;
using NUnit.Framework;

[TestFixture]
public sealed class TensorRuntimeStreamTests
{
    [Test]
    public void DefaultStream_IsStableAcrossThreads_AndMatchesNativeDefaultStream()
    {
        TestHelpers.RequireNativeOrIgnore();

        var currentThreadStream = GetDefaultTensorRuntimeStream();
        var sameThreadStream = GetDefaultTensorRuntimeStream();

        Assert.That(MlxStream.Equal(currentThreadStream, sameThreadStream), Is.True);

        MlxStreamHandle otherThreadStream = default;
        Exception? workerException = null;

        var worker = new Thread(() =>
        {
            try
            {
                otherThreadStream = GetDefaultTensorRuntimeStream();
            }
            catch (Exception ex)
            {
                workerException = ex;
            }
        });

        worker.Start();
        worker.Join();

        if (workerException is not null)
            ExceptionDispatchInfo.Capture(workerException).Throw();

        Assert.That(otherThreadStream.ctx, Is.Not.EqualTo(0));
        Assert.That(MlxStream.Equal(currentThreadStream, otherThreadStream), Is.True);

        Assert.That(MlxDevice.GetDefaultDevice(out var device), Is.Zero);
        try
        {
            Assert.That(MlxStream.GetDefaultStream(out var nativeDefaultStream, device), Is.Zero);
            try
            {
                Assert.That(MlxStream.Equal(currentThreadStream, nativeDefaultStream), Is.True);
            }
            finally
            {
                if (nativeDefaultStream.ctx != 0)
                    Assert.That(MlxStream.Free(nativeDefaultStream), Is.Zero);
            }
        }
        finally
        {
            if (device.ctx != 0)
                Assert.That(MlxDevice.Free(device), Is.Zero);
        }
    }

    private static MlxStreamHandle GetDefaultTensorRuntimeStream()
    {
        var method = typeof(TensorRuntime).GetMethod("DefaultStream", BindingFlags.NonPublic | BindingFlags.Static);
        if (method is null)
            throw new InvalidOperationException("TensorRuntime.DefaultStream() was not found.");

        return (MlxStreamHandle)(method.Invoke(null, null) ?? throw new InvalidOperationException("Default stream call returned null."));
    }
}
