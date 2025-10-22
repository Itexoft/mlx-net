// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Linq;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace Itexoft.Mlx.Nn.Tests;

[TestFixture]
public sealed class PoolingTests
{
    [Test]
    public void MaxPool1d_Stride1()
    {
        TestHelpers.RequireNativeOrIgnore();

        var input = CreateArray([0f, 1f, 2f, 3f], [1, 4, 1]);
        try
        {
            using var pool = new MaxPool1d(2, 1);
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 1f, 2f, 3f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void MaxPool1d_Stride2()
    {
        TestHelpers.RequireNativeOrIgnore();

        var inputValues = Enumerable.Range(0, 8).Select(v => (float)v).ToArray();
        var input = CreateArray(inputValues, [2, 4, 1]);
        try
        {
            using var pool = new MaxPool1d(2, 2);
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 1f, 3f, 5f, 7f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void MaxPool2d_Stride1()
    {
        TestHelpers.RequireNativeOrIgnore();

        var inputValues = Enumerable.Range(0, 16).Select(v => (float)v).ToArray();
        var input = CreateArray(inputValues, [1, 4, 4, 1]);
        try
        {
            using var pool = new MaxPool2d((2, 2), (1, 1));
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 5f, 6f, 7f, 9f, 10f, 11f, 13f, 14f, 15f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void MaxPool2d_Stride2()
    {
        TestHelpers.RequireNativeOrIgnore();

        var inputValues = Enumerable.Range(0, 32).Select(v => (float)v).ToArray();
        var input = CreateArray(inputValues, [2, 4, 4, 1]);
        try
        {
            using var pool = new MaxPool2d((2, 2), (2, 2));
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 5f, 7f, 13f, 15f, 21f, 23f, 29f, 31f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void AvgPool1d_Stride1()
    {
        TestHelpers.RequireNativeOrIgnore();

        var input = CreateArray([0f, 1f, 2f, 3f], [1, 4, 1]);
        try
        {
            using var pool = new AvgPool1d(2, 1);
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 0.5f, 1.5f, 2.5f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void AvgPool1d_Stride2()
    {
        TestHelpers.RequireNativeOrIgnore();

        var inputValues = Enumerable.Range(0, 8).Select(v => (float)v).ToArray();
        var input = CreateArray(inputValues, [2, 4, 1]);
        try
        {
            using var pool = new AvgPool1d(2, 2);
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 0.5f, 2.5f, 4.5f, 6.5f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void AvgPool2d_Stride1()
    {
        TestHelpers.RequireNativeOrIgnore();

        var inputValues = Enumerable.Range(0, 16).Select(v => (float)v).ToArray();
        var input = CreateArray(inputValues, [1, 4, 4, 1]);
        try
        {
            using var pool = new AvgPool2d((2, 2), (1, 1));
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                var expected = new[] { 2.5f, 3.5f, 4.5f, 6.5f, 7.5f, 8.5f, 10.5f, 11.5f, 12.5f };
                Assert.That(values, Is.EqualTo(expected).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void AvgPool2d_Stride2()
    {
        TestHelpers.RequireNativeOrIgnore();

        var inputValues = Enumerable.Range(0, 16).Select(v => (float)v).ToArray();
        var input = CreateArray(inputValues, [1, 4, 4, 1]);
        try
        {
            using var pool = new AvgPool2d((2, 2), (2, 2));
            var output = pool.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Is.EqualTo(new[] { 2.5f, 4.5f, 10.5f, 12.5f }).Within(1e-5));
            }
            finally
            {
                MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    private static unsafe MlxArrayHandle CreateArray(float[] values, int[] shape)
    {
        fixed (float* data = values)
        fixed (int* dims = shape)
        {
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MLX_FLOAT32);
        }
    }
}