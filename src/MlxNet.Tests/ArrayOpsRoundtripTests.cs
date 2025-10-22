// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace MlxNet.Tests;

[TestFixture]
public unsafe class ArrayOpsRoundtripTests
{
    [Test]
    public void Reshape_PreservesOrdering()
    {
        TestHelpers.RequireNativeOrIgnore();

        var data = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var originalShape = new[] { 2, 3 };
        var reshapedShape = new[] { 3, 2 };

        var original = CreateArray(data, originalShape);
        try
        {
            var reshaped = original.Reshape(reshapedShape);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(reshaped), "eval reshape");
                var values = TestHelpers.ToFloat32(reshaped);
                Assert.That(values, Is.EqualTo(data).Within(1e-6));
                var shape = TestHelpers.ShapeOf(reshaped);
                Assert.That(shape, Is.EqualTo(reshapedShape));
            }
            finally
            {
                if (reshaped.ctx != 0)
                    MlxArray.Free(reshaped);
            }
        }
        finally
        {
            MlxArray.Free(original);
        }
    }

    [Test]
    public void Transpose_SwapsAxes()
    {
        TestHelpers.RequireNativeOrIgnore();

        var data = new[] { 1f, 2f, 3f, 4f }; // 2x2 matrix
        var shape = new[] { 2, 2 };
        var array = CreateArray(data, shape);
        var status = MlxOps.Transpose(out var transposed, array, TensorUtilities.DefaultStream());
        TestHelpers.Ok(status, "transpose");
        var expected = CreateArray([1f, 3f, 2f, 4f], [2, 2]);
        try
        {
            TestHelpers.Ok(MlxArray.Eval(transposed), "eval transpose");
            var diff = transposed.Subtract(expected);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(diff), "eval diff");
                var values = TestHelpers.ToFloat32(diff);
                Assert.That(values, Has.All.EqualTo(0f).Within(1e-6));
            }
            finally
            {
                if (diff.ctx != 0)
                    MlxArray.Free(diff);
            }
        }
        finally
        {
            if (transposed.ctx != 0)
                MlxArray.Free(transposed);
            MlxArray.Free(expected);
            MlxArray.Free(array);
        }
    }

    [Test]
    public void TakeAlong_GathersExpectedElements()
    {
        TestHelpers.RequireNativeOrIgnore();

        var data = new[]
        {
            10f, 20f, 30f,
            40f, 50f, 60f
        };
        var shape = new[] { 2, 3 };
        var array = CreateArray(data, shape);
        var indicesValues = new[]
        {
            2, 1, 0,
            0, 2, 1
        };
        var indices = CreateIntArray(indicesValues, shape);
        try
        {
            var gathered = array.TakeAlong(indices, 1);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(gathered), "eval take_along");
                var values = TestHelpers.ToFloat32(gathered);
                Assert.That(values, Is.EqualTo(new[] { 30f, 20f, 10f, 40f, 60f, 50f }).Within(1e-6));
            }
            finally
            {
                if (gathered.ctx != 0)
                    MlxArray.Free(gathered);
            }
        }
        finally
        {
            MlxArray.Free(array);
            MlxArray.Free(indices);
        }
    }

    private static MlxArrayHandle CreateArray(float[] values, int[] shape)
    {
        fixed (float* data = values)
        fixed (int* dims = shape)
        {
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MLX_FLOAT32);
        }
    }

    private static MlxArrayHandle CreateIntArray(int[] values, int[] shape)
    {
        fixed (int* data = values)
        fixed (int* dims = shape)
        {
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MLX_INT32);
        }
    }
}