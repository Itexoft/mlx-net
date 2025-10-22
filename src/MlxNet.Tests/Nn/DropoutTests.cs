// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace Itexoft.Mlx.Nn.Tests;

[TestFixture]
public unsafe class DropoutTests
{
    [Test]
    public void Dropout_WithZeroProbability_IsIdentityDuringTraining()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var dropout = new Dropout(p: 0f);
        dropout.Train(true);

        var input = CreateArray([1f, -2f, 3f, -4f], [1, 4]);
        try
        {
            var result = dropout.Forward(input);
            Assert.That(result.ctx, Is.EqualTo(input.ctx), "Dropout should return the original handle when keep probability is 1.");

            TestHelpers.Ok(MlxArray.Eval(result), "eval dropout result");
            var values = TestHelpers.ToFloat32(result);
            Assert.That(values, Is.EqualTo(new[] { 1f, -2f, 3f, -4f }).Within(1e-6));
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void Dropout2d_WithZeroProbability_IsIdentity()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var dropout = new Dropout2d(p: 0f);
        dropout.Train(true);

        var input = CreateArray([1f, 2f, 3f, 4f], [1, 2, 2, 1]);
        try
        {
            var result = dropout.Forward(input);
            Assert.That(result.ctx, Is.EqualTo(input.ctx));

            TestHelpers.Ok(MlxArray.Eval(result), "eval dropout2d result");
            var values = TestHelpers.ToFloat32(result);
            Assert.That(values, Is.EqualTo(new[] { 1f, 2f, 3f, 4f }).Within(1e-6));
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void Dropout3d_WithZeroProbability_IsIdentity()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var dropout = new Dropout3d(p: 0f);
        dropout.Train(true);

        var input = CreateArray([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], [1, 2, 2, 2, 1]);
        try
        {
            var result = dropout.Forward(input);
            Assert.That(result.ctx, Is.EqualTo(input.ctx));

            TestHelpers.Ok(MlxArray.Eval(result), "eval dropout3d result");
            var values = TestHelpers.ToFloat32(result);
            Assert.That(values, Is.EqualTo(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }).Within(1e-6));
        }
        finally
        {
            MlxArray.Free(input);
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
}