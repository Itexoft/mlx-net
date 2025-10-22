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
public sealed class LossTests
{
    [Test]
    public void CrossEntropy_ClassIndicesProducesZeroLoss()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logits = CreateFloatArray(
            [0f, float.NegativeInfinity, float.NegativeInfinity, 0f],
            [2, 2]);

        var targets = CreateIntArray([0, 1], [2]);

        try
        {
            var loss = Losses.CrossEntropy(logits, targets, reduction: LossReduction.None);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(loss), "eval loss");
                var values = TestHelpers.ToFloat32(loss);
                Assert.That(values, Is.EqualTo(new[] { 0f, 0f }).Within(1e-6));
            }
            finally
            {
                MlxArray.Free(loss);
            }
        }
        finally
        {
            MlxArray.Free(logits);
            MlxArray.Free(targets);
        }
    }

    [Test]
    public void CrossEntropy_ProbabilityTargetsReturnNan()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logits = CreateFloatArray(
            [0f, float.NegativeInfinity, float.NegativeInfinity, 0f],
            [2, 2]);

        var probabilities = CreateFloatArray(
            [1f, 0f, 0f, 1f],
            [2, 2]);

        try
        {
            var loss = Losses.CrossEntropy(logits, probabilities, reduction: LossReduction.None);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(loss), "eval loss");
                var values = TestHelpers.ToFloat32(loss);
                Assert.That(values.All(float.IsNaN), Is.True, "Expected NaN loss for probability targets.");
            }
            finally
            {
                MlxArray.Free(loss);
            }
        }
        finally
        {
            MlxArray.Free(logits);
            MlxArray.Free(probabilities);
        }
    }

    private static unsafe MlxArrayHandle CreateFloatArray(float[] values, int[] shape)
    {
        fixed (float* data = values)
        fixed (int* dims = shape)
        {
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MLX_FLOAT32);
        }
    }

    private static unsafe MlxArrayHandle CreateIntArray(int[] values, int[] shape)
    {
        fixed (int* data = values)
        fixed (int* dims = shape)
        {
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MLX_INT32);
        }
    }
}