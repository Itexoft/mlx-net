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
public sealed class UpsampleTests
{
    [Test]
    public void UpsampleNearest_MatchesExpectedTiling()
    {
        TestHelpers.RequireNativeOrIgnore();

        var input = CreateArray([1f, 2f, 3f, 4f], [1, 2, 2, 1]);
        try
        {
            using var upsample = new Upsample(2f, UpsampleMode.Nearest);
            var result = upsample.Forward(input);
            try
            {
                var squeezed = result.Squeezed(0, 3);
                try
                {
                    TestHelpers.Ok(MlxArray.Eval(squeezed), "eval squeezed");
                    var values = TestHelpers.ToFloat32(squeezed);
                    var expected = new[]
                    {
                        1f, 1f, 2f, 2f,
                        1f, 1f, 2f, 2f,
                        3f, 3f, 4f, 4f,
                        3f, 3f, 4f, 4f
                    };
                    Assert.That(values, Is.EqualTo(expected).Within(1e-5));
                }
                finally
                {
                    MlxArray.Free(squeezed);
                }
            }
            finally
            {
                MlxArray.Free(result);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void UpsampleLinear_ProducesInterpolatedSurface()
    {
        TestHelpers.RequireNativeOrIgnore();

        var input = CreateArray([1f, 2f, 3f, 4f], [1, 2, 2, 1]);
        try
        {
            using var upsample = new Upsample(2f, UpsampleMode.Linear);
            var result = upsample.Forward(input);
            try
            {
                var squeezed = result.Squeezed(0, 3);
                try
                {
                    TestHelpers.Ok(MlxArray.Eval(squeezed), "eval squeezed");
                    var values = TestHelpers.ToFloat32(squeezed);
                    var expected = new[]
                    {
                        1.0f, 1.25f, 1.75f, 2.0f,
                        1.5f, 1.75f, 2.25f, 2.5f,
                        2.5f, 2.75f, 3.25f, 3.5f,
                        3.0f, 3.25f, 3.75f, 4.0f
                    };
                    Assert.That(values, Is.EqualTo(expected).Within(1e-5));
                }
                finally
                {
                    MlxArray.Free(squeezed);
                }
            }
            finally
            {
                MlxArray.Free(result);
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