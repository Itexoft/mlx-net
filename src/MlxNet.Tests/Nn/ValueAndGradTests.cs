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
public sealed class ValueAndGradTests
{
    [Test]
    public void Build_ReturnsCorrectLossAndGradient()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var linear = new Linear(1, 1, true);

        // deterministically set parameters
        linear.Weight.SetValue(CreateFloatArray([0.5f], [1, 1]));
        linear.Bias!.SetValue(CreateFloatArray([0.1f], [1]));

        var closure = ValueAndGrad.Build(
            linear,
            static (module, input, target) =>
            {
                var layer = (Linear)module;
                var prediction = layer.Forward(input);
                var diff = prediction.Subtract(target);
                MlxArray.Free(prediction);
                var squared = diff.Multiply(diff);
                MlxArray.Free(diff);
                var mean = squared.Mean([0, 1], false);
                MlxArray.Free(squared);

                return mean;
            });

        var input = CreateFloatArray([2f], [1, 1]);
        var target = CreateFloatArray([1f], [1, 1]);

        try
        {
            var result = closure(linear, input, target);
            var lossHandle = result.Item1;
            var gradients = result.Item2;
            try
            {
                TestHelpers.Ok(MlxArray.Eval(lossHandle), "eval loss");
                var lossValue = TestHelpers.ToFloat32(lossHandle);
                Assert.That(lossValue[0], Is.EqualTo(0.01f).Within(1e-6));

                Assert.That(gradients.Count, Is.EqualTo(2));
                Assert.That(gradients.Keys, Does.Contain("weight"));
                Assert.That(gradients.Keys, Does.Contain("bias"));

                var weightGrad = gradients["weight"].Value;
                TestHelpers.Ok(MlxArray.Eval(weightGrad), "eval weight gradient");
                var weightValue = TestHelpers.ToFloat32(weightGrad);
                Assert.That(weightValue[0], Is.EqualTo(0.4f).Within(1e-5));

                var biasGrad = gradients["bias"].Value;
                TestHelpers.Ok(MlxArray.Eval(biasGrad), "eval bias gradient");
                var biasValue = TestHelpers.ToFloat32(biasGrad);
                Assert.That(biasValue[0], Is.EqualTo(0.2f).Within(1e-5));
            }
            finally
            {
                foreach (var kv in gradients)
                {
                    var entry = kv.Value;
                    if (entry.Value.ctx != 0)
                        MlxArray.Free(entry.Value);
                }

                if (lossHandle.ctx != 0)
                    MlxArray.Free(lossHandle);
            }
        }
        finally
        {
            MlxArray.Free(input);
            MlxArray.Free(target);
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
}