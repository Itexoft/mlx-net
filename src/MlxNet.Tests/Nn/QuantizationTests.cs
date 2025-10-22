// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Linq;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace Itexoft.Mlx.Nn.Tests;

[TestFixture]
public unsafe class QuantizationTests
{
    [Test]
    public void QuantizeSingle_ReturnsQuantizedLinear()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var linear = new Linear(128, 32, true);
        linear.Weight.SetValue(CreateArray(Enumerable.Repeat(0.25f, 32 * 128).ToArray(), [32, 128]));
        linear.Bias!.SetValue(CreateArray(Enumerable.Repeat(0.1f, 32).ToArray(), [32]));

        using var quantized = Quantization.QuantizeSingle(linear, 32, 4) as QuantizedLinear;

        Assert.That(quantized, Is.Not.Null);
        Assert.That(quantized!.GroupSize, Is.EqualTo(32));
        Assert.That(quantized.Bits, Is.EqualTo(4));
        Assert.That(quantized.Mode, Is.EqualTo(QuantizationMode.Affine));

        var input = CreateArray(Enumerable.Range(0, 128).Select(i => (float)(i - 64) / 32f).ToArray(), [1, 128]);
        try
        {
            var output = quantized.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval quantized output");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values.Length, Is.EqualTo(32));
            }
            finally
            {
                if (output.ctx != 0)
                    MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void Quantize_FilterAppliesToSelectedModules()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new DualLinearModule();

        Quantization.Quantize(
            module,
            32,
            4,
            QuantizationMode.Affine,
            (path, _) => string.Equals(path, "second", StringComparison.Ordinal));

        var flattened = module.FlattenModules();
        Assert.That(flattened["first"], Is.TypeOf<Linear>());
        Assert.That(flattened["second"], Is.TypeOf<QuantizedLinear>());
    }

    [Test]
    public void QuantizeWithSelector_AllowsDifferentParameters()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new DualLinearModule();

        Quantization.Quantize(
            module,
            (path, _) => path switch
            {
                "first" => (groupSize: 32, bits: 4, mode: QuantizationMode.Affine),
                _ => null
            });

        var flattened = module.FlattenModules();
        Assert.That(flattened["first"], Is.TypeOf<QuantizedLinear>());
        Assert.That(flattened["second"], Is.TypeOf<Linear>());

        var quantizedFirst = (QuantizedLinear)flattened["first"];
        Assert.That(quantizedFirst.GroupSize, Is.EqualTo(32));
    }

    private static MlxArrayHandle CreateArray(float[] values, int[] shape)
    {
        fixed (float* data = values)
        fixed (int* dims = shape)
        {
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MLX_FLOAT32);
        }
    }

    private sealed class DualLinearModule : Module
    {
        public DualLinearModule()
        {
            this.First = this.RegisterModule("first", new Linear(128, 32));
            this.Second = this.RegisterModule("second", new Linear(128, 32));

            this.First.Weight.SetValue(CreateArray(Enumerable.Repeat(0.25f, 32 * 128).ToArray(), [32, 128]));
            this.First.Bias!.SetValue(CreateArray(Enumerable.Repeat(0.05f, 32).ToArray(), [32]));

            this.Second.Weight.SetValue(CreateArray(Enumerable.Repeat(0.5f, 32 * 128).ToArray(), [32, 128]));
            this.Second.Bias!.SetValue(CreateArray(Enumerable.Repeat(0.1f, 32).ToArray(), [32]));
        }

        public Linear First { get; }

        public Linear Second { get; }
    }
}