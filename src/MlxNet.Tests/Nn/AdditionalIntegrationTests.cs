// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace Itexoft.Mlx.Nn.Tests;

[TestFixture]
public unsafe class AdditionalIntegrationTests
{
    [Test]
    public void Sequential_ComposesIdentityAndRelu()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var sequential = new Sequential(new Identity(), new ReLU());
        var input = CreateFloatArray([1f, -2f, 3f, -4f], [1, 4]);
        try
        {
            var result = sequential.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(result), "eval sequential");
                var values = TestHelpers.ToFloat32(result);
                Assert.That(values, Is.EqualTo(new[] { 1f, 0f, 3f, 0f }).Within(1e-6));
            }
            finally
            {
                if (result.ctx != 0)
                    MlxArray.Free(result);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    [Test]
    public void Embedding_ForwardsIndicesAndProjects()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var embedding = new Embedding(3, 2);
        embedding.Weight.SetValue(CreateFloatArray([1f, 0f, 0f, 1f, 1f, 1f], [3, 2]));

        var indices = CreateIntArray([2, 0, 1], [3]);
        try
        {
            var output = embedding.Forward(indices);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval embedding forward");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(
                    values,
                    Is.EqualTo(new[] { 1f, 1f, 1f, 0f, 0f, 1f }).Within(1e-6));
            }
            finally
            {
                if (output.ctx != 0)
                    MlxArray.Free(output);
            }
        }
        finally
        {
            MlxArray.Free(indices);
        }

        var linearInput = CreateFloatArray([1f, 2f], [1, 2]);
        try
        {
            var projected = embedding.AsLinear(linearInput);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(projected), "eval embedding linear");
                var values = TestHelpers.ToFloat32(projected);
                Assert.That(values, Is.EqualTo(new[] { 1f, 2f, 3f }).Within(1e-6));
            }
            finally
            {
                if (projected.ctx != 0)
                    MlxArray.Free(projected);
            }
        }
        finally
        {
            MlxArray.Free(linearInput);
        }
    }

    [Test]
    public void RmsNorm_NormalizesVectorMagnitude()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var norm = new RmsNorm(2, 0f);
        var input = CreateFloatArray([3f, 4f], [1, 2]);
        try
        {
            var output = norm.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval rmsnorm");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values[0], Is.EqualTo(0.8485281f).Within(1e-5));
                Assert.That(values[1], Is.EqualTo(1.1313708f).Within(1e-5));
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
    public void Conv3d_ComputesSlidingWindowSum()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var conv = new Conv3d(1, 1, (2, 2, 2), bias: false);
        conv.KernelParameter.SetValue(CreateFloatArray([1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f], [1, 2, 2, 2, 1]));

        var depth = 3;
        var height = 3;
        var width = 3;
        var inputValues = new float[depth * height * width];
        for (var idx = 0; idx < inputValues.Length; idx++)
            inputValues[idx] = idx + 1;

        var input = CreateFloatArray(inputValues, [1, depth, height, width, 1]);
        try
        {
            var output = conv.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval conv3d");
                var values = TestHelpers.ToFloat32(output);
                var expected = new float[8];
                var o = 0;
                for (var z = 0; z < depth - 1; z++)
                for (var y = 0; y < height - 1; y++)
                for (var x = 0; x < width - 1; x++)
                {
                    var sum = 0f;
                    for (var kz = 0; kz < 2; kz++)
                    for (var ky = 0; ky < 2; ky++)
                    for (var kx = 0; kx < 2; kx++)
                    {
                        var dz = z + kz;
                        var dy = y + ky;
                        var dx = x + kx;
                        sum += inputValues[(dz * height + dy) * width + dx];
                    }

                    expected[o++] = sum;
                }

                Assert.That(values, Is.EqualTo(expected).Within(1e-5));
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
    public void ConvTranspose3d_ReconstructsExpectedVolume()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var conv = new ConvTranspose3d(
            1,
            1,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            outputPadding: (0, 0, 0),
            bias: false);
        conv.KernelParameter.SetValue(CreateFloatArray([1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f], [1, 2, 2, 2, 1]));

        var depth = 2;
        var height = 2;
        var width = 2;
        var inputValues = new float[depth * height * width];
        for (var idx = 0; idx < inputValues.Length; idx++)
            inputValues[idx] = idx + 1;

        var input = CreateFloatArray(inputValues, [1, depth, height, width, 1]);
        try
        {
            var output = conv.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval convtranspose3d");
                var values = TestHelpers.ToFloat32(output);
                var outDepth = 3;
                var outHeight = 3;
                var outWidth = 3;
                var expected = new float[outDepth * outHeight * outWidth];

                for (var z = 0; z < depth; z++)
                for (var y = 0; y < height; y++)
                for (var x = 0; x < width; x++)
                {
                    var contribution = inputValues[(z * height + y) * width + x];
                    for (var kz = 0; kz < 2; kz++)
                    for (var ky = 0; ky < 2; ky++)
                    for (var kx = 0; kx < 2; kx++)
                    {
                        var oz = z + kz;
                        var oy = y + ky;
                        var ox = x + kx;
                        expected[(oz * outHeight + oy) * outWidth + ox] += contribution;
                    }
                }

                Assert.That(values, Is.EqualTo(expected).Within(1e-5));
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
    public void Rnn_WithZeroWeights_ProducesZeroSequence()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var rnn = new Rnn(1, 1, false);
        ZeroParameter(rnn, "wxh", [0f], [1, 1]);
        ZeroParameter(rnn, "whh", [0f], [1, 1]);

        var input = CreateFloatArray([0.5f, -0.25f], [2, 1]);
        try
        {
            var output = rnn.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval rnn");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Has.Length.EqualTo(2));
                Assert.That(values[0], Is.EqualTo(0f).Within(1e-6));
                Assert.That(values[1], Is.EqualTo(0f).Within(1e-6));
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
    public void Gru_WithZeroWeights_ProducesZeroSequence()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var gru = new Gru(1, 1, true);
        ZeroParameter(gru, "wx", new float[3], [3, 1]);
        ZeroParameter(gru, "wh", new float[3], [3, 1]);
        ZeroParameter(gru, "bias", new float[3], [3]);
        ZeroParameter(gru, "bias_hidden", new float[1], [1]);

        var input = CreateFloatArray([0.5f, 0.25f], [2, 1]);
        try
        {
            var output = gru.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval gru");
                var values = TestHelpers.ToFloat32(output);
                Assert.That(values, Has.Length.EqualTo(2));
                Assert.That(values[0], Is.EqualTo(0f).Within(1e-6));
                Assert.That(values[1], Is.EqualTo(0f).Within(1e-6));
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
    public void Lstm_WithZeroWeights_ProducesZeroStates()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var lstm = new Lstm(1, 1, true);
        ZeroParameter(lstm, "wx", new float[4], [4, 1]);
        ZeroParameter(lstm, "wh", new float[4], [4, 1]);
        ZeroParameter(lstm, "bias", new float[4], [4]);

        var input = CreateFloatArray([1f, -1f], [2, 1]);
        try
        {
            var (hidden, cell) = lstm.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(hidden), "eval lstm hidden");
                TestHelpers.Ok(MlxArray.Eval(cell), "eval lstm cell");

                var hiddenValues = TestHelpers.ToFloat32(hidden);
                var cellValues = TestHelpers.ToFloat32(cell);

                Assert.That(hiddenValues, Has.Length.EqualTo(2));
                Assert.That(cellValues, Has.Length.EqualTo(2));
                Assert.That(hiddenValues[0], Is.EqualTo(0f).Within(1e-6));
                Assert.That(hiddenValues[1], Is.EqualTo(0f).Within(1e-6));
                Assert.That(cellValues[0], Is.EqualTo(0f).Within(1e-6));
                Assert.That(cellValues[1], Is.EqualTo(0f).Within(1e-6));
            }
            finally
            {
                if (hidden.ctx != 0)
                    MlxArray.Free(hidden);
                if (cell.ctx != 0)
                    MlxArray.Free(cell);
            }
        }
        finally
        {
            MlxArray.Free(input);
        }
    }

    private static MlxArrayHandle CreateFloatArray(float[] values, int[] shape)
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

    private static void ZeroParameter(Module module, string path, float[] values, int[] shape)
    {
        var replacement = new ParameterCollection();
        replacement.AddOrUpdate(path, new(CreateFloatArray(values, shape), true));
        module.UpdateParameters(replacement, false, true);
    }
}