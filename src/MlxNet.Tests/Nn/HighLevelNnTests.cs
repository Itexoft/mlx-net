// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using NUnit.Framework;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;

namespace MlxNet.Tests.Nn;

[TestFixture]
public unsafe class HighLevelNnTests
{
    [Test]
    public void Linear_ForwardsExpectedValues()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var linear = new Linear(2, 2, true);

        var weightData = stackalloc float[] { 1f, 2f, 3f, 4f };
        var weightShape = stackalloc int[] { 2, 2 };
        var weight = MlxArray.NewData(weightData, weightShape, 2, MlxDType.MLX_FLOAT32);
        linear.Weight.SetValue(weight);

        var biasData = stackalloc float[] { 0f, 1f };
        var biasShape = stackalloc int[] { 2 };
        var bias = MlxArray.NewData(biasData, biasShape, 1, MlxDType.MLX_FLOAT32);
        linear.Bias!.SetValue(bias);

        var inputData = stackalloc float[] { 1f, 2f };
        var inputShape = stackalloc int[] { 1, 2 };
        var input = MlxArray.NewData(inputData, inputShape, 2, MlxDType.MLX_FLOAT32);

        var result = linear.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(result), "eval");

        var values = TestHelpers.ToFloat32(result);
        Assert.That(values[0], Is.EqualTo(5f).Within(1e-4));
        Assert.That(values[1], Is.EqualTo(12f).Within(1e-4));

        MlxArray.Free(result);
        MlxArray.Free(input);
    }

    [Test]
    public void LayerNorm_NormalizesPerFeature()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var norm = new LayerNorm(2);
        var inputData = stackalloc float[] { 1f, 3f };
        var inputShape = stackalloc int[] { 1, 2 };
        var input = MlxArray.NewData(inputData, inputShape, 2, MlxDType.MLX_FLOAT32);

        var result = norm.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(result), "eval");

        var values = TestHelpers.ToFloat32(result);
        Assert.That(values[0], Is.EqualTo(-1f).Within(1e-4));
        Assert.That(values[1], Is.EqualTo(1f).Within(1e-4));

        MlxArray.Free(result);
        MlxArray.Free(input);
    }

    [Test]
    public void MultiHeadAttention_ProducesCorrectShape()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var mha = new MultiHeadAttention(4, 2, bias: false);

        var stream = stackalloc int[] { 1, 3, 4 };

        var queriesData = stackalloc float[12];
        for (var i = 0; i < 12; i++)
            queriesData[i] = 0f;
        var queries = MlxArray.NewData(queriesData, stream, 3, MlxDType.MLX_FLOAT32);
        var keysData = stackalloc float[12];
        for (var i = 0; i < 12; i++)
            keysData[i] = 0f;
        var keys = MlxArray.NewData(keysData, stream, 3, MlxDType.MLX_FLOAT32);
        var valuesData = stackalloc float[12];
        for (var i = 0; i < 12; i++)
            valuesData[i] = 0f;
        var values = MlxArray.NewData(valuesData, stream, 3, MlxDType.MLX_FLOAT32);

        var output = mha.Forward(queries, keys, values);
        TestHelpers.Ok(MlxArray.Eval(output), "eval");

        var shape = TestHelpers.ShapeOf(output);
        Assert.That(shape, Is.EqualTo(new[] { 1, 3, 4 }));

        MlxArray.Free(output);
        MlxArray.Free(values);
        MlxArray.Free(keys);
        MlxArray.Free(queries);
    }

    [Test]
    public void Conv1d_ComputesSlidingWindowSum()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var conv = new Conv1d(1, 1, 3, 1, 0, 1, 1, false);

        var weightData = stackalloc float[] { 1f, 1f, 1f };
        var weightShape = stackalloc int[] { 1, 3, 1 };
        var weight = MlxArray.NewData(weightData, weightShape, 3, MlxDType.MLX_FLOAT32);
        conv.KernelParameter.SetValue(weight);

        var inputData = stackalloc float[] { 1f, 2f, 3f, 4f, 5f };
        var inputShape = stackalloc int[] { 1, 5, 1 };
        var input = MlxArray.NewData(inputData, inputShape, 3, MlxDType.MLX_FLOAT32);

        var output = conv.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(output), "eval");

        var values = TestHelpers.ToFloat32(output);
        Assert.That(values, Is.EqualTo(new[] { 6f, 9f, 12f }).Within(1e-5));

        MlxArray.Free(output);
        MlxArray.Free(input);
    }

    [Test]
    public void ConvTranspose1d_ReconstructsExpectedSignal()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var conv = new ConvTranspose1d(1, 1, 3, 1, 0, 1, 1, bias: false);

        var weightData = stackalloc float[] { 1f, 1f, 1f };
        var weightShape = stackalloc int[] { 1, 3, 1 };
        var weight = MlxArray.NewData(weightData, weightShape, 3, MlxDType.MLX_FLOAT32);
        conv.KernelParameter.SetValue(weight);

        var inputValues = new[] { 1f, 2f, 3f };
        var inputData = stackalloc float[inputValues.Length];
        for (var i = 0; i < inputValues.Length; i++)
            inputData[i] = inputValues[i];
        var inputShape = stackalloc int[] { 1, inputValues.Length, 1 };
        var input = MlxArray.NewData(inputData, inputShape, 3, MlxDType.MLX_FLOAT32);

        var output = conv.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(output), "eval");

        var expectedLength = (inputValues.Length - 1) * 1 - 2 * 0 + (3 - 1) * 1 + 1;
        var expected = new float[expectedLength];
        for (var i = 0; i < inputValues.Length; i++)
        for (var k = 0; k < 3; k++)
        {
            var y = i * 1 - 0 + k * 1;
            if (y >= 0 && y < expectedLength)
                expected[y] += inputValues[i];
        }

        var values = TestHelpers.ToFloat32(output);
        Assert.That(values, Is.EqualTo(expected).Within(1e-5));

        MlxArray.Free(output);
        MlxArray.Free(input);
    }

    [Test]
    public void ConvTranspose2d_ReconstructsExpectedSpatialSignal()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var conv = new ConvTranspose2d(
            1,
            1,
            (2, 2),
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            bias: false);

        var weightData = stackalloc float[] { 1f, 1f, 1f, 1f };
        var weightShape = stackalloc int[] { 1, 2, 2, 1 };
        var weight = MlxArray.NewData(weightData, weightShape, 4, MlxDType.MLX_FLOAT32);
        conv.KernelParameter.SetValue(weight);

        var inputHeight = 2;
        var inputWidth = 2;
        var inputValues = new float[,] { { 1f, 2f }, { 3f, 4f } };
        var inputData = stackalloc float[inputHeight * inputWidth];
        var index = 0;
        for (var h = 0; h < inputHeight; h++)
        for (var w = 0; w < inputWidth; w++)
            inputData[index++] = inputValues[h, w];

        var inputShape = stackalloc int[] { 1, inputHeight, inputWidth, 1 };
        var input = MlxArray.NewData(inputData, inputShape, 4, MlxDType.MLX_FLOAT32);

        var output = conv.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(output), "eval");

        var outputHeight = (inputHeight - 1) * 1 - 2 * 0 + (2 - 1) * 1 + 1;
        var outputWidth = (inputWidth - 1) * 1 - 2 * 0 + (2 - 1) * 1 + 1;
        var expected = new float[outputHeight * outputWidth];

        for (var h = 0; h < inputHeight; h++)
        for (var w = 0; w < inputWidth; w++)
        {
            var value = inputValues[h, w];
            for (var kh = 0; kh < 2; kh++)
            for (var kw = 0; kw < 2; kw++)
            {
                var y = h * 1 - 0 + kh * 1;
                var x = w * 1 - 0 + kw * 1;
                if (y >= 0 && y < outputHeight && x >= 0 && x < outputWidth)
                    expected[y * outputWidth + x] += value;
            }
        }

        var values = TestHelpers.ToFloat32(output);
        Assert.That(values, Is.EqualTo(expected).Within(1e-5));

        MlxArray.Free(output);
        MlxArray.Free(input);
    }

    [Test]
    public void MaxPool2d_ReducesSpatialDimensions()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var pool = new MaxPool2d((2, 2), (2, 2));

        var data = stackalloc float[16];
        for (var i = 0; i < 16; i++)
            data[i] = i + 1;
        var shape = stackalloc int[] { 1, 4, 4, 1 };
        var input = MlxArray.NewData(data, shape, 4, MlxDType.MLX_FLOAT32);

        var output = pool.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(output), "eval");

        var expected = new[] { 6f, 8f, 14f, 16f };
        var values = TestHelpers.ToFloat32(output);
        Assert.That(values, Is.EqualTo(expected).Within(1e-5));

        var outputShape = TestHelpers.ShapeOf(output);
        Assert.That(outputShape, Is.EqualTo(new[] { 1, 2, 2, 1 }));

        MlxArray.Free(output);
        MlxArray.Free(input);
    }

    [Test]
    public void AvgPool1d_ComputesAverages()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var pool = new AvgPool1d(2, 2);

        var data = stackalloc float[] { 2f, 4f, 6f, 8f };
        var shape = stackalloc int[] { 1, 4, 1 };
        var input = MlxArray.NewData(data, shape, 3, MlxDType.MLX_FLOAT32);

        var output = pool.Forward(input);
        TestHelpers.Ok(MlxArray.Eval(output), "eval");

        var values = TestHelpers.ToFloat32(output);
        Assert.That(values, Is.EqualTo(new[] { 3f, 7f }).Within(1e-5));

        MlxArray.Free(output);
        MlxArray.Free(input);
    }

    [Test]
    public void MseLoss_IsZeroForIdenticalInputs()
    {
        TestHelpers.RequireNativeOrIgnore();

        var data = stackalloc float[] { 1f, 2f, 3f };
        var shape = stackalloc int[] { 3 };
        var predictions = MlxArray.NewData(data, shape, 1, MlxDType.MLX_FLOAT32);
        var targets = MlxArray.NewData(data, shape, 1, MlxDType.MLX_FLOAT32);

        var loss = Losses.MseLoss(predictions, targets, LossReduction.Sum);
        TestHelpers.Ok(MlxArray.Eval(loss), "eval");

        var values = TestHelpers.ToFloat32(loss);
        Assert.That(values[0], Is.EqualTo(0f).Within(1e-6));

        MlxArray.Free(loss);
        MlxArray.Free(predictions);
        MlxArray.Free(targets);
    }

    [Test]
    public void CrossEntropy_MatchesLogTwoForUniformLogits()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logitsData = stackalloc float[] { 0f, 0f };
        var logitsShape = stackalloc int[] { 1, 2 };
        var logits = MlxArray.NewData(logitsData, logitsShape, 2, MlxDType.MLX_FLOAT32);

        var targetsData = stackalloc int[] { 0 };
        var targetsShape = stackalloc int[] { 1 };
        var targets = MlxArray.NewData(targetsData, targetsShape, 1, MlxDType.MLX_INT32);

        var loss = Losses.CrossEntropy(logits, targets, reduction: LossReduction.Mean);
        TestHelpers.Ok(MlxArray.Eval(loss), "eval");

        var values = TestHelpers.ToFloat32(loss);
        Assert.That(values[0], Is.EqualTo(MathF.Log(2f)).Within(1e-5));

        MlxArray.Free(loss);
        MlxArray.Free(logits);
        MlxArray.Free(targets);
    }
}