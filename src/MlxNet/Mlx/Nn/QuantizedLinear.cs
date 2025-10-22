// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Mlx.Nn;

public sealed class QuantizedLinear : Linear, IQuantized
{
    private readonly ModuleBuffer _scalesBuffer;
    private readonly ModuleBuffer? _biasesBuffer;

    public int GroupSize { get; }
    public int Bits { get; }
    public QuantizationMode Mode { get; }

    public QuantizedLinear(
        int inputDimensions,
        int outputDimensions,
        bool bias = true,
        int groupSize = 64,
        int bits = 4,
        QuantizationMode mode = QuantizationMode.Affine)
        : this(
            CreateRandomWeight(inputDimensions, outputDimensions),
            bias ? CreateRandomBias(inputDimensions, outputDimensions) : (MlxArrayHandle?)null,
            groupSize,
            bits,
            mode) { }

    public QuantizedLinear(Linear other, int groupSize = 64, int bits = 4, QuantizationMode mode = QuantizationMode.Affine)
        : this(other.Weight.Value.Copy(), other.Bias is null ? null : other.Bias.Value.Copy(), groupSize, bits, mode) { }

    public QuantizedLinear(MlxArrayHandle weight, MlxArrayHandle? bias, int groupSize, int bits, QuantizationMode mode)
        : base(QuantizeWeights(weight, groupSize, bits, mode, out var scales, out var biases), bias, false)
    {
        this.GroupSize = groupSize;
        this.Bits = bits;
        this.Mode = mode;

        this._scalesBuffer = this.RegisterBuffer("scales", scales);
        this._biasesBuffer = TensorUtilities.IsNull(biases) ? null : this.RegisterBuffer("quant_biases", biases);

        this.Weight.Trainable = false;
        if (this.Bias is { } biasParameter)
            biasParameter.Trainable = false;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var biasesHandle = this._biasesBuffer?.Value ?? default;
        var status = MlxOps.QuantizedMatmul(
            out var result,
            input,
            this.Weight.Value,
            this._scalesBuffer.Value,
            biasesHandle,
            true,
            this.GroupSize,
            this.Bits,
            this.Mode.ToNativeString(),
            TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "quantized_matmul");

        if (this.Bias is { } bias)
        {
            var withBias = result.Add(bias.Value);
            MlxArray.Free(result);
            result = withBias;
        }

        return result;
    }

    private static MlxArrayHandle QuantizeWeights(
        MlxArrayHandle weight,
        int groupSize,
        int bits,
        QuantizationMode mode,
        out MlxArrayHandle scales,
        out MlxArrayHandle biases)
    {
        var status = MlxOps.Quantize(
            out var packed,
            weight,
            groupSize,
            bits,
            mode.ToNativeString(),
            TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "quantize");
        MlxArray.Free(weight);

        try
        {
            TensorUtilities.CheckStatus(MlxVector.ArrayGet(out var quantized, packed, 0), "quantize[0]");
            TensorUtilities.CheckStatus(MlxVector.ArrayGet(out scales, packed, 1), "quantize[1]");

            biases = default;
            if (MlxVector.ArraySize(packed) > 2)
                TensorUtilities.CheckStatus(MlxVector.ArrayGet(out biases, packed, 2), "quantize[2]");

            return quantized;
        }
        finally
        {
            MlxVector.ArrayFree(packed);
        }
    }

    private static MlxArrayHandle CreateRandomWeight(int inputDimensions, int outputDimensions)
    {
        var scale = MathF.Sqrt(1f / inputDimensions);

        return TensorFactory.Uniform(-scale, scale, [outputDimensions, inputDimensions]);
    }

    private static MlxArrayHandle CreateRandomBias(int inputDimensions, int outputDimensions)
    {
        var scale = MathF.Sqrt(1f / inputDimensions);

        return TensorFactory.Uniform(-scale, scale, [outputDimensions]);
    }
}
