// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Mlx.Nn;

public sealed class QuantizedEmbedding : Embedding, IQuantized
{
    private readonly ModuleBuffer _scalesBuffer;
    private readonly ModuleBuffer? _biasesBuffer;

    public int GroupSize { get; }
    public int Bits { get; }
    public QuantizationMode Mode { get; }

    public QuantizedEmbedding(
        int embeddingCount,
        int dimensions,
        int groupSize = 64,
        int bits = 4,
        QuantizationMode mode = QuantizationMode.Affine)
        : this(CreateRandomWeight(embeddingCount, dimensions), groupSize, bits, mode) { }

    public QuantizedEmbedding(Embedding other, int groupSize = 64, int bits = 4, QuantizationMode mode = QuantizationMode.Affine)
        : this(other.Weight.Value.Copy(), groupSize, bits, mode) { }

    public QuantizedEmbedding(MlxArrayHandle weight, int groupSize, int bits, QuantizationMode mode)
        : base(QuantizeWeight(weight, groupSize, bits, mode, out var scales, out var biases), false)
    {
        this.GroupSize = groupSize;
        this.Bits = bits;
        this.Mode = mode;

        this._scalesBuffer = this.RegisterBuffer("scales", scales);
        this._biasesBuffer = TensorUtilities.IsNull(biases) ? null : this.RegisterBuffer("quant_biases", biases);

        this.Weight.Trainable = false;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle indices)
    {
        var originalShape = indices.Shape();
        var flatSize = checked((int)MlxArray.Size(indices));

        var flat = new int[] { flatSize };
        var flattened = indices.Reshape(flat);
        var flatIndices = flattened.AsType(MlxDType.MLX_INT32);
        MlxArray.Free(flattened);

        var weightRows = GatherRows(this.Weight.Value, flatIndices);
        var scaleRows = GatherRows(this._scalesBuffer.Value, flatIndices);

        MlxArrayHandle biasRows = default;
        if (this._biasesBuffer is not null)
            biasRows = GatherRows(this._biasesBuffer.Value, flatIndices);

        var status = MlxOps.Dequantize(
            out var dequantized,
            weightRows,
            scaleRows,
            biasRows,
            this.GroupSize,
            this.Bits,
            this.Mode.ToNativeString(),
            TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "dequantize");

        MlxArray.Free(weightRows);
        MlxArray.Free(scaleRows);
        if (!TensorUtilities.IsNull(biasRows))
            MlxArray.Free(biasRows);

        var embeddingShape = dequantized.Shape();
        var embeddingDimension = embeddingShape[^1];

        var outputShape = new int[originalShape.Length + 1];
        for (var i = 0; i < originalShape.Length; i++)
            outputShape[i] = originalShape[i];
        outputShape[^1] = embeddingDimension;

        var output = dequantized.Reshape(outputShape);
        MlxArray.Free(dequantized);
        MlxArray.Free(flatIndices);

        return output;
    }

    public override MlxArrayHandle AsLinear(MlxArrayHandle input)
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

        return result;
    }

    private static MlxArrayHandle QuantizeWeight(
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

    private static MlxArrayHandle GatherRows(MlxArrayHandle matrix, MlxArrayHandle indices)
    {
        var matrixShape = matrix.Shape();
        var rank = matrixShape.Length;
        var count = indices.Shape()[0];

        var reshapeIndices = indices.Reshape(new[] { count, 1 });

        var broadcastShape = new int[rank];
        broadcastShape[0] = count;
        for (var i = 1; i < rank; i++)
            broadcastShape[i] = matrixShape[i];

        var broadcast = reshapeIndices.BroadcastTo(broadcastShape);
        MlxArray.Free(reshapeIndices);

        var gathered = matrix.TakeAlong(broadcast, 0);
        MlxArray.Free(broadcast);

        return gathered;
    }

    private static MlxArrayHandle CreateRandomWeight(int embeddingCount, int dimensions)
    {
        var scale = MathF.Sqrt(1f / dimensions);

        return TensorFactory.Normal(0f, scale, [embeddingCount, dimensions]);
    }
}
