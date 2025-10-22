// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Multi-head scaled dot-product attention.
/// </summary>
public class MultiHeadAttention : Module
{
    private readonly Linear _queryProjection;
    private readonly Linear _keyProjection;
    private readonly Linear _valueProjection;
    private readonly Linear _outProjection;

    public MultiHeadAttention(
        int dimensions,
        int numHeads,
        int? queryInputDimensions = null,
        int? keyInputDimensions = null,
        int? valueInputDimensions = null,
        int? valueDimensions = null,
        int? valueOutputDimensions = null,
        bool bias = false)
    {
        if (dimensions % numHeads != 0)
            throw new ArgumentException("Dimensions must be divisible by the number of heads.", nameof(dimensions));

        this.NumHeads = numHeads;

        var qInput = queryInputDimensions ?? dimensions;
        var kInput = keyInputDimensions ?? dimensions;
        var vInput = valueInputDimensions ?? dimensions;
        var vDim = valueDimensions ?? dimensions;
        var vOutDim = valueOutputDimensions ?? dimensions;

        this._queryProjection = this.RegisterModule("query_proj", new Linear(qInput, dimensions, bias));
        this._keyProjection = this.RegisterModule("key_proj", new Linear(kInput, dimensions, bias));
        this._valueProjection = this.RegisterModule("value_proj", new Linear(vInput, vDim, bias));
        this._outProjection = this.RegisterModule("out_proj", new Linear(vDim, vOutDim, bias));
    }

    public int NumHeads { get; }

    public Linear QueryProjection => this._queryProjection;

    public Linear KeyProjection => this._keyProjection;

    public Linear ValueProjection => this._valueProjection;

    public Linear OutProjection => this._outProjection;

    public static MlxArrayHandle CreateAdditiveCausalMask(int n, MlxDType dtype = MlxDType.MLX_FLOAT32)
    {
        var indices = TensorFactory.Arange(0, n, 1, MlxDType.MLX_INT32);
        var rows = indices.ExpandedDimension(1);
        var cols = indices.ExpandedDimension(0);
        var maskBool = rows.LessThan(cols);
        var mask = maskBool.AsType(dtype);
        var scale = TensorFactory.Scalar(-1e9f, dtype);
        var scaled = mask.Multiply(scale);

        MlxArray.Free(scale);
        MlxArray.Free(mask);
        MlxArray.Free(maskBool);
        MlxArray.Free(cols);
        MlxArray.Free(rows);
        MlxArray.Free(indices);

        return scaled;
    }

    public MlxArrayHandle Forward(
        MlxArrayHandle queries,
        MlxArrayHandle keys,
        MlxArrayHandle values,
        MlxArrayHandle? mask = null)
    {
        var qProj = this._queryProjection.Forward(queries);
        var kProj = this._keyProjection.Forward(keys);
        var vProj = this._valueProjection.Forward(values);

        try
        {
            var batch = qProj.Dim(0);
            var targetLen = qProj.Dim(1);
            var embedDim = qProj.Dim(2);
            var headDim = embedDim / this.NumHeads;
            var sourceLen = kProj.Dim(1);
            var valueEmbed = vProj.Dim(2);
            var valueDim = valueEmbed / this.NumHeads;

            var qHeads = qProj.Reshape(batch, targetLen, this.NumHeads, headDim).Transposed(0, 2, 1, 3);
            var kHeads = kProj.Reshape(batch, sourceLen, this.NumHeads, headDim).Transposed(0, 2, 3, 1);
            var vHeads = vProj.Reshape(batch, sourceLen, this.NumHeads, valueDim).Transposed(0, 2, 1, 3);

            var scale = TensorFactory.Scalar((float)Math.Sqrt(1.0f / headDim), MlxArray.DType(qHeads));
            var scaledQueries = qHeads.Multiply(scale);
            MlxArray.Free(scale);

            var scores = scaledQueries.Matmul(kHeads);
            MlxArray.Free(scaledQueries);
            MlxArray.Free(qHeads);
            MlxArray.Free(kHeads);

            if (mask is { } maskHandle && !TensorUtilities.IsNull(maskHandle))
            {
                var castMask = maskHandle.AsType(MlxArray.DType(scores));
                var maskedScores = scores.Add(castMask);
                MlxArray.Free(scores);
                scores = maskedScores;
                MlxArray.Free(castMask);
            }

            var attention = scores.Softmax(-1);
            MlxArray.Free(scores);

            var weighted = attention.Matmul(vHeads);
            MlxArray.Free(attention);
            MlxArray.Free(vHeads);

            var transposed = weighted.Transposed(0, 2, 1, 3);
            MlxArray.Free(weighted);

            var reshaped = transposed.Reshape(batch, targetLen, this.NumHeads * valueDim);
            MlxArray.Free(transposed);

            var output = this._outProjection.Forward(reshaped);
            MlxArray.Free(reshaped);

            return output;
        }
        finally
        {
            MlxArray.Free(qProj);
            MlxArray.Free(kProj);
            MlxArray.Free(vProj);
        }
    }

    public MlxArrayHandle Forward(
        MlxArrayHandle input,
        MlxArrayHandle? mask = null)
        => this.Forward(input, input, input, mask);
}

/// <summary>
/// Transformer encoder block consisting of multi-head attention and feed-forward sublayers.
/// </summary>
public class TransformerEncoderLayer : Module, IUnaryLayer
{
    private readonly MultiHeadAttention _attention;
    private readonly LayerNorm _ln1;
    private readonly LayerNorm _ln2;
    private readonly Linear _linear1;
    private readonly Linear _linear2;
    private readonly Dropout _dropout1;
    private readonly Dropout _dropout2;
    private readonly IUnaryLayer _activation;
    private readonly bool _normFirst;

    public TransformerEncoderLayer(
        int dimensions,
        int numHeads,
        int? mlpDimensions = null,
        float dropout = 0f,
        IUnaryLayer? activation = null,
        bool normFirst = false)
    {
        this._attention = this.RegisterModule("self_attn", new MultiHeadAttention(dimensions, numHeads));
        this._ln1 = this.RegisterModule("norm1", new LayerNorm(dimensions));
        this._ln2 = this.RegisterModule("norm2", new LayerNorm(dimensions));

        var hidden = mlpDimensions ?? dimensions * 4;
        this._linear1 = this.RegisterModule("linear1", new Linear(dimensions, hidden));
        this._linear2 = this.RegisterModule("linear2", new Linear(hidden, dimensions));

        this._dropout1 = this.RegisterModule("dropout1", new Dropout(dropout));
        this._dropout2 = this.RegisterModule("dropout2", new Dropout(dropout));

        this._activation = activation ?? new ReLU();
        if (this._activation is Module moduleActivation)
            this.RegisterModule("activation", moduleActivation);

        this._normFirst = normFirst;
    }

    public MlxArrayHandle Forward(MlxArrayHandle input, MlxArrayHandle? mask = null)
        => this._normFirst ? this.ForwardNormFirst(input, mask) : this.ForwardPost(input, mask);

    MlxArrayHandle IUnaryLayer.Forward(MlxArrayHandle input) => this.Forward(input, null);

    private MlxArrayHandle ForwardNormFirst(MlxArrayHandle input, MlxArrayHandle? mask)
    {
        var x = input;

        var normed = this._ln1.Forward(x);
        var attn = this._attention.Forward(normed, normed, normed, mask);
        MlxArray.Free(normed);

        var droppedAttn = this._dropout1.Forward(attn);
        ReleaseIfDistinct(attn, droppedAttn);

        var residual1 = x.Add(droppedAttn);
        MlxArray.Free(droppedAttn);

        var normed2 = this._ln2.Forward(residual1);
        var ff1 = this._linear1.Forward(normed2);
        MlxArray.Free(normed2);

        var activated = this._activation.Forward(ff1);
        MlxArray.Free(ff1);

        var dropped = this._dropout2.Forward(activated);
        ReleaseIfDistinct(activated, dropped);

        var ff2 = this._linear2.Forward(dropped);
        MlxArray.Free(dropped);

        var output = residual1.Add(ff2);
        MlxArray.Free(ff2);
        MlxArray.Free(residual1);

        return output;
    }

    private MlxArrayHandle ForwardPost(MlxArrayHandle input, MlxArrayHandle? mask)
    {
        var x = input;

        var attn = this._attention.Forward(x, x, x, mask);
        var droppedAttn = this._dropout1.Forward(attn);
        ReleaseIfDistinct(attn, droppedAttn);

        var residual1 = x.Add(droppedAttn);
        MlxArray.Free(droppedAttn);

        var normed1 = this._ln1.Forward(residual1);
        var ff1 = this._linear1.Forward(normed1);
        MlxArray.Free(normed1);

        var activated = this._activation.Forward(ff1);
        MlxArray.Free(ff1);

        var dropped = this._dropout2.Forward(activated);
        ReleaseIfDistinct(activated, dropped);

        var ff2 = this._linear2.Forward(dropped);
        MlxArray.Free(dropped);

        var residual2 = residual1.Add(ff2);
        MlxArray.Free(ff2);
        MlxArray.Free(residual1);

        var output = this._ln2.Forward(residual2);
        MlxArray.Free(residual2);

        return output;
    }

    private static void ReleaseIfDistinct(MlxArrayHandle original, MlxArrayHandle result)
    {
        if (!TensorUtilities.IsNull(original) && original.ctx != result.ctx)
            MlxArray.Free(original);
    }
}