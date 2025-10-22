// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Provides a lookup table that maps token indices to embedding vectors.
/// </summary>
public class Embedding : Module, IUnaryLayer, IQuantizable
{
    private readonly ModuleParameter _weight;

    public Embedding(int embeddingCount, int dimensions)
    {
        var scale = (float)System.Math.Sqrt(1.0f / dimensions);
        var weight = TensorFactory.Normal(0f, scale, [embeddingCount, dimensions]);
        this._weight = this.RegisterParameter("weight", weight);
    }

    public Embedding(MlxArrayHandle weight, bool trainable = true) => this._weight = this.RegisterParameter("weight", weight, trainable);

    public ModuleParameter Weight => this._weight;

    public virtual MlxArrayHandle Forward(MlxArrayHandle indices)
    {
        var status = MlxOps.TakeAxis(out var result, this._weight.Value, indices, 0, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "embedding_take");

        return result;
    }

    /// <summary>
    /// Reinterprets the embedding weights as a linear projection.
    /// </summary>
    public virtual MlxArrayHandle AsLinear(MlxArrayHandle input)
    {
        var weightT = this._weight.Value.Transpose();
        try
        {
            return input.Matmul(weightT);
        }
        finally
        {
            if (!TensorUtilities.IsNull(weightT))
                MlxArray.Free(weightT);
        }
    }

    Module IQuantizable.ToQuantized(int groupSize, int bits, QuantizationMode mode)
        => new QuantizedEmbedding(this, groupSize, bits, mode);
}