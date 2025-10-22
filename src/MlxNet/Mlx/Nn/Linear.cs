// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Applies an affine transformation <c>x ↦ x · Wᵀ + b</c>.
/// </summary>
public class Linear : Module, IUnaryLayer, IQuantizable
{
    private readonly ModuleParameter _weight;
    private readonly ModuleParameter? _bias;

    public Linear(int inputDimensions, int outputDimensions, bool bias = true)
    {
        var scale = (float)Math.Sqrt(1.0f / inputDimensions);
        var weight = TensorFactory.Uniform(-scale, scale, [outputDimensions, inputDimensions]);
        this._weight = this.RegisterParameter("weight", weight);

        if (bias)
        {
            var biasHandle = TensorFactory.Uniform(-scale, scale, [outputDimensions]);
            this._bias = this.RegisterParameter("bias", biasHandle);
        }
    }

    public Linear(MlxArrayHandle weight, MlxArrayHandle? bias = null, bool trainable = true)
    {
        this._weight = this.RegisterParameter("weight", weight, trainable);
        if (bias is { } biasHandle)
            this._bias = this.RegisterParameter("bias", biasHandle, trainable);
    }

    public ModuleParameter Weight => this._weight;

    public ModuleParameter? Bias => this._bias;

    public virtual MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var weightT = this._weight.Value.Transpose();
        try
        {
            var projected = input.Matmul(weightT);

            if (this._bias is { } bias)
            {
                var withBias = projected.Add(bias.Value);
                MlxArray.Free(projected);

                return withBias;
            }

            return projected;
        }
        finally
        {
            if (!TensorUtilities.IsNull(weightT))
                MlxArray.Free(weightT);
        }
    }

    Module IQuantizable.ToQuantized(int groupSize, int bits, QuantizationMode mode)
        => new QuantizedLinear(this, groupSize, bits, mode);
}