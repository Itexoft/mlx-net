// Copyright (c) 2011-2026 Denis Kudelin
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
    private readonly ModuleParameter? bias;
    private readonly ModuleParameter weight;

    public Linear(int inputDimensions, int outputDimensions, bool bias = true)
    {
        var scale = (float)Math.Sqrt(1.0f / inputDimensions);
        var weight = TensorFactory.Uniform(-scale, scale, [outputDimensions, inputDimensions]);
        this.weight = this.RegisterParameter("weight", weight);

        if (bias)
        {
            var biasHandle = TensorFactory.Uniform(-scale, scale, [outputDimensions]);
            this.bias = this.RegisterParameter("bias", biasHandle);
        }
    }

    public Linear(MlxArrayHandle weight, MlxArrayHandle? bias = null, bool trainable = true)
    {
        this.weight = this.RegisterParameter("weight", weight, trainable);

        if (bias is { } biasHandle)
            this.bias = this.RegisterParameter("bias", biasHandle, trainable);
    }

    public ModuleParameter Weight => this.weight;

    public ModuleParameter? Bias => this.bias;

    Module IQuantizable.ToQuantized(int groupSize, int bits, QuantizationMode mode) => new QuantizedLinear(this, groupSize, bits, mode);

    public virtual MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var weightT = this.weight.Value.Transpose();

        try
        {
            var projected = input.Matmul(weightT);

            if (this.bias is { } bias)
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
}
