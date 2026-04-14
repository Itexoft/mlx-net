// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Applies layer normalization over the last dimension.
/// </summary>
public class LayerNorm : Module, IUnaryLayer
{
    private readonly ModuleParameter? bias;
    private readonly float eps;
    private readonly ModuleParameter? weight;

    public LayerNorm(int dimensions, float eps = 1e-5f, bool affine = true)
    {
        this.eps = eps;

        if (affine)
        {
            this.weight = this.RegisterParameter("weight", TensorFactory.Ones([dimensions]));
            this.bias = this.RegisterParameter("bias", TensorFactory.Zeros([dimensions]));
        }
    }

    public ModuleParameter? Weight => this.weight;
    public ModuleParameter? Bias => this.bias;

    public float Epsilon => this.eps;

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var weight = this.weight?.Value ?? default;
        var bias = this.bias?.Value ?? default;
        var status = MlxFast.LayerNorm(out var result, input, weight, bias, this.eps, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "layer_norm");

        return result;
    }
}

/// <summary>
/// Applies Root Mean Square normalization.
/// </summary>
public class RmsNorm : Module, IUnaryLayer
{
    private readonly float eps;
    private readonly ModuleParameter weight;

    public RmsNorm(int dimensions, float eps = 1e-5f)
    {
        this.eps = eps;
        this.weight = this.RegisterParameter("weight", TensorFactory.Ones([dimensions]));
    }

    public ModuleParameter Weight => this.weight;

    public float Epsilon => this.eps;

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxFast.RmsNorm(out var result, input, this.weight.Value, this.eps, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "rms_norm");

        return result;
    }
}
