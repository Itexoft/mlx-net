// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Applies layer normalization over the last dimension.
/// </summary>
public class LayerNorm : Module, IUnaryLayer
{
    private readonly ModuleParameter? _weight;
    private readonly ModuleParameter? _bias;
    private readonly float _eps;

    public LayerNorm(int dimensions, float eps = 1e-5f, bool affine = true)
    {
        this._eps = eps;
        if (affine)
        {
            this._weight = this.RegisterParameter("weight", TensorFactory.Ones([dimensions]));
            this._bias = this.RegisterParameter("bias", TensorFactory.Zeros([dimensions]));
        }
    }

    public ModuleParameter? Weight => this._weight;
    public ModuleParameter? Bias => this._bias;

    public float Epsilon => this._eps;

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var weight = this._weight?.Value ?? default;
        var bias = this._bias?.Value ?? default;
        var status = MlxFast.LayerNorm(out var result, input, weight, bias, this._eps, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "layer_norm");

        return result;
    }
}

/// <summary>
/// Applies Root Mean Square normalization.
/// </summary>
public class RmsNorm : Module, IUnaryLayer
{
    private readonly ModuleParameter _weight;
    private readonly float _eps;

    public RmsNorm(int dimensions, float eps = 1e-5f)
    {
        this._eps = eps;
        this._weight = this.RegisterParameter("weight", TensorFactory.Ones([dimensions]));
    }

    public ModuleParameter Weight => this._weight;

    public float Epsilon => this._eps;

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxFast.RmsNorm(out var result, input, this._weight.Value, this._eps, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "rms_norm");

        return result;
    }
}