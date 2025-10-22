// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// A placeholder activation that returns its argument unchanged.
/// </summary>
public sealed class Identity : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input) => input;
}

/// <summary>
/// Applies the logistic sigmoid function element-wise.
/// </summary>
public sealed class Sigmoid : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Sigmoid(out var result, input, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "sigmoid");

        return result;
    }
}

/// <summary>
/// Applies the hyperbolic tangent function element-wise.
/// </summary>
public sealed class Tanh : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Tanh(out var result, input, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "tanh");

        return result;
    }
}

/// <summary>
/// Applies the Rectified Linear Unit.
/// </summary>
public sealed class ReLU : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var dtype = MlxArray.DType(input);
        var zero = TensorFactory.Scalar(0f, dtype);
        try
        {
            var result = input.Maximum(zero);

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(zero))
                MlxArray.Free(zero);
        }
    }
}

/// <summary>
/// Applies a leaky variant of the Rectified Linear Unit.
/// </summary>
public sealed class LeakyReLU(float negativeSlope = 0.01f) : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var dtype = MlxArray.DType(input);
        var slope = TensorFactory.Scalar(negativeSlope, dtype);
        try
        {
            var scaled = input.Multiply(slope);
            try
            {
                var result = scaled.Maximum(input);

                return result;
            }
            finally
            {
                MlxArray.Free(scaled);
            }
        }
        finally
        {
            if (!TensorUtilities.IsNull(slope))
                MlxArray.Free(slope);
        }
    }
}

/// <summary>
/// Applies the Sigmoid Linear Unit (also known as Swish).
/// </summary>
public sealed class SiLU : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Sigmoid(out var sigma, input, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "sigmoid");
        var result = input.Multiply(sigma);
        MlxArray.Free(sigma);

        return result;
    }
}

/// <summary>
/// Applies the Gaussian Error Linear Unit using the exact formulation.
/// </summary>
public sealed class Gelu : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var dtype = MlxArray.DType(input);
        var sqrt2 = TensorFactory.Scalar((float)Math.Sqrt(2.0), dtype);
        var half = TensorFactory.Scalar(0.5f, dtype);
        var one = TensorFactory.Scalar(1.0f, dtype);

        MlxArrayHandle scaled = default;
        MlxArrayHandle erf = default;
        MlxArrayHandle term = default;
        MlxArrayHandle prod = default;
        try
        {
            scaled = input.Divide(sqrt2);
            erf = scaled.Erf();
            term = erf.Add(one);
            prod = input.Multiply(term);
            var result = prod.Multiply(half);

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(scaled))
                MlxArray.Free(scaled);
            if (!TensorUtilities.IsNull(erf))
                MlxArray.Free(erf);
            if (!TensorUtilities.IsNull(term))
                MlxArray.Free(term);
            if (!TensorUtilities.IsNull(prod))
                MlxArray.Free(prod);
            if (!TensorUtilities.IsNull(sqrt2))
                MlxArray.Free(sqrt2);
            if (!TensorUtilities.IsNull(half))
                MlxArray.Free(half);
            if (!TensorUtilities.IsNull(one))
                MlxArray.Free(one);
        }
    }
}

/// <summary>
/// Applies the softmax function along a specified axis.
/// </summary>
public sealed class Softmax(int axis = -1) : Module, IUnaryLayer
{
    public int Axis { get; } = axis;

    public MlxArrayHandle Forward(MlxArrayHandle input)
        => input.Softmax(this.Axis);
}