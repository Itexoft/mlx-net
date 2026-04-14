// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Drops elements with probability <c>p</c> during training.
/// </summary>
public class Dropout : Module, IUnaryLayer
{
    private readonly float keepProbability;

    public Dropout(float p = 0.5f)
    {
        if (p < 0f || p >= 1f)
            throw new ArgumentOutOfRangeException(nameof(p), "Dropout probability must be in [0, 1).");

        this.keepProbability = 1f - p;
    }

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        if (!this.Training || Math.Abs(this.keepProbability - 1f) < float.Epsilon)
            return input;

        var maskBool = TensorFactory.Bernoulli(this.keepProbability, input.ShapeSpan());
        var dtype = MlxArray.DType(input);
        var mask = maskBool.AsType(dtype);
        var dropped = mask.Multiply(input);

        var scale = TensorFactory.Scalar(1f / this.keepProbability, dtype);
        var scaled = dropped.Multiply(scale);

        MlxArray.Free(scale);
        MlxArray.Free(dropped);
        MlxArray.Free(mask);
        MlxArray.Free(maskBool);

        return scaled;
    }
}

/// <summary>
/// Drops entire channels in 2D inputs during training.
/// </summary>
public sealed class Dropout2D : Module, IUnaryLayer
{
    private readonly float keepProbability;

    public Dropout2D(float p = 0.5f)
    {
        if (p < 0f || p >= 1f)
            throw new ArgumentOutOfRangeException(nameof(p), "Dropout probability must be in [0, 1).");

        this.keepProbability = 1f - p;
    }

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        if (!this.Training || Math.Abs(this.keepProbability - 1f) < float.Epsilon)
            return input;

        var ndim = input.Rank();

        if (ndim != 3 && ndim != 4)
            throw new ArgumentException("Dropout2d expects NWHC or WHC shaped inputs.");

        Span<int> maskShape = stackalloc int[ndim];
        input.ShapeSpan().CopyTo(maskShape);
        maskShape[^2] = 1;
        maskShape[^3] = 1;

        var maskBool = TensorFactory.Bernoulli(this.keepProbability, maskShape);
        var dtype = MlxArray.DType(input);
        var mask = maskBool.AsType(dtype);
        var dropped = mask.Multiply(input);

        var scale = TensorFactory.Scalar(1f / this.keepProbability, dtype);
        var scaled = dropped.Multiply(scale);

        MlxArray.Free(scale);
        MlxArray.Free(dropped);
        MlxArray.Free(mask);
        MlxArray.Free(maskBool);

        return scaled;
    }
}

/// <summary>
/// Drops entire channels in 3D inputs during training.
/// </summary>
public sealed class Dropout3D : Module, IUnaryLayer
{
    private readonly float keepProbability;

    public Dropout3D(float p = 0.5f)
    {
        if (p < 0f || p >= 1f)
            throw new ArgumentOutOfRangeException(nameof(p), "Dropout probability must be in [0, 1).");

        this.keepProbability = 1f - p;
    }

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        if (!this.Training || Math.Abs(this.keepProbability - 1f) < float.Epsilon)
            return input;

        var ndim = input.Rank();

        if (ndim != 4 && ndim != 5)
            throw new ArgumentException("Dropout3d expects NDHWC or DHWC shaped inputs.");

        Span<int> maskShape = stackalloc int[ndim];
        input.ShapeSpan().CopyTo(maskShape);
        maskShape[^2] = 1;
        maskShape[^3] = 1;
        maskShape[^4] = 1;

        var maskBool = TensorFactory.Bernoulli(this.keepProbability, maskShape);
        var dtype = MlxArray.DType(input);
        var mask = maskBool.AsType(dtype);
        var dropped = mask.Multiply(input);

        var scale = TensorFactory.Scalar(1f / this.keepProbability, dtype);
        var scaled = dropped.Multiply(scale);

        MlxArray.Free(scale);
        MlxArray.Free(dropped);
        MlxArray.Free(mask);
        MlxArray.Free(maskBool);

        return scaled;
    }
}
