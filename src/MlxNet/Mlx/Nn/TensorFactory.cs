// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Threading;

namespace Itexoft.Mlx.Nn;

internal static class TensorFactory
{
    private static long _seedCounter = DateTime.UtcNow.Ticks;

    public static void Seed(ulong seed)
    {
        Interlocked.Exchange(ref _seedCounter, unchecked((long)seed));
    }

    public static MlxArrayHandle Scalar(float value, MlxDType dtype = MlxDType.MLX_FLOAT32)
        => CreateScalar(value, dtype);

    public static unsafe MlxArrayHandle Full(float value, ReadOnlySpan<int> shape, MlxDType dtype = MlxDType.MLX_FLOAT32)
    {
        var fill = CreateScalar(value, dtype);
        try
        {
            fixed (int* shapePtr = shape)
            {
                var status = MlxOps.Full(
                    out var result,
                    shapePtr,
                    (nuint)shape.Length,
                    fill,
                    dtype,
                    TensorUtilities.DefaultStream());
                TensorUtilities.CheckStatus(status, "full");

                return result;
            }
        }
        finally
        {
            if (!TensorUtilities.IsNull(fill))
                MlxArray.Free(fill);
        }
    }

    public static MlxArrayHandle Zeros(ReadOnlySpan<int> shape, MlxDType dtype = MlxDType.MLX_FLOAT32)
        => Full(0f, shape, dtype);

    public static MlxArrayHandle Ones(ReadOnlySpan<int> shape, MlxDType dtype = MlxDType.MLX_FLOAT32)
        => Full(1f, shape, dtype);

    public static unsafe MlxArrayHandle Uniform(float low, float high, ReadOnlySpan<int> shape, MlxDType dtype = MlxDType.MLX_FLOAT32)
    {
        var lowScalar = CreateScalar(low, dtype);
        var highScalar = CreateScalar(high, dtype);
        var key = NextRandomKey();

        try
        {
            fixed (int* shapePtr = shape)
            {
                var status = MlxRandom.Uniform(
                    out var result,
                    lowScalar,
                    highScalar,
                    shapePtr,
                    (nuint)shape.Length,
                    dtype,
                    key,
                    TensorUtilities.DefaultStream());
                TensorUtilities.CheckStatus(status, "random_uniform");

                return result;
            }
        }
        finally
        {
            if (!TensorUtilities.IsNull(lowScalar))
                MlxArray.Free(lowScalar);
            if (!TensorUtilities.IsNull(highScalar))
                MlxArray.Free(highScalar);
            if (!TensorUtilities.IsNull(key))
                MlxArray.Free(key);
        }
    }

    public static unsafe MlxArrayHandle Normal(
        float mean,
        float std,
        ReadOnlySpan<int> shape,
        MlxDType dtype = MlxDType.MLX_FLOAT32)
    {
        var key = NextRandomKey();
        try
        {
            fixed (int* shapePtr = shape)
            {
                var status = MlxRandom.Normal(
                    out var result,
                    shapePtr,
                    (nuint)shape.Length,
                    dtype,
                    mean,
                    std,
                    key,
                    TensorUtilities.DefaultStream());
                TensorUtilities.CheckStatus(status, "random_normal");

                return result;
            }
        }
        finally
        {
            if (!TensorUtilities.IsNull(key))
                MlxArray.Free(key);
        }
    }

    public static unsafe MlxArrayHandle Bernoulli(float probability, ReadOnlySpan<int> shape)
    {
        var prob = Scalar(probability, MlxDType.MLX_FLOAT32);
        var key = NextRandomKey();
        try
        {
            fixed (int* shapePtr = shape)
            {
                var status = MlxRandom.Bernoulli(
                    out var result,
                    prob,
                    shapePtr,
                    (nuint)shape.Length,
                    key,
                    TensorUtilities.DefaultStream());
                TensorUtilities.CheckStatus(status, "random_bernoulli");

                return result;
            }
        }
        finally
        {
            if (!TensorUtilities.IsNull(prob))
                MlxArray.Free(prob);
            if (!TensorUtilities.IsNull(key))
                MlxArray.Free(key);
        }
    }

    public static MlxArrayHandle Arange(int start, int stop, int step = 1, MlxDType dtype = MlxDType.MLX_INT32)
    {
        var status = MlxOps.Arange(out var result, start, stop, step, dtype, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "arange");

        return result;
    }

    public static MlxArrayHandle Arange(float start, float stop, float step = 1f, MlxDType dtype = MlxDType.MLX_FLOAT32)
    {
        var status = MlxOps.Arange(out var result, start, stop, step, dtype, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "arange");

        return result;
    }

    private static MlxArrayHandle NextRandomKey()
    {
        var seed = (ulong)Interlocked.Increment(ref _seedCounter);
        var status = MlxRandom.Key(out var key, seed);
        TensorUtilities.CheckStatus(status, "random_key");

        return key;
    }

    public static MlxArrayHandle ScalarLike(MlxArrayHandle reference, float value)
        => Scalar(value, MlxArray.DType(reference));

    public static MlxArrayHandle OnesLike(MlxArrayHandle reference)
    {
        var shape = reference.Shape();

        return Ones(shape, MlxArray.DType(reference));
    }

    public static MlxArrayHandle ZerosLike(MlxArrayHandle reference)
    {
        var shape = reference.Shape();

        return Zeros(shape, MlxArray.DType(reference));
    }

    private static MlxArrayHandle CreateScalar(float value, MlxDType dtype)
    {
        var scalar = MlxArray.NewFloat32(value);

        if (dtype == MlxDType.MLX_FLOAT32)
            return scalar;

        var status = MlxOps.Astype(out var converted, scalar, dtype, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "astype_scalar");
        MlxArray.Free(scalar);

        return converted;
    }
}