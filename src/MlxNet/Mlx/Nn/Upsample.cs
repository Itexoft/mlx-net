// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using Itexoft.Mlx;

namespace Itexoft.Mlx.Nn;

public enum UpsampleMode
{
    Nearest,
    Linear,
    Cubic
}

/// <summary>
/// Upsamples spatial dimensions of the input tensor.
/// </summary>
public sealed class Upsample(FloatOrArray scaleFactor, UpsampleMode mode = UpsampleMode.Nearest, bool alignCorners = false)
    : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var rank = input.Rank();

        if (rank < 3)
            throw new ArgumentException("Upsample expects input tensors with at least one spatial dimension (rank >= 3).", nameof(input));

        var spatialDimensions = rank - 2;
        var scales = scaleFactor.AsArray(spatialDimensions);

        return mode switch
        {
            UpsampleMode.Nearest => UpsampleNearest(input, scales),
            UpsampleMode.Linear => this.UpsampleInterpolate(input, scales, false),
            UpsampleMode.Cubic => this.UpsampleInterpolate(input, scales, true),
            _ => throw new ArgumentOutOfRangeException()
        };
    }

    private static MlxArrayHandle UpsampleNearest(MlxArrayHandle input, float[] scales)
    {
        var originalShape = input.Shape();
        var spatialDims = originalShape.Length - 2;

        var integerScales = new int[spatialDims];
        var useFastPath = true;
        for (var i = 0; i < spatialDims; i++)
        {
            var rounded = MathF.Round(scales[i]);
            if (MathF.Abs(scales[i] - rounded) > 1e-6f || rounded <= 0f)
            {
                useFastPath = false;

                break;
            }

            integerScales[i] = (int)rounded;
        }

        if (useFastPath)
            return UpsampleNearestInteger(input, scales, integerScales, originalShape);

        return UpsampleNearestGeneral(input, scales, originalShape);
    }

    private static MlxArrayHandle UpsampleNearestInteger(MlxArrayHandle input, float[] scales, int[] integerScales, int[] originalShape)
    {
        var spatialDims = integerScales.Length;
        var shapeList = new List<int>(originalShape);

        for (var d = 0; d < spatialDims; d++)
            shapeList.Insert(2 + 2 * d, 1);

        var reshaped = input.Reshape(shapeList.ToArray());

        for (var d = 0; d < spatialDims; d++)
            shapeList[2 + 2 * d] = integerScales[d];

        var broadcasted = reshaped.BroadcastTo(shapeList.ToArray());
        MlxArray.Free(reshaped);

        var finalShape = new int[originalShape.Length];
        finalShape[0] = originalShape[0];
        for (var d = 0; d < spatialDims; d++)
            finalShape[1 + d] = originalShape[1 + d] * integerScales[d];
        finalShape[^1] = originalShape[^1];

        var result = broadcasted.Reshape(finalShape);
        MlxArray.Free(broadcasted);

        return result;
    }

    private static MlxArrayHandle UpsampleNearestGeneral(MlxArrayHandle input, float[] scales, int[] originalShape)
    {
        var spatialDims = scales.Length;
        var result = input;
        var ownsResult = false;

        for (var dim = 0; dim < spatialDims; dim++)
        {
            var n = originalShape[1 + dim];
            var indices = CreateNearestIndices(n, scales[dim], dim, spatialDims, out var outputLength);

            var targetShape = result.Shape();
            targetShape[1 + dim] = outputLength;
            var broadcast = indices.BroadcastTo(targetShape);
            var gathered = result.TakeAlong(broadcast, 1 + dim);
            MlxArray.Free(broadcast);
            MlxArray.Free(indices);

            if (ownsResult)
                MlxArray.Free(result);

            result = gathered;
            ownsResult = true;
        }

        return result;
    }

    private MlxArrayHandle UpsampleInterpolate(MlxArrayHandle input, float[] scales, bool isCubic)
    {
        var dims = scales.Length;
        var rank = dims + 2;
        var originalShape = input.Shape();

        var perDimension = new IndexWeight[dims][];
        for (var dim = 0; dim < dims; dim++)
        {
            var n = originalShape[1 + dim];
            perDimension[dim] = isCubic
                ? BuildCubicIndexWeights(n, scales[dim], dim, dims, alignCorners)
                : BuildLinearIndexWeights(n, scales[dim], dim, dims, alignCorners);
        }

        var combinations = CartesianProduct(perDimension);
        MlxArrayHandle? accumulator = null;

        foreach (var combo in combinations)
        {
            var sample = input;
            var sampleOwned = false;

            for (var dim = 0; dim < dims; dim++)
            {
                var axis = 1 + dim;
                var targetShape = sample.Shape();
                targetShape[axis] = combo[dim].OutputLength;

                var broadcastIndices = combo[dim].Indices.BroadcastTo(targetShape);
                var gathered = sample.TakeAlong(broadcastIndices, axis);
                MlxArray.Free(broadcastIndices);

                if (sampleOwned)
                    MlxArray.Free(sample);

                sample = gathered;
                sampleOwned = true;
            }

            var combinedWeight = CombineWeights(combo);
            var sampleShape = sample.Shape();
            var broadcastWeight = combinedWeight.BroadcastTo(sampleShape);
            MlxArray.Free(combinedWeight);

            var weightedSample = sample.Multiply(broadcastWeight);
            MlxArray.Free(broadcastWeight);
            if (sampleOwned)
                MlxArray.Free(sample);

            if (accumulator is null)
            {
                accumulator = weightedSample;
            }
            else
            {
                var sum = accumulator.Value.Add(weightedSample);
                MlxArray.Free(accumulator.Value);
                MlxArray.Free(weightedSample);
                accumulator = sum;
            }
        }

        DisposeIndexWeights(perDimension);

        return accumulator ?? input;
    }

    private static MlxArrayHandle CombineWeights(IReadOnlyList<IndexWeight> weights)
    {
        var combined = weights[0].Weight.Copy();
        for (var i = 1; i < weights.Count; i++)
        {
            var multiplied = combined.Multiply(weights[i].Weight);
            MlxArray.Free(combined);
            combined = multiplied;
        }

        return combined;
    }

    private static IndexWeight[] BuildLinearIndexWeights(int dimension, float scale, int dimIndex, int dims, bool alignCorners)
    {
        var scaled = CreateScaledIndices(dimension, scale, alignCorners, dimIndex, dims, out var outputLength);
        var clipped = scaled.Clip(0f, dimension - 1f);

        var indicesLeftFloat = clipped.Floor();
        var indicesRightFloat = clipped.Ceil();

        var diff = clipped.Subtract(indicesLeftFloat);
        var inv = TensorFactory.ScalarLike(diff, 1f).Subtract(diff);

        var indicesLeft = indicesLeftFloat.AsType(MlxDType.MLX_INT32);
        var indicesRight = indicesRightFloat.AsType(MlxDType.MLX_INT32);

        MlxArray.Free(indicesLeftFloat);
        MlxArray.Free(indicesRightFloat);
        MlxArray.Free(clipped);

        return [new(indicesLeft, inv, outputLength), new(indicesRight, diff, outputLength)];
    }

    private static IndexWeight[] BuildCubicIndexWeights(int dimension, float scale, int dimIndex, int dims, bool alignCorners)
    {
        var scaled = CreateScaledIndices(dimension, scale, alignCorners, dimIndex, dims, out var outputLength);

        var indicesL1 = scaled.Floor();
        var indicesR1 = scaled.AddScalar(1f).Floor();
        var indicesL2 = indicesL1.AddScalar(-1f);
        var indicesR2 = indicesR1.AddScalar(1f);

        var weightL1 = ComputeCubicWeight(scaled, indicesL1, 1);
        var weightR1 = ComputeCubicWeight(scaled, indicesR1, 1);
        var weightL2 = ComputeCubicWeight(scaled, indicesL2, 2);
        var weightR2 = ComputeCubicWeight(scaled, indicesR2, 2);

        var clippedL1 = indicesL1.Clip(0f, dimension - 1f).AsType(MlxDType.MLX_INT32);
        var clippedR1 = indicesR1.Clip(0f, dimension - 1f).AsType(MlxDType.MLX_INT32);
        var clippedL2 = indicesL2.Clip(0f, dimension - 1f).AsType(MlxDType.MLX_INT32);
        var clippedR2 = indicesR2.Clip(0f, dimension - 1f).AsType(MlxDType.MLX_INT32);

        MlxArray.Free(indicesL1);
        MlxArray.Free(indicesR1);
        MlxArray.Free(indicesL2);
        MlxArray.Free(indicesR2);
        MlxArray.Free(scaled);

        return
        [
            new(clippedL1, weightL1, outputLength),
            new(clippedR1, weightR1, outputLength),
            new(clippedL2, weightL2, outputLength),
            new(clippedR2, weightR2, outputLength)
        ];
    }

    private static MlxArrayHandle ComputeCubicWeight(MlxArrayHandle indices, MlxArrayHandle grid, int distance)
    {
        var diff = indices.Subtract(grid);
        var abs = diff.Abs();
        MlxArray.Free(diff);

        const float a = -0.75f;

        MlxArrayHandle result;
        if (distance == 1)
        {
            var term1 = abs.MultiplyScalar(a + 2f);
            var term2 = term1.AddScalar(-(a + 3f));
            MlxArray.Free(term1);

            var prod1 = term2.Multiply(abs);
            var prod2 = prod1.Multiply(abs);
            MlxArray.Free(prod1);
            MlxArray.Free(term2);

            result = prod2.AddScalar(1f);
            MlxArray.Free(prod2);
        }
        else
        {
            var minusFive = abs.AddScalar(-5f);
            var prod1 = minusFive.Multiply(abs);
            MlxArray.Free(minusFive);

            var plusEight = prod1.AddScalar(8f);
            MlxArray.Free(prod1);

            var prod2 = plusEight.Multiply(abs);
            MlxArray.Free(plusEight);

            var minusFour = prod2.AddScalar(-4f);
            MlxArray.Free(prod2);

            result = minusFour.MultiplyScalar(a);
            MlxArray.Free(minusFour);
        }

        MlxArray.Free(abs);

        return result;
    }

    private static List<IndexWeight[]> CartesianProduct(IndexWeight[][] perDimension)
    {
        var result = new List<IndexWeight[]>();

        if (perDimension.Length == 0)
            return result;

        var stack = new IndexWeight[perDimension.Length];

        void Recurse(int depth)
        {
            if (depth == perDimension.Length)
            {
                var combination = new IndexWeight[stack.Length];
                Array.Copy(stack, combination, stack.Length);
                result.Add(combination);

                return;
            }

            foreach (var element in perDimension[depth])
            {
                stack[depth] = element;
                Recurse(depth + 1);
            }
        }

        Recurse(0);

        return result;
    }

    private static void DisposeIndexWeights(IndexWeight[][] perDimension)
    {
        foreach (var array in perDimension)
        foreach (var item in array)
        {
            if (!TensorUtilities.IsNull(item.Indices))
                MlxArray.Free(item.Indices);

            if (!TensorUtilities.IsNull(item.Weight))
                MlxArray.Free(item.Weight);
        }
    }

    private static MlxArrayHandle CreateNearestIndices(int dimension, float scale, int dimIndex, int dims, out int outputLength)
    {
        var m = Math.Max(1, (int)MathF.Round(scale * dimension));
        outputLength = m;

        var indices = TensorFactory.Arange(0f, m, 1f, MlxDType.MLX_FLOAT32);
        if (m > dimension)
        {
            var shifted = indices.AddScalar(0.5f);
            var scaled = shifted.MultiplyScalar(dimension / (float)m);
            MlxArray.Free(shifted);
            var offset = scaled.AddScalar(-0.5f);
            MlxArray.Free(scaled);
            indices = offset.Round();
        }
        else
        {
            var scaled = indices.MultiplyScalar(dimension / (float)m);
            MlxArray.Free(indices);
            indices = scaled;
        }

        var clipped = indices.Clip(0f, dimension - 1f);
        MlxArray.Free(indices);

        var intIndices = clipped.AsType(MlxDType.MLX_INT32);
        MlxArray.Free(clipped);

        return ReshapeIndices(intIndices, dims, dimIndex, outputLength);
    }

    private static MlxArrayHandle CreateScaledIndices(
        int dimension,
        float scale,
        bool alignCorners,
        int dimIndex,
        int dims,
        out int outputLength)
    {
        var m = Math.Max(1, (int)MathF.Round(scale * dimension));
        outputLength = m;

        var indices = TensorFactory.Arange(0f, m, 1f, MlxDType.MLX_FLOAT32);

        if (alignCorners)
        {
            if (m == 1)
            {
                var zeros = indices.MultiplyScalar(0f);
                MlxArray.Free(indices);
                indices = zeros;
            }
            else
            {
                var factor = (dimension - 1f) / (m - 1f);
                var scaled = indices.MultiplyScalar(factor);
                MlxArray.Free(indices);
                indices = scaled;
            }
        }
        else
        {
            var step = 1f / scale;
            var scaled = indices.MultiplyScalar(step);
            MlxArray.Free(indices);

            var start = ((m - 1f) * step - dimension + 1f) / 2f;
            var adjusted = scaled.AddScalar(-start);
            MlxArray.Free(scaled);
            indices = adjusted;
        }

        return ReshapeFloatIndices(indices, dims, dimIndex, outputLength);
    }

    private static MlxArrayHandle ReshapeIndices(MlxArrayHandle indices, int dims, int dimIndex, int length)
    {
        var rank = dims + 2;
        var reshape = new int[rank];
        for (var i = 0; i < rank; i++)
            reshape[i] = 1;

        reshape[1 + dimIndex] = length;
        reshape[rank - 1] = 1;

        var reshaped = indices.Reshape(reshape);
        MlxArray.Free(indices);

        return reshaped;
    }

    private static MlxArrayHandle ReshapeFloatIndices(MlxArrayHandle indices, int dims, int dimIndex, int length)
    {
        var rank = dims + 2;
        var reshape = new int[rank];
        for (var i = 0; i < rank; i++)
            reshape[i] = 1;

        reshape[1 + dimIndex] = length;
        reshape[rank - 1] = 1;

        var reshaped = indices.Reshape(reshape);
        MlxArray.Free(indices);

        return reshaped;
    }

    private readonly struct IndexWeight(MlxArrayHandle indices, MlxArrayHandle weight, int outputLength)
    {
        public MlxArrayHandle Indices { get; } = indices;
        public MlxArrayHandle Weight { get; } = weight;
        public int OutputLength { get; } = outputLength;
    }
}