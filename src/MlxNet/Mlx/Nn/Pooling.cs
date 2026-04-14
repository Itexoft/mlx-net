// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

public delegate MlxArrayHandle PoolingOperation(MlxArrayHandle array, ReadOnlySpan<int> axes);

/// <summary>
/// Base implementation for sliding-window pooling layers working on last-channel tensors.
/// </summary>
public abstract class Pool : Module, IUnaryLayer
{
    private readonly int[] kernelSize;
    private readonly PoolingOperation poolingOp;
    private readonly int[] stride;

    protected Pool(int[] kernelSize, int[] stride, PoolingOperation poolingOp)
    {
        ArgumentNullException.ThrowIfNull(kernelSize);
        ArgumentNullException.ThrowIfNull(stride);
        ArgumentNullException.ThrowIfNull(poolingOp);

        if (kernelSize.Length == 0)
            throw new ArgumentException("Kernel size must contain at least one dimension.", nameof(kernelSize));

        if (kernelSize.Length != stride.Length)
            throw new ArgumentException("Kernel size and stride dimensions must match.", nameof(stride));

        this.kernelSize = (int[])kernelSize.Clone();
        this.stride = (int[])stride.Clone();
        this.poolingOp = poolingOp;
    }

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var shape = input.ShapeSpan();

        if (shape.Length < this.kernelSize.Length + 2)
            throw new ArgumentException("Input rank is insufficient for pooling operation.", nameof(input));

        var bufferLength = 1 + this.kernelSize.Length * 2 + 1;
        var kernelRank = this.kernelSize.Length;
        Span<int> finalShape = stackalloc int[bufferLength];
        Span<long> finalStrides = stackalloc long[bufferLength];
        Span<int> axes = stackalloc int[kernelRank];

        this.BuildFinalShape(shape, finalShape);
        this.BuildFinalStrides(shape, finalStrides);
        BuildReductionAxes(kernelRank, axes);

        var strided = input.AsStrided(finalShape, finalStrides);
        var pooled = this.poolingOp(strided, axes);
        MlxArray.Free(strided);

        return pooled;
    }

    private void BuildFinalShape(ReadOnlySpan<int> originalShape, Span<int> destination)
    {
        var kernelRank = this.kernelSize.Length;
        destination[0] = originalShape[0];

        for (var i = 0; i < kernelRank; i++)
        {
            var size = originalShape[i + 1];
            var window = this.kernelSize[i];
            var stride1 = this.stride[i];
            var output = (size - window) / stride1 + 1;

            destination[1 + i] = output;
            destination[1 + kernelRank + i] = window;
        }

        destination[^1] = originalShape[^1];
    }

    private void BuildFinalStrides(ReadOnlySpan<int> originalShape, Span<long> destination)
    {
        var kernelRank = this.kernelSize.Length;
        Span<long> contiguous = stackalloc long[originalShape.Length];

        ComputeContiguousStrides(originalShape, contiguous);

        destination[0] = contiguous[0];

        for (var i = 0; i < kernelRank; i++)
        {
            destination[1 + i] = contiguous[1 + i] * this.stride[i];
            destination[1 + kernelRank + i] = contiguous[1 + i];
        }

        destination[^1] = 1;
    }

    private static void ComputeContiguousStrides(ReadOnlySpan<int> shape, Span<long> destination)
    {
        destination[^1] = 1;

        for (var i = shape.Length - 2; i >= 0; i--)
            destination[i] = destination[i + 1] * shape[i + 1];
    }

    private static void BuildReductionAxes(int kernelRank, Span<int> axes)
    {
        for (var i = 0; i < kernelRank; i++)
            axes[i] = 1 + kernelRank + i;
    }
}

/// <summary>
/// 1D max pooling layer operating on NLC tensors.
/// </summary>
public sealed class MaxPool1D(int kernelSize, int stride) : Pool([kernelSize], [stride], (array, axes) => array.Max(axes, false));

/// <summary>
/// 2D max pooling layer operating on NHWC tensors.
/// </summary>
public sealed class MaxPool2D(IntPair kernelSize, IntPair stride) : Pool(kernelSize.Values, stride.Values, (array, axes) => array.Max(axes, false));

/// <summary>
/// 1D average pooling layer operating on NLC tensors.
/// </summary>
public sealed class AvgPool1D(int kernelSize, int stride) : Pool([kernelSize], [stride], (array, axes) => array.Mean(axes, false));

/// <summary>
/// 2D average pooling layer operating on NHWC tensors.
/// </summary>
public sealed class AvgPool2D(IntPair kernelSize, IntPair stride) : Pool(kernelSize.Values, stride.Values, (array, axes) => array.Mean(axes, false));
