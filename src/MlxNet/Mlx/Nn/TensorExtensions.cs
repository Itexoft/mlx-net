// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Itexoft.Mlx;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Provides convenience extension methods over the low-level MLX bindings.
/// </summary>
public static unsafe class TensorExtensions
{
    /// <summary>
    /// Returns the number of dimensions of the tensor.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Rank(this MlxArrayHandle array)
        => (int)MlxArray.Ndim(array);

    /// <summary>
    /// Returns the shape of the tensor.
    /// </summary>
    public static int[] Shape(this MlxArrayHandle array)
    {
        var rank = array.Rank();
        var pointer = MlxArray.Shape(array);
        var result = new int[rank];
        for (var i = 0; i < rank; i++)
            result[i] = Marshal.ReadInt32((nint)pointer, i * sizeof(int));

        return result;
    }

    /// <summary>
    /// Returns the size of a specified axis.
    /// </summary>
    public static int Dim(this MlxArrayHandle array, int axis)
    {
        var rank = array.Rank();
        var normalized = NormalizeAxis(axis, rank);
        var pointer = MlxArray.Shape(array);

        return Marshal.ReadInt32((nint)pointer, normalized * sizeof(int));
    }

    /// <summary>
    /// Returns the total number of elements.
    /// </summary>
    public static int Size(this MlxArrayHandle array)
        => checked((int)MlxArray.Size(array));

    /// <summary>
    /// Reshapes the tensor without copying.
    /// </summary>
    public static MlxArrayHandle Reshape(this MlxArrayHandle array, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        fixed (int* ptr = shape)
        {
            var status = MlxOps.Reshape(out var result, array, ptr, (nuint)shape.Length, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "reshape");

            return result;
        }
    }

    /// <summary>
    /// Transposes the last two axes by default or two specified axes.
    /// </summary>
    public static MlxArrayHandle Transpose(this MlxArrayHandle array, int axis0 = -2, int axis1 = -1)
    {
        var rank = array.Rank();
        axis0 = NormalizeAxis(axis0, rank);
        axis1 = NormalizeAxis(axis1, rank);

        Span<int> permutation = stackalloc int[rank];
        for (var i = 0; i < rank; i++)
            permutation[i] = i;

        (permutation[axis0], permutation[axis1]) = (permutation[axis1], permutation[axis0]);
        fixed (int* ptr = permutation)
        {
            var status = MlxOps.TransposeAxes(out var result, array, ptr, (nuint)rank, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "transpose");

            return result;
        }
    }

    /// <summary>
    /// Applies a permutation of axes.
    /// </summary>
    public static MlxArrayHandle Transposed(this MlxArrayHandle array, params int[] axes)
    {
        ArgumentNullException.ThrowIfNull(axes);

        if (axes.Length != array.Rank())
            throw new ArgumentException("Number of axes must match tensor rank.", nameof(axes));

        fixed (int* ptr = axes)
        {
            var status = MlxOps.TransposeAxes(out var result, array, ptr, (nuint)axes.Length, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "transpose");

            return result;
        }
    }

    /// <summary>
    /// Swaps two axes.
    /// </summary>
    public static MlxArrayHandle SwappedAxes(this MlxArrayHandle array, int axis0, int axis1)
    {
        var rank = array.Rank();
        axis0 = NormalizeAxis(axis0, rank);
        axis1 = NormalizeAxis(axis1, rank);
        Span<int> permutation = stackalloc int[rank];
        for (var i = 0; i < rank; i++)
            permutation[i] = i;
        (permutation[axis0], permutation[axis1]) = (permutation[axis1], permutation[axis0]);
        fixed (int* ptr = permutation)
        {
            var status = MlxOps.TransposeAxes(out var result, array, ptr, (nuint)rank, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "swap_axes");

            return result;
        }
    }

    /// <summary>
    /// Unsqueezes a dimension of size 1 at the specified axis.
    /// </summary>
    public static MlxArrayHandle ExpandedDimension(this MlxArrayHandle array, int axis)
    {
        var rank = array.Rank();
        axis = NormalizeAxis(axis, rank + 1);

        Span<int> shape = stackalloc int[rank + 1];
        var idx = 0;
        for (var i = 0; i < rank + 1; i++)
            if (i == axis)
                shape[i] = 1;
            else
                shape[i] = array.Dim(idx++);

        fixed (int* ptr = shape)
        {
            var status = MlxOps.Reshape(out var result, array, ptr, (nuint)shape.Length, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "expand_dims");

            return result;
        }
    }

    /// <summary>
    /// Removes singleton dimensions.
    /// </summary>
    public static MlxArrayHandle Squeezed(this MlxArrayHandle array, params int[] axes)
    {
        if (axes is { Length: > 0 })
        {
            fixed (int* ptr = axes)
            {
                var status = MlxOps.SqueezeAxes(out var result, array, ptr, (nuint)axes.Length, TensorUtilities.DefaultStream());
                TensorUtilities.CheckStatus(status, "squeeze_axes");

                return result;
            }
        }
        else
        {
            var status = MlxOps.Squeeze(out var result, array, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "squeeze");

            return result;
        }
    }

    /// <summary>
    /// Broadcasts an array to the specified shape.
    /// </summary>
    public static MlxArrayHandle BroadcastTo(this MlxArrayHandle array, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        fixed (int* ptr = shape)
        {
            var status = MlxOps.BroadcastTo(out var result, array, ptr, (nuint)shape.Length, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "broadcast_to");

            return result;
        }
    }

    /// <summary>
    /// Creates a view into the tensor with the specified shape and strides.
    /// </summary>
    public static MlxArrayHandle AsStrided(
        this MlxArrayHandle array,
        ReadOnlySpan<int> shape,
        ReadOnlySpan<long> strides,
        nuint offset = 0)
    {
        if (shape.Length != strides.Length)
            throw new ArgumentException("Shape and strides must have the same length.");

        fixed (int* shapePtr = shape)
        fixed (long* stridesPtr = strides)
        {
            var status = MlxOps.AsStrided(
                out var result,
                array,
                shapePtr,
                (nuint)shape.Length,
                stridesPtr,
                (nuint)strides.Length,
                offset,
                TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "as_strided");

            return result;
        }
    }

    /// <summary>
    /// Computes element-wise addition.
    /// </summary>
    public static MlxArrayHandle Add(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Add(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "add");

        return result;
    }

    /// <summary>
    /// Computes element-wise subtraction.
    /// </summary>
    public static MlxArrayHandle Subtract(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Subtract(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "subtract");

        return result;
    }

    /// <summary>
    /// Computes element-wise multiplication.
    /// </summary>
    public static MlxArrayHandle Multiply(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Multiply(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "multiply");

        return result;
    }

    /// <summary>
    /// Computes element-wise less-than comparison, yielding a boolean mask.
    /// </summary>
    public static MlxArrayHandle LessThan(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Less(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "less");

        return result;
    }

    /// <summary>
    /// Computes element-wise division.
    /// </summary>
    public static MlxArrayHandle Divide(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Divide(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "divide");

        return result;
    }

    /// <summary>
    /// Computes the element-wise maximum.
    /// </summary>
    public static MlxArrayHandle Maximum(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Maximum(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "maximum");

        return result;
    }

    /// <summary>
    /// Computes the element-wise minimum.
    /// </summary>
    public static MlxArrayHandle Minimum(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Minimum(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "minimum");

        return result;
    }

    /// <summary>
    /// Applies the sigmoid function element-wise.
    /// </summary>
    public static MlxArrayHandle Sigmoid(this MlxArrayHandle array)
    {
        var status = MlxOps.Sigmoid(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "sigmoid");

        return result;
    }

    /// <summary>
    /// Applies the hyperbolic tangent element-wise.
    /// </summary>
    public static MlxArrayHandle Tanh(this MlxArrayHandle array)
    {
        var status = MlxOps.Tanh(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "tanh");

        return result;
    }

    /// <summary>
    /// Performs matrix multiplication.
    /// </summary>
    public static MlxArrayHandle Matmul(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Matmul(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "matmul");

        return result;
    }

    /// <summary>
    /// Computes <c>result = beta * c + alpha * (a @ b)</c>.
    /// </summary>
    public static MlxArrayHandle Addmm(
        this MlxArrayHandle c,
        MlxArrayHandle a,
        MlxArrayHandle b,
        float alpha = 1f,
        float beta = 1f)
    {
        var status = MlxOps.Addmm(out var result, c, a, b, alpha, beta, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "addmm");

        return result;
    }

    /// <summary>
    /// Computes the mean over the specified axis.
    /// </summary>
    public static MlxArrayHandle Mean(this MlxArrayHandle array, int axis, bool keepDims = false)
        => array.Mean([axis], keepDims);

    /// <summary>
    /// Computes the mean over the specified axes.
    /// </summary>
    public static MlxArrayHandle Mean(this MlxArrayHandle array, ReadOnlySpan<int> axes, bool keepDims = false)
    {
        if (axes.Length == 0)
            return array;

        if (axes.Length == 1)
        {
            var axis = NormalizeAxis(axes[0], array.Rank());
            var statusSingle = MlxOps.MeanAxis(out var single, array, axis, keepDims, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(statusSingle, "mean_axis");

            return single;
        }

        var normalized = NormalizeAxes(axes, array.Rank());
        fixed (int* axesPtr = normalized)
        {
            var status = MlxOps.MeanAxes(
                out var result,
                array,
                axesPtr,
                (nuint)normalized.Length,
                keepDims,
                TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "mean_axes");

            return result;
        }
    }

    /// <summary>
    /// Computes the maximum over the specified axis.
    /// </summary>
    public static MlxArrayHandle Max(this MlxArrayHandle array, int axis, bool keepDims = false)
        => array.Max([axis], keepDims);

    /// <summary>
    /// Computes the maximum over the specified axes.
    /// </summary>
    public static MlxArrayHandle Max(this MlxArrayHandle array, ReadOnlySpan<int> axes, bool keepDims = false)
    {
        if (axes.Length == 0)
            return array;

        if (axes.Length == 1)
        {
            var axis = NormalizeAxis(axes[0], array.Rank());
            var statusSingle = MlxOps.MaxAxis(out var single, array, axis, keepDims, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(statusSingle, "max_axis");

            return single;
        }

        var normalized = NormalizeAxes(axes, array.Rank());
        fixed (int* axesPtr = normalized)
        {
            var status = MlxOps.MaxAxes(
                out var result,
                array,
                axesPtr,
                (nuint)normalized.Length,
                keepDims,
                TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "max_axes");

            return result;
        }
    }

    /// <summary>
    /// Computes the variance over the specified axis.
    /// </summary>
    public static MlxArrayHandle Variance(this MlxArrayHandle array, int axis, bool keepDims = false)
    {
        var mean = array.Mean(axis, true);
        var centered = array.Subtract(mean);
        var squared = centered.Multiply(centered);
        var result = squared.Mean(axis, keepDims);
        MlxArray.Free(centered);
        MlxArray.Free(squared);
        MlxArray.Free(mean);

        return result;
    }

    /// <summary>
    /// Applies the softmax function along an axis.
    /// </summary>
    public static MlxArrayHandle Softmax(this MlxArrayHandle array, int axis)
    {
        axis = NormalizeAxis(axis, array.Rank());
        var status = MlxOps.SoftmaxAxis(out var result, array, axis, false, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "softmax");

        return result;
    }

    /// <summary>
    /// Computes the natural exponential of each element.
    /// </summary>
    public static MlxArrayHandle Exp(this MlxArrayHandle array)
    {
        var status = MlxOps.Exp(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "exp");

        return result;
    }

    /// <summary>
    /// Computes the natural logarithm of each element.
    /// </summary>
    public static MlxArrayHandle Log(this MlxArrayHandle array)
    {
        var status = MlxOps.Log(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "log");

        return result;
    }

    /// <summary>
    /// Computes the error function element-wise.
    /// </summary>
    public static MlxArrayHandle Erf(this MlxArrayHandle array)
    {
        var status = MlxOps.Erf(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "erf");

        return result;
    }

    /// <summary>
    /// Computes element-wise absolute value.
    /// </summary>
    public static MlxArrayHandle Abs(this MlxArrayHandle array)
    {
        var status = MlxOps.Abs(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "abs");

        return result;
    }

    /// <summary>
    /// Casts the tensor to a new data type.
    /// </summary>
    public static MlxArrayHandle AsType(this MlxArrayHandle array, MlxDType dtype)
    {
        var status = MlxOps.Astype(out var result, array, dtype, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "astype");

        return result;
    }

    internal static int NormalizeAxis(int axis, int rank)
    {
        if (axis < 0)
            axis += rank;

        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis must be in range [-{rank}, {rank - 1}].");

        return axis;
    }

    internal static int[] NormalizeAxes(ReadOnlySpan<int> axes, int rank)
    {
        var normalized = new int[axes.Length];
        for (var i = 0; i < axes.Length; i++)
            normalized[i] = NormalizeAxis(axes[i], rank);
        Array.Sort(normalized);

        return normalized;
    }

    /// <summary>
    /// Computes the sum of all elements.
    /// </summary>
    public static MlxArrayHandle Sum(this MlxArrayHandle array, bool keepDims = false)
    {
        var status = MlxOps.Sum(out var result, array, keepDims, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "sum");

        return result;
    }

    /// <summary>
    /// Computes the sum over the specified axes.
    /// </summary>
    public static MlxArrayHandle Sum(this MlxArrayHandle array, ReadOnlySpan<int> axes, bool keepDims = false)
    {
        if (axes.Length == 0)
            return array;

        if (axes.Length == 1)
        {
            var axis = NormalizeAxis(axes[0], array.Rank());
            var statusSingle = MlxOps.SumAxis(out var single, array, axis, keepDims, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(statusSingle, "sum_axis");

            return single;
        }

        var normalized = NormalizeAxes(axes, array.Rank());
        fixed (int* axesPtr = normalized)
        {
            var status = MlxOps.SumAxes(
                out var result,
                array,
                axesPtr,
                (nuint)normalized.Length,
                keepDims,
                TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "sum_axes");

            return result;
        }
    }

    /// <summary>
    /// Computes the sum along a given axis.
    /// </summary>
    public static MlxArrayHandle Sum(this MlxArrayHandle array, int axis, bool keepDims = false)
        => array.Sum([axis], keepDims);

    /// <summary>
    /// Computes the log-sum-exp along the specified axis.
    /// </summary>
    public static MlxArrayHandle LogSumExp(this MlxArrayHandle array, int axis, bool keepDims = false)
    {
        axis = NormalizeAxis(axis, array.Rank());
        var status = MlxOps.LogsumexpAxis(out var result, array, axis, keepDims, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "logsumexp_axis");

        return result;
    }

    /// <summary>
    /// Computes the log-sum-exp over the specified axes.
    /// </summary>
    public static MlxArrayHandle LogSumExp(this MlxArrayHandle array, ReadOnlySpan<int> axes, bool keepDims = false)
    {
        if (axes.Length == 0)
            return array;

        if (axes.Length == 1)
            return array.LogSumExp(axes[0], keepDims);

        var normalized = NormalizeAxes(axes, array.Rank());
        fixed (int* axesPtr = normalized)
        {
            var status = MlxOps.LogsumexpAxes(
                out var result,
                array,
                axesPtr,
                (nuint)normalized.Length,
                keepDims,
                TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "logsumexp_axes");

            return result;
        }
    }

    /// <summary>
    /// Computes element-wise log-add-exp of two arrays.
    /// </summary>
    public static MlxArrayHandle LogAddExp(this MlxArrayHandle lhs, MlxArrayHandle rhs)
    {
        var status = MlxOps.Logaddexp(out var result, lhs, rhs, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "logaddexp");

        return result;
    }

    /// <summary>
    /// Applies clipping to the tensor values.
    /// </summary>
    public static MlxArrayHandle Clip(this MlxArrayHandle array, float? min = null, float? max = null)
    {
        var dtype = MlxArray.DType(array);
        var minScalar = TensorFactory.Scalar(min ?? float.NegativeInfinity, dtype);
        var maxScalar = TensorFactory.Scalar(max ?? float.PositiveInfinity, dtype);
        try
        {
            var status = MlxOps.Clip(out var result, array, minScalar, maxScalar, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "clip");

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(minScalar))
                MlxArray.Free(minScalar);
            if (!TensorUtilities.IsNull(maxScalar))
                MlxArray.Free(maxScalar);
        }
    }

    /// <summary>
    /// Returns the floor of each element.
    /// </summary>
    public static MlxArrayHandle Floor(this MlxArrayHandle array)
    {
        var status = MlxOps.Floor(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "floor");

        return result;
    }

    /// <summary>
    /// Returns the ceiling of each element.
    /// </summary>
    public static MlxArrayHandle Ceil(this MlxArrayHandle array)
    {
        var status = MlxOps.Ceil(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "ceil");

        return result;
    }

    /// <summary>
    /// Rounds each element to the nearest integer.
    /// </summary>
    public static MlxArrayHandle Round(this MlxArrayHandle array)
    {
        const int decimals = 0;
        var status = MlxOps.Round(out var result, array, decimals, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "round");

        return result;
    }

    /// <summary>
    /// Computes the element-wise square.
    /// </summary>
    public static MlxArrayHandle Square(this MlxArrayHandle array)
    {
        var status = MlxOps.Square(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "square");

        return result;
    }

    /// <summary>
    /// Computes the element-wise square root.
    /// </summary>
    public static MlxArrayHandle Sqrt(this MlxArrayHandle array)
    {
        var status = MlxOps.Sqrt(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "sqrt");

        return result;
    }

    /// <summary>
    /// Raises each element to the given scalar power.
    /// </summary>
    public static MlxArrayHandle Pow(this MlxArrayHandle array, float exponent)
    {
        var scalar = TensorFactory.Scalar(exponent, MlxArray.DType(array));
        try
        {
            var status = MlxOps.Power(out var result, array, scalar, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "power");

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(scalar))
                MlxArray.Free(scalar);
        }
    }

    /// <summary>
    /// Negates each element of the tensor.
    /// </summary>
    public static MlxArrayHandle Negative(this MlxArrayHandle array)
    {
        var scalar = TensorFactory.ScalarLike(array, -1f);
        try
        {
            var status = MlxOps.Multiply(out var result, array, scalar, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "negative");

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(scalar))
                MlxArray.Free(scalar);
        }
    }

    /// <summary>
    /// Creates a copy of the tensor.
    /// </summary>
    public static MlxArrayHandle Copy(this MlxArrayHandle array)
    {
        var status = MlxOps.Copy(out var result, array, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "copy");

        return result;
    }

    /// <summary>
    /// Selects elements from two tensors based on a boolean condition.
    /// </summary>
    public static MlxArrayHandle Where(this MlxArrayHandle condition, MlxArrayHandle whenTrue, MlxArrayHandle whenFalse)
    {
        var status = MlxOps.Where(out var result, condition, whenTrue, whenFalse, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "where");

        return result;
    }

    /// <summary>
    /// Takes elements along the specified axis using the provided indices.
    /// </summary>
    public static MlxArrayHandle TakeAlong(this MlxArrayHandle array, MlxArrayHandle indices, int axis)
    {
        axis = NormalizeAxis(axis, array.Rank());
        var status = MlxOps.TakeAlongAxis(out var result, array, indices, axis, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "take_along_axis");

        return result;
    }

    /// <summary>
    /// Creates a slice along a specific axis with the provided range and stride.
    /// </summary>
    public static MlxArrayHandle Slice(this MlxArrayHandle array, int axis, int start, int? stop = null, int stride = 1)
    {
        var rank = array.Rank();
        var normalizedAxis = NormalizeAxis(axis, rank);
        var shape = array.Shape();

        var sliceStop = stop ?? shape[normalizedAxis];
        if (sliceStop < 0)
            sliceStop = shape[normalizedAxis] + sliceStop;

        var startBuffer = stackalloc int[rank];
        var stopBuffer = stackalloc int[rank];
        var strideBuffer = stackalloc int[rank];

        for (var i = 0; i < rank; i++)
        {
            startBuffer[i] = 0;
            stopBuffer[i] = shape[i];
            strideBuffer[i] = 1;
        }

        startBuffer[normalizedAxis] = start;
        stopBuffer[normalizedAxis] = sliceStop;
        strideBuffer[normalizedAxis] = stride;

        var status = MlxOps.Slice(
            out var result,
            array,
            startBuffer,
            (nuint)rank,
            stopBuffer,
            (nuint)rank,
            strideBuffer,
            (nuint)rank,
            TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "slice");

        return result;
    }

    /// <summary>
    /// Splits an array into a specified number of equal parts along an axis.
    /// </summary>
    public static MlxArrayHandle[] Split(this MlxArrayHandle array, int parts, int axis)
    {
        axis = NormalizeAxis(axis, array.Rank());
        var status = MlxOps.Split(out var vector, array, parts, axis, TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "split");

        return TensorVectorUtilities.Consume(vector);
    }

    /// <summary>
    /// Stacks a collection of tensors along a new axis.
    /// </summary>
    public static MlxArrayHandle Stack(this IReadOnlyList<MlxArrayHandle> arrays, int axis = 0)
    {
        if (arrays.Count == 0)
            throw new ArgumentException("At least one tensor is required to stack.", nameof(arrays));

        var rank = arrays[0].Rank();
        var normalizedAxis = axis < 0 ? axis + rank + 1 : axis;

        if (normalizedAxis < 0 || normalizedAxis > rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        var temp = new MlxArrayHandle[arrays.Count];
        for (var i = 0; i < arrays.Count; i++)
            temp[i] = arrays[i];

        var vector = TensorVectorUtilities.Create(temp);
        try
        {
            var status = MlxOps.StackAxis(out var result, vector, normalizedAxis, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "stack_axis");

            return result;
        }
        finally
        {
            MlxVector.ArrayFree(vector);
        }
    }

    /// <summary>
    /// Adds a scalar value to the tensor.
    /// </summary>
    public static MlxArrayHandle AddScalar(this MlxArrayHandle array, float value)
    {
        var scalar = TensorFactory.ScalarLike(array, value);
        try
        {
            var status = MlxOps.Add(out var result, array, scalar, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "add_scalar");

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(scalar))
                MlxArray.Free(scalar);
        }
    }

    /// <summary>
    /// Multiplies the tensor by a scalar value.
    /// </summary>
    public static MlxArrayHandle MultiplyScalar(this MlxArrayHandle array, float value)
    {
        var scalar = TensorFactory.ScalarLike(array, value);
        try
        {
            var status = MlxOps.Multiply(out var result, array, scalar, TensorUtilities.DefaultStream());
            TensorUtilities.CheckStatus(status, "multiply_scalar");

            return result;
        }
        finally
        {
            if (!TensorUtilities.IsNull(scalar))
                MlxArray.Free(scalar);
        }
    }
}
