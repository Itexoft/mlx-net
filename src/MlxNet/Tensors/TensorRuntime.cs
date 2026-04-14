// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;

namespace Itexoft.Tensors;

internal enum ReductionKind
{
    Sum,
    Mean,
}

internal enum UnaryTensorOp
{
    Abs,
    Exp,
    Log,
    LogicalNot,
    Negate,
    Sqrt,
    TransposeLastTwo,
}

internal enum BinaryTensorOp
{
    Add,
    Divide,
    LogicalAnd,
    LogicalOr,
    Matmul,
    Maximum,
    Multiply,
    Subtract,
}

internal static unsafe class TensorRuntime
{
    internal static MlxStreamHandle DefaultStream() => TensorUtilities.DefaultStream();

    internal static int Rank(MlxArrayHandle handle) => checked((int)MlxArray.Ndim(handle));

    internal static int Dim(MlxArrayHandle handle, int axis)
    {
        var rank = Rank(handle);
        axis = NormalizeAxis(axis, rank);

        return MlxArray.Dim(handle, axis);
    }

    internal static int Size(MlxArrayHandle handle) => checked((int)MlxArray.Size(handle));

    internal static void Dispose(MlxArrayHandle handle)
    {
        if (handle.ctx != 0)
            MlxArray.Free(handle);
    }

    internal static MlxArrayHandle Create(ReadOnlySpan<int> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxInt32);

    internal static MlxArrayHandle Create(ReadOnlySpan<long> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxInt64);

    internal static MlxArrayHandle Create(ReadOnlySpan<float> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxFloat32);

    internal static MlxArrayHandle Create(ReadOnlySpan<double> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxFloat64);

    internal static MlxArrayHandle Create(ReadOnlySpan<Half> values, ReadOnlySpan<int> shape) => CreateNewData(
        MemoryMarshal.Cast<Half, ushort>(values),
        shape,
        MlxDType.MlxFloat16);

    internal static MlxArrayHandle Create(ReadOnlySpan<bool> values, ReadOnlySpan<int> shape) =>
        CreateNewData(MemoryMarshal.AsBytes(values), shape, MlxDType.MlxBool);

    internal static MlxArrayHandle Cast(MlxArrayHandle handle, MlxDType dtype)
    {
        CheckStatus(MlxOps.Astype(out var result, handle, dtype, DefaultStream()), "astype");

        return result;
    }

    internal static MlxArrayHandle Unary(MlxArrayHandle handle, UnaryTensorOp op) => ApplyUnary(handle, op);

    internal static MlxArrayHandle Binary(MlxArrayHandle left, MlxArrayHandle right, BinaryTensorOp op) => ApplyBinary(left, right, op);

    internal static MlxArrayHandle ScalarMaximum(MlxArrayHandle handle, int value) =>
        ApplyBinary(handle, CreateScalar(value), BinaryTensorOp.Maximum, true);

    internal static MlxArrayHandle ScalarMaximum(MlxArrayHandle handle, long value) =>
        ApplyBinary(handle, CreateScalar(value), BinaryTensorOp.Maximum, true);

    internal static MlxArrayHandle ScalarMaximum(MlxArrayHandle handle, float value) =>
        ApplyBinary(handle, CreateScalar(value), BinaryTensorOp.Maximum, true);

    internal static MlxArrayHandle ScalarMaximum(MlxArrayHandle handle, double value) =>
        ApplyBinary(handle, CreateScalar(value), BinaryTensorOp.Maximum, true);

    internal static MlxArrayHandle ScalarMaximum(MlxArrayHandle handle, Half value) =>
        ApplyBinary(handle, CreateScalar(value), BinaryTensorOp.Maximum, true);

    internal static MlxArrayHandle Reduction(MlxArrayHandle handle, Index axis, ReductionKind kind, bool keepDims)
    {
        var rank = Rank(handle);
        var normalizedAxis = ResolveIndex(axis, rank);

        return ApplyReduction(handle, [normalizedAxis], kind, keepDims);
    }

    internal static MlxArrayHandle Reduction(MlxArrayHandle handle, Range axes, ReductionKind kind, bool keepDims)
    {
        var rank = Rank(handle);
        Span<int> buffer = stackalloc int[rank];
        var resolvedAxes = ResolveRange(axes, rank, buffer);

        return ApplyReduction(handle, resolvedAxes, kind, keepDims);
    }

    internal static MlxArrayHandle Softmax(MlxArrayHandle handle, Index axis)
    {
        var rank = Rank(handle);
        var normalizedAxis = ResolveIndex(axis, rank);
        CheckStatus(MlxOps.SoftmaxAxis(out var result, handle, normalizedAxis, false, DefaultStream()), "softmax_axis");

        return result;
    }

    internal static MlxArrayHandle Slice(MlxArrayHandle handle, ReadOnlySpan<AxisSelector> selectors) => ApplySlice(handle, selectors);

    internal static float ReadFloat32(MlxArrayHandle handle)
    {
        EnsureScalar(handle);
        CheckStatus(MlxArray.ItemFloat32(out var result, handle), "item_float32");

        return result;
    }

    internal static double ReadFloat64(MlxArrayHandle handle)
    {
        EnsureScalar(handle);
        CheckStatus(MlxArray.ItemFloat64(out var result, handle), "item_float64");

        return result;
    }

    internal static int ReadInt32(MlxArrayHandle handle)
    {
        EnsureScalar(handle);
        CheckStatus(MlxArray.ItemInt32(out var result, handle), "item_int32");

        return result;
    }

    internal static long ReadInt64(MlxArrayHandle handle)
    {
        EnsureScalar(handle);
        CheckStatus(MlxArray.ItemInt64(out var result, handle), "item_int64");

        return result;
    }

    internal static bool ReadBool(MlxArrayHandle handle)
    {
        EnsureScalar(handle);
        CheckStatus(MlxArray.ItemBool(out var result, handle), "item_bool");

        return result != 0;
    }

    internal static Half ReadFloat16(MlxArrayHandle handle)
    {
        EnsureScalar(handle);
        CheckStatus(MlxArray.ItemFloat16(out var result, handle), "item_float16");

        return BitConverter.UInt16BitsToHalf(result);
    }

    internal static float[] ReadFlatFloat32(MlxArrayHandle handle)
    {
        CheckStatus(MlxArray.Eval(handle), "eval");

        return ReadStrided(handle, MlxArray.DataFloat32(handle));
    }

    internal static float[][] ReadVectorsFloat32(MlxArrayHandle handle)
    {
        if (Rank(handle) != 2)
            throw new InvalidOperationException("ReadVectors() requires a rank-2 tensor.");

        var rows = Dim(handle, 0);
        var cols = Dim(handle, 1);
        var flat = ReadFlatFloat32(handle);
        var result = new float[rows][];

        for (var row = 0; row < rows; row++)
        {
            var vector = new float[cols];
            Array.Copy(flat, row * cols, vector, 0, cols);
            result[row] = vector;
        }

        return result;
    }

    internal static MlxArrayHandle MeanPool(MlxArrayHandle values, MlxArrayHandle mask)
    {
        if (Rank(values) != 3)
            throw new InvalidOperationException("MeanPool expects values with shape [batch, tokens, hidden].");

        if (Rank(mask) != 2)
            throw new InvalidOperationException("MeanPool expects mask with shape [batch, tokens].");

        var batch = Dim(values, 0);
        var tokens = Dim(values, 1);
        var hidden = Dim(values, 2);

        if (Dim(mask, 0) != batch || Dim(mask, 1) != tokens)
            throw new InvalidOperationException("Mask shape must match the first two axes of the values tensor.");

        var valuesDType = MlxArray.DType(values);

        if (valuesDType != MlxArray.DType(mask))
            throw new InvalidOperationException("MeanPool requires values and mask to share the same dtype.");

        var maskExpanded = expandLast(mask, batch, tokens, hidden);

        try
        {
            var weighted = ApplyBinary(values, maskExpanded, BinaryTensorOp.Multiply);

            try
            {
                var weightedSum = ApplyReduction(weighted, [1], ReductionKind.Sum, false);

                try
                {
                    var maskSum = ApplyReduction(maskExpanded, [1], ReductionKind.Sum, false);

                    try
                    {
                        var epsilon = CreateScalar(1e-12, valuesDType);

                        try
                        {
                            var denominator = ApplyBinary(maskSum, epsilon, BinaryTensorOp.Maximum);

                            try
                            {
                                return ApplyBinary(weightedSum, denominator, BinaryTensorOp.Divide);
                            }
                            finally
                            {
                                Dispose(denominator);
                            }
                        }
                        finally
                        {
                            Dispose(epsilon);
                        }
                    }
                    finally
                    {
                        Dispose(maskSum);
                    }
                }
                finally
                {
                    Dispose(weightedSum);
                }
            }
            finally
            {
                Dispose(weighted);
            }
        }
        finally
        {
            Dispose(maskExpanded);
        }

        static MlxArrayHandle expandLast(MlxArrayHandle maskHandle, int batchSize, int tokenCount, int hiddenSize)
        {
            Span<int> expandedShape = stackalloc int[3] { batchSize, tokenCount, 1 };
            var expanded = Reshape(maskHandle, expandedShape);

            try
            {
                Span<int> broadcastShape = stackalloc int[3] { batchSize, tokenCount, hiddenSize };

                return Broadcast(expanded, broadcastShape);
            }
            finally
            {
                Dispose(expanded);
            }
        }
    }

    internal static MlxArrayHandle Broadcast(MlxArrayHandle handle, ReadOnlySpan<int> shape)
    {
        fixed (int* shapePtr = shape)
        {
            CheckStatus(MlxOps.BroadcastTo(out var result, handle, shapePtr, (nuint)shape.Length, DefaultStream()), "broadcast_to");

            return result;
        }
    }

    internal static MlxArrayHandle Reshape(MlxArrayHandle handle, ReadOnlySpan<int> shape)
    {
        fixed (int* shapePtr = shape)
        {
            CheckStatus(MlxOps.Reshape(out var result, handle, shapePtr, (nuint)shape.Length, DefaultStream()), "reshape");

            return result;
        }
    }

    internal static int NormalizeAxis(int axis, int rank)
    {
        if (axis < 0)
            axis += rank;

        if ((uint)axis >= (uint)rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        return axis;
    }

    internal static int ResolveIndex(Index index, int length)
    {
        var resolved = index.GetOffset(length);

        if ((uint)resolved >= (uint)length)
            throw new ArgumentOutOfRangeException(nameof(index));

        return resolved;
    }

    internal static ReadOnlySpan<int> ResolveRange(Range range, int length, Span<int> buffer)
    {
        var (start, count) = range.GetOffsetAndLength(length);

        if (count == 0)
            throw new ArgumentException("Axis range must contain at least one axis.", nameof(range));

        for (var i = 0; i < count; i++)
            buffer[i] = start + i;

        return buffer[..count];
    }

    private static T[] ReadStrided<T>(MlxArrayHandle handle, T* data) where T : unmanaged
    {
        var rank = Rank(handle);
        var count = Size(handle);
        var shape = MlxArray.Shape(handle);
        var strides = MlxArray.Strides(handle);
        var values = new T[count];

        if (count == 0)
            return values;

        if (rank == 0)
        {
            values[0] = *data;

            return values;
        }

        var coordinates = stackalloc int[rank];

        for (var linearIndex = 0; linearIndex < count; linearIndex++)
        {
            var remainder = linearIndex;

            for (var axis = rank - 1; axis >= 0; axis--)
            {
                var axisLength = shape[axis];
                coordinates[axis] = remainder % axisLength;
                remainder /= axisLength;
            }

            nuint offset = 0;

            for (var axis = 0; axis < rank; axis++)
                offset += (nuint)coordinates[axis] * strides[axis];

            values[linearIndex] = data[offset];
        }

        return values;
    }

    private static MlxArrayHandle ApplyUnary(MlxArrayHandle handle, UnaryTensorOp op)
    {
        switch (op)
        {
            case UnaryTensorOp.Abs:
                CheckStatus(MlxOps.Abs(out var absResult, handle, DefaultStream()), "abs");

                return absResult;
            case UnaryTensorOp.Exp:
                CheckStatus(MlxOps.Exp(out var expResult, handle, DefaultStream()), "exp");

                return expResult;
            case UnaryTensorOp.Log:
                CheckStatus(MlxOps.Log(out var logResult, handle, DefaultStream()), "log");

                return logResult;
            case UnaryTensorOp.LogicalNot:
                CheckStatus(MlxOps.LogicalNot(out var logicalNotResult, handle, DefaultStream()), "logical_not");

                return logicalNotResult;
            case UnaryTensorOp.Negate:
                CheckStatus(MlxOps.Negative(out var negResult, handle, DefaultStream()), "negative");

                return negResult;
            case UnaryTensorOp.Sqrt:
                CheckStatus(MlxOps.Sqrt(out var sqrtResult, handle, DefaultStream()), "sqrt");

                return sqrtResult;
            case UnaryTensorOp.TransposeLastTwo:
                return TransposeLastTwo(handle);
            default:
                throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    private static MlxArrayHandle ApplyBinary(MlxArrayHandle left, MlxArrayHandle right, BinaryTensorOp op, bool ownsRight = false)
    {
        try
        {
            switch (op)
            {
                case BinaryTensorOp.Add:
                    CheckStatus(MlxOps.Add(out var addResult, left, right, DefaultStream()), "add");

                    return addResult;
                case BinaryTensorOp.Divide:
                    CheckStatus(MlxOps.Divide(out var divResult, left, right, DefaultStream()), "divide");

                    return divResult;
                case BinaryTensorOp.LogicalAnd:
                    CheckStatus(MlxOps.LogicalAnd(out var logicalAndResult, left, right, DefaultStream()), "logical_and");

                    return logicalAndResult;
                case BinaryTensorOp.LogicalOr:
                    CheckStatus(MlxOps.LogicalOr(out var logicalOrResult, left, right, DefaultStream()), "logical_or");

                    return logicalOrResult;
                case BinaryTensorOp.Matmul:
                    CheckStatus(MlxOps.Matmul(out var matmulResult, left, right, DefaultStream()), "matmul");

                    return matmulResult;
                case BinaryTensorOp.Maximum:
                    CheckStatus(MlxOps.Maximum(out var maxResult, left, right, DefaultStream()), "maximum");

                    return maxResult;
                case BinaryTensorOp.Multiply:
                    CheckStatus(MlxOps.Multiply(out var mulResult, left, right, DefaultStream()), "multiply");

                    return mulResult;
                case BinaryTensorOp.Subtract:
                    CheckStatus(MlxOps.Subtract(out var subResult, left, right, DefaultStream()), "subtract");

                    return subResult;
                default:
                    throw new ArgumentOutOfRangeException(nameof(op));
            }
        }
        finally
        {
            if (ownsRight)
                Dispose(right);
        }
    }

    private static MlxArrayHandle ApplyReduction(MlxArrayHandle handle, ReadOnlySpan<int> axes, ReductionKind kind, bool keepDims)
    {
        if (axes.Length == 0)
            throw new ArgumentException("At least one axis is required.", nameof(axes));

        if (axes.Length == 1)
        {
            switch (kind)
            {
                case ReductionKind.Sum:
                    CheckStatus(MlxOps.SumAxis(out var singleSum, handle, axes[0], keepDims, DefaultStream()), "sum_axis");

                    return singleSum;
                case ReductionKind.Mean:
                    CheckStatus(MlxOps.MeanAxis(out var singleMean, handle, axes[0], keepDims, DefaultStream()), "mean_axis");

                    return singleMean;
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind));
            }
        }

        fixed (int* axesPtr = axes)
        {
            switch (kind)
            {
                case ReductionKind.Sum:
                    CheckStatus(MlxOps.SumAxes(out var sumResult, handle, axesPtr, (nuint)axes.Length, keepDims, DefaultStream()), "sum_axes");

                    return sumResult;
                case ReductionKind.Mean:
                    CheckStatus(MlxOps.MeanAxes(out var meanResult, handle, axesPtr, (nuint)axes.Length, keepDims, DefaultStream()), "mean_axes");

                    return meanResult;
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind));
            }
        }
    }

    private static MlxArrayHandle ApplySlice(MlxArrayHandle handle, ReadOnlySpan<AxisSelector> selectors)
    {
        var rank = Rank(handle);

        if (selectors.Length > rank)
            throw new ArgumentException("Too many selectors for tensor rank.", nameof(selectors));

        Span<int> starts = stackalloc int[rank];
        Span<int> stops = stackalloc int[rank];
        Span<int> strides = stackalloc int[rank];
        Span<int> squeezeAxes = stackalloc int[selectors.Length];
        var squeezeCount = 0;

        for (var axis = 0; axis < rank; axis++)
        {
            starts[axis] = 0;
            stops[axis] = Dim(handle, axis);
            strides[axis] = 1;
        }

        for (var axis = 0; axis < selectors.Length; axis++)
        {
            var selector = selectors[axis];
            var dimension = Dim(handle, axis);

            if (selector.IsIndex)
            {
                var index = ResolveIndex(selector.Index, dimension);
                starts[axis] = index;
                stops[axis] = index + 1;
                squeezeAxes[squeezeCount++] = axis;
            }
            else
            {
                var (start, count) = selector.Range.GetOffsetAndLength(dimension);
                starts[axis] = start;
                stops[axis] = start + count;
            }
        }

        fixed (int* startsPtr = starts)
        fixed (int* stopsPtr = stops)
        fixed (int* stridesPtr = strides)
        {
            CheckStatus(
                MlxOps.Slice(out var slice, handle, startsPtr, (nuint)rank, stopsPtr, (nuint)rank, stridesPtr, (nuint)rank, DefaultStream()),
                "slice");

            if (squeezeCount == 0)
                return slice;

            fixed (int* squeezePtr = squeezeAxes[..squeezeCount])
            {
                try
                {
                    CheckStatus(MlxOps.SqueezeAxes(out var squeezed, slice, squeezePtr, (nuint)squeezeCount, DefaultStream()), "squeeze_axes");

                    return squeezed;
                }
                finally
                {
                    Dispose(slice);
                }
            }
        }
    }

    private static MlxArrayHandle TransposeLastTwo(MlxArrayHandle handle)
    {
        var rank = Rank(handle);

        if (rank < 2)
            throw new InvalidOperationException("Transpose requires rank >= 2.");

        Span<int> permutation = stackalloc int[rank];

        for (var i = 0; i < rank; i++)
            permutation[i] = i;

        (permutation[rank - 2], permutation[rank - 1]) = (permutation[rank - 1], permutation[rank - 2]);

        fixed (int* permutationPtr = permutation)
        {
            CheckStatus(MlxOps.TransposeAxes(out var result, handle, permutationPtr, (nuint)rank, DefaultStream()), "transpose");

            return result;
        }
    }

    private static MlxArrayHandle CreateScalar(int value)
    {
        Span<int> shape = stackalloc int[1] { 1 };
        Span<int> values = stackalloc int[1] { value };

        return Create(values, shape);
    }

    private static MlxArrayHandle CreateScalar(long value)
    {
        Span<int> shape = stackalloc int[1] { 1 };
        Span<long> values = stackalloc long[1] { value };

        return Create(values, shape);
    }

    private static MlxArrayHandle CreateScalar(float value)
    {
        Span<int> shape = stackalloc int[1] { 1 };
        Span<float> values = stackalloc float[1] { value };

        return Create(values, shape);
    }

    private static MlxArrayHandle CreateScalar(double value)
    {
        Span<int> shape = stackalloc int[1] { 1 };
        Span<double> values = stackalloc double[1] { value };

        return Create(values, shape);
    }

    private static MlxArrayHandle CreateScalar(Half value)
    {
        Span<int> shape = stackalloc int[1] { 1 };
        Span<Half> values = stackalloc Half[1] { value };

        return Create(values, shape);
    }

    private static MlxArrayHandle CreateScalar(double value, MlxDType dtype) => dtype switch
    {
        MlxDType.MlxFloat16 => CreateScalar((Half)value),
        MlxDType.MlxFloat32 => CreateScalar((float)value),
        MlxDType.MlxFloat64 => CreateScalar(value),
        _ => throw new InvalidOperationException($"Scalar epsilon is not supported for dtype '{dtype}'."),
    };

    private static MlxArrayHandle CreateNewData<T>(ReadOnlySpan<T> values, ReadOnlySpan<int> shape, MlxDType dtype) where T : unmanaged
    {
        ValidateElementCount(shape, values.Length);

        fixed (T* dataPtr = values)
        fixed (int* shapePtr = shape)
            return MlxArray.NewData(dataPtr, shapePtr, shape.Length, dtype);
    }

    private static void ValidateElementCount(ReadOnlySpan<int> shape, int valueCount)
    {
        long total = 1;

        foreach (var dimension in shape)
        {
            if (dimension < 0)
                throw new ArgumentOutOfRangeException(nameof(shape), "Shape dimensions cannot be negative.");

            total *= dimension;
        }

        if (total != valueCount)
            throw new ArgumentException("Value count must match the provided shape.", nameof(shape));
    }

    private static void EnsureScalar(MlxArrayHandle handle)
    {
        if (Size(handle) != 1)
            throw new InvalidOperationException("Scalar extraction requires exactly one tensor element.");
    }

    private static void CheckStatus(int status, string operation)
    {
        if (status != 0)
            throw new InvalidOperationException($"MLX operation '{operation}' failed with status code {status}.");
    }
}
