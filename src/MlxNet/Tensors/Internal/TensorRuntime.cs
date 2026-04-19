// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;
using System.Threading;
using Itexoft.Mlx;

namespace Itexoft.Tensors.Internal;

internal enum ReductionKind
{
    Sum,
    Mean,
    Max,
    Min,
}

internal enum UnaryTensorOp
{
    Abs,
    Exp,
    Erf,
    Log,
    LogicalNot,
    Negate,
    Sqrt,
    Tanh,
    TransposeLastTwo,
}

internal enum BinaryTensorOp
{
    Add,
    Divide,
    Equal,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Matmul,
    Maximum,
    Minimum,
    Multiply,
    NotEqual,
    Remainder,
    Subtract,
}

internal static unsafe class TensorRuntime
{
    private static readonly Lock defaultStreamSync = new();
    private static MlxStreamHandle sCpuDefaultStream;
    private static MlxStreamHandle sGpuDefaultStream;

    internal static MlxArrayHandle RetainHandle(MlxArrayHandle source)
    {
        var slot = MlxArray.New();

        try
        {
            CheckStatus(MlxArray.Set(ref slot, source), "array_set");

            return slot;
        }
        catch
        {
            DisposeHandle(slot);

            throw;
        }
    }

    internal static void DisposeHandle(MlxArrayHandle handle)
    {
        if (handle.ctx != 0)
            CheckStatus(MlxArray.Free(handle), "array_free");
    }

    internal static MlxDType DType(MlxArrayHandle handle) => MlxArray.DType(handle);

    internal static int Rank(MlxArrayHandle handle) => checked((int)MlxArray.Ndim(handle));

    internal static int Size(MlxArrayHandle handle) => checked((int)MlxArray.Size(handle));

    internal static int Dim(MlxArrayHandle handle, int axis)
    {
        var rank = Rank(handle);
        axis = NormalizeAxis(axis, rank);

        return MlxArray.Dim(handle, axis);
    }

    internal static void Eval(MlxArrayHandle handle) => CheckStatus(MlxArray.Eval(handle), "eval");

    internal static MlxArrayHandle Create(ReadOnlySpan<bool> values, ReadOnlySpan<int> shape) =>
        CreateNewData(MemoryMarshal.AsBytes(values), shape, MlxDType.MlxBool);

    internal static MlxArrayHandle Create(ReadOnlySpan<int> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxInt32);

    internal static MlxArrayHandle Create(ReadOnlySpan<long> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxInt64);

    internal static MlxArrayHandle Create(ReadOnlySpan<Half> values, ReadOnlySpan<int> shape) =>
        CreateNewData(MemoryMarshal.Cast<Half, ushort>(values), shape, MlxDType.MlxFloat16);

    internal static MlxArrayHandle Create(ReadOnlySpan<float> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxFloat32);

    internal static MlxArrayHandle Create(ReadOnlySpan<double> values, ReadOnlySpan<int> shape) => CreateNewData(values, shape, MlxDType.MlxFloat64);

    internal static MlxArrayHandle CreateScalar(bool value) => MlxArray.NewBool(value);

    internal static MlxArrayHandle CreateScalar(int value, MlxDType dtype = MlxDType.MlxInt32)
    {
        var scalar = MlxArray.NewInt(value);

        return dtype == MlxDType.MlxInt32 ? scalar : CastOwned(scalar, dtype);
    }

    internal static MlxArrayHandle CreateScalar(long value, MlxDType dtype = MlxDType.MlxInt64)
    {
        Span<long> values = stackalloc long[1] { value };

        return CreateScalarFromData(values, dtype);
    }

    internal static MlxArrayHandle CreateScalar(Half value, MlxDType dtype = MlxDType.MlxFloat16)
    {
        Span<ushort> values = stackalloc ushort[1] { BitConverter.HalfToUInt16Bits(value) };

        return CreateScalarFromData(values, dtype);
    }

    internal static MlxArrayHandle CreateScalar(float value, MlxDType dtype = MlxDType.MlxFloat32)
    {
        var scalar = MlxArray.NewFloat32(value);

        return dtype == MlxDType.MlxFloat32 ? scalar : CastOwned(scalar, dtype);
    }

    internal static MlxArrayHandle CreateScalar(double value, MlxDType dtype = MlxDType.MlxFloat64)
    {
        var scalar = MlxArray.NewFloat64(value);

        return dtype == MlxDType.MlxFloat64 ? scalar : CastOwned(scalar, dtype);
    }

    internal static MlxArrayHandle CreateCompatibleScalar(MlxArrayHandle reference, int value) => DType(reference) switch
    {
        MlxDType.MlxInt32 => CreateScalar(value, MlxDType.MlxInt32),
        MlxDType.MlxInt64 => CreateScalar((long)value, MlxDType.MlxInt64),
        MlxDType.MlxFloat16 => CreateScalar((Half)value, MlxDType.MlxFloat16),
        MlxDType.MlxFloat32 => CreateScalar((float)value, MlxDType.MlxFloat32),
        MlxDType.MlxFloat64 => CreateScalar((double)value, MlxDType.MlxFloat64),
        _ => throw new InvalidOperationException($"Scalar arithmetic is not supported for dtype '{DType(reference)}'."),
    };

    internal static MlxArrayHandle CreateCompatibleScalar(MlxArrayHandle reference, float value) => DType(reference) switch
    {
        MlxDType.MlxInt32 => CreateScalar(value, MlxDType.MlxFloat32),
        MlxDType.MlxInt64 => CreateScalar(value, MlxDType.MlxFloat32),
        MlxDType.MlxFloat16 => CreateScalar((Half)value, MlxDType.MlxFloat16),
        MlxDType.MlxFloat32 => CreateScalar(value, MlxDType.MlxFloat32),
        MlxDType.MlxFloat64 => CreateScalar((double)value, MlxDType.MlxFloat64),
        _ => throw new InvalidOperationException($"Scalar arithmetic is not supported for dtype '{DType(reference)}'."),
    };

    internal static MlxArrayHandle Cast(MlxArrayHandle handle, MlxDType dtype)
    {
        CheckStatus(MlxOps.Astype(out var result, handle, dtype, DefaultStream()), "astype");

        return result;
    }

    internal static MlxArrayHandle Zeros(ReadOnlySpan<int> shape, MlxDType dtype)
    {
        fixed (int* shapePtr = shape)
        {
            CheckStatus(MlxOps.Zeros(out var result, shapePtr, (nuint)shape.Length, dtype, DefaultStream()), "zeros");

            return result;
        }
    }

    internal static MlxArrayHandle Ones(ReadOnlySpan<int> shape, MlxDType dtype)
    {
        fixed (int* shapePtr = shape)
        {
            CheckStatus(MlxOps.Ones(out var result, shapePtr, (nuint)shape.Length, dtype, DefaultStream()), "ones");

            return result;
        }
    }

    internal static MlxArrayHandle Full(ReadOnlySpan<int> shape, float value, MlxDType dtype)
    {
        var fill = CreateScalar(value, dtype);

        try
        {
            fixed (int* shapePtr = shape)
            {
                CheckStatus(MlxOps.Full(out var result, shapePtr, (nuint)shape.Length, fill, dtype, DefaultStream()), "full");

                return result;
            }
        }
        finally
        {
            DisposeHandle(fill);
        }
    }

    internal static MlxArrayHandle Arange(int start, int stop, int step, MlxDType dtype)
    {
        CheckStatus(MlxOps.Arange(out var result, start, stop, step, dtype, DefaultStream()), "arange");

        return result;
    }

    internal static MlxArrayHandle Arange(float start, float stop, float step, MlxDType dtype)
    {
        CheckStatus(MlxOps.Arange(out var result, start, stop, step, dtype, DefaultStream()), "arange");

        return result;
    }

    internal static MlxArrayHandle Unary(MlxArrayHandle handle, UnaryTensorOp op)
    {
        switch (op)
        {
            case UnaryTensorOp.Abs:
                CheckStatus(MlxOps.Abs(out var absResult, handle, DefaultStream()), "abs");

                return absResult;
            case UnaryTensorOp.Exp:
                CheckStatus(MlxOps.Exp(out var expResult, handle, DefaultStream()), "exp");

                return expResult;
            case UnaryTensorOp.Erf:
                CheckStatus(MlxOps.Erf(out var erfResult, handle, DefaultStream()), "erf");

                return erfResult;
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
            case UnaryTensorOp.Tanh:
                CheckStatus(MlxOps.Tanh(out var tanhResult, handle, DefaultStream()), "tanh");

                return tanhResult;
            case UnaryTensorOp.TransposeLastTwo:
                return TransposeLastTwo(handle);
            default:
                throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    internal static MlxArrayHandle Binary(MlxArrayHandle left, MlxArrayHandle right, BinaryTensorOp op)
    {
        switch (op)
        {
            case BinaryTensorOp.Add:
                CheckStatus(MlxOps.Add(out var addResult, left, right, DefaultStream()), "add");

                return addResult;
            case BinaryTensorOp.Divide:
                CheckStatus(MlxOps.Divide(out var divResult, left, right, DefaultStream()), "divide");

                return divResult;
            case BinaryTensorOp.Equal:
                CheckStatus(MlxOps.Equal(out var equalResult, left, right, DefaultStream()), "equal");

                return equalResult;
            case BinaryTensorOp.Greater:
                CheckStatus(MlxOps.Greater(out var greaterResult, left, right, DefaultStream()), "greater");

                return greaterResult;
            case BinaryTensorOp.GreaterEqual:
                CheckStatus(MlxOps.GreaterEqual(out var greaterEqualResult, left, right, DefaultStream()), "greater_equal");

                return greaterEqualResult;
            case BinaryTensorOp.Less:
                CheckStatus(MlxOps.Less(out var lessResult, left, right, DefaultStream()), "less");

                return lessResult;
            case BinaryTensorOp.LessEqual:
                CheckStatus(MlxOps.LessEqual(out var lessEqualResult, left, right, DefaultStream()), "less_equal");

                return lessEqualResult;
            case BinaryTensorOp.LogicalAnd:
                CheckStatus(MlxOps.LogicalAnd(out var logicalAndResult, left, right, DefaultStream()), "logical_and");

                return logicalAndResult;
            case BinaryTensorOp.LogicalOr:
                CheckStatus(MlxOps.LogicalOr(out var logicalOrResult, left, right, DefaultStream()), "logical_or");

                return logicalOrResult;
            case BinaryTensorOp.LogicalXor:
                CheckStatus(MlxOps.BitwiseXor(out var logicalXorResult, left, right, DefaultStream()), "logical_xor");

                return logicalXorResult;
            case BinaryTensorOp.Matmul:
                CheckStatus(MlxOps.Matmul(out var matmulResult, left, right, DefaultStream()), "matmul");

                return matmulResult;
            case BinaryTensorOp.Maximum:
                CheckStatus(MlxOps.Maximum(out var maxResult, left, right, DefaultStream()), "maximum");

                return maxResult;
            case BinaryTensorOp.Minimum:
                CheckStatus(MlxOps.Minimum(out var minResult, left, right, DefaultStream()), "minimum");

                return minResult;
            case BinaryTensorOp.Multiply:
                CheckStatus(MlxOps.Multiply(out var mulResult, left, right, DefaultStream()), "multiply");

                return mulResult;
            case BinaryTensorOp.NotEqual:
                CheckStatus(MlxOps.NotEqual(out var notEqualResult, left, right, DefaultStream()), "not_equal");

                return notEqualResult;
            case BinaryTensorOp.Remainder:
                CheckStatus(MlxOps.Remainder(out var remainderResult, left, right, DefaultStream()), "remainder");

                return remainderResult;
            case BinaryTensorOp.Subtract:
                CheckStatus(MlxOps.Subtract(out var subResult, left, right, DefaultStream()), "subtract");

                return subResult;
            default:
                throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    internal static MlxArrayHandle Where(MlxArrayHandle condition, MlxArrayHandle whenTrue, MlxArrayHandle whenFalse)
    {
        CheckStatus(MlxOps.Where(out var result, condition, whenTrue, whenFalse, DefaultStream()), "where");

        return result;
    }

    internal static MlxArrayHandle TakeAlong(MlxArrayHandle handle, MlxArrayHandle indices, int axis)
    {
        axis = NormalizeAxis(axis, Rank(handle));
        CheckStatus(MlxOps.TakeAlongAxis(out var result, handle, indices, axis, DefaultStream()), "take_along_axis");

        return result;
    }

    internal static MlxArrayHandle Softmax(MlxArrayHandle handle, Index axis)
    {
        var resolvedAxis = ResolveIndex(axis, Rank(handle));
        CheckStatus(MlxOps.SoftmaxAxis(out var result, handle, resolvedAxis, false, DefaultStream()), "softmax_axis");

        return result;
    }

    internal static MlxArrayHandle Reshape(MlxArrayHandle handle, ReadOnlySpan<int> shape)
    {
        fixed (int* shapePtr = shape)
        {
            CheckStatus(MlxOps.Reshape(out var result, handle, shapePtr, (nuint)shape.Length, DefaultStream()), "reshape");

            return result;
        }
    }

    internal static MlxArrayHandle ExpandDims(MlxArrayHandle handle, int axis)
    {
        axis = NormalizeAxis(axis, Rank(handle) + 1);
        CheckStatus(MlxOps.ExpandDims(out var result, handle, axis, DefaultStream()), "expand_dims");

        return result;
    }

    internal static MlxArrayHandle Squeeze(MlxArrayHandle handle, ReadOnlySpan<int> axes)
    {
        if (axes.Length == 0)
        {
            CheckStatus(MlxOps.Squeeze(out var result, handle, DefaultStream()), "squeeze");

            return result;
        }

        Span<int> normalized = stackalloc int[axes.Length];

        for (var i = 0; i < axes.Length; i++)
            normalized[i] = NormalizeAxis(axes[i], Rank(handle));

        fixed (int* axesPtr = normalized)
        {
            CheckStatus(MlxOps.SqueezeAxes(out var result, handle, axesPtr, (nuint)normalized.Length, DefaultStream()), "squeeze_axes");

            return result;
        }
    }

    internal static MlxArrayHandle Transpose(MlxArrayHandle handle, ReadOnlySpan<int> axes)
    {
        if (axes.Length != Rank(handle))
            throw new ArgumentException("Number of axes must match tensor rank.", nameof(axes));

        Span<int> normalized = stackalloc int[axes.Length];

        for (var i = 0; i < axes.Length; i++)
            normalized[i] = NormalizeAxis(axes[i], axes.Length);

        fixed (int* axesPtr = normalized)
        {
            CheckStatus(MlxOps.TransposeAxes(out var result, handle, axesPtr, (nuint)normalized.Length, DefaultStream()), "transpose");

            return result;
        }
    }

    internal static MlxArrayHandle MoveAxis(MlxArrayHandle handle, int source, int destination)
    {
        var rank = Rank(handle);
        source = NormalizeAxis(source, rank);
        destination = NormalizeAxis(destination, rank);
        CheckStatus(MlxOps.Moveaxis(out var result, handle, source, destination, DefaultStream()), "moveaxis");

        return result;
    }

    internal static MlxArrayHandle Broadcast(MlxArrayHandle handle, ReadOnlySpan<int> shape)
    {
        fixed (int* shapePtr = shape)
        {
            CheckStatus(MlxOps.BroadcastTo(out var result, handle, shapePtr, (nuint)shape.Length, DefaultStream()), "broadcast_to");

            return result;
        }
    }

    internal static MlxArrayHandle Copy(MlxArrayHandle handle)
    {
        CheckStatus(MlxOps.Copy(out var result, handle, DefaultStream()), "copy");

        return result;
    }

    internal static MlxArrayHandle Pow(MlxArrayHandle handle, float exponent)
    {
        var scalar = CreateCompatibleScalar(handle, exponent);

        try
        {
            CheckStatus(MlxOps.Power(out var result, handle, scalar, DefaultStream()), "power");

            return result;
        }
        finally
        {
            DisposeHandle(scalar);
        }
    }

    internal static MlxArrayHandle Slice(MlxArrayHandle handle, ReadOnlySpan<AxisSelector> selectors)
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
                    DisposeHandle(slice);
                }
            }
        }
    }

    internal static MlxArrayHandle Reduce(MlxArrayHandle handle, ReductionKind kind, bool keepDims)
    {
        switch (kind)
        {
            case ReductionKind.Sum:
                CheckStatus(MlxOps.Sum(out var sumResult, handle, keepDims, DefaultStream()), "sum");

                return sumResult;
            case ReductionKind.Mean:
                CheckStatus(MlxOps.Mean(out var meanResult, handle, keepDims, DefaultStream()), "mean");

                return meanResult;
            case ReductionKind.Max:
                CheckStatus(MlxOps.Max(out var maxResult, handle, keepDims, DefaultStream()), "max");

                return maxResult;
            case ReductionKind.Min:
                CheckStatus(MlxOps.Min(out var minResult, handle, keepDims, DefaultStream()), "min");

                return minResult;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }

    internal static MlxArrayHandle Reduce(MlxArrayHandle handle, Index axis, ReductionKind kind, bool keepDims)
    {
        var resolved = ResolveIndex(axis, Rank(handle));

        return Reduce(handle, [resolved], kind, keepDims);
    }

    internal static MlxArrayHandle Reduce(MlxArrayHandle handle, Range axes, ReductionKind kind, bool keepDims)
    {
        Span<int> buffer = stackalloc int[Rank(handle)];
        var resolved = ResolveRange(axes, Rank(handle), buffer);

        return Reduce(handle, resolved, kind, keepDims);
    }

    internal static MlxArrayHandle Reduce(MlxArrayHandle handle, ReadOnlySpan<int> axes, ReductionKind kind, bool keepDims)
    {
        if (axes.Length == 0)
            return Reduce(handle, kind, keepDims);

        if (axes.Length == 1)
        {
            return kind switch
            {
                ReductionKind.Sum => ReduceSingleAxis(handle, axes[0], keepDims, kind),
                ReductionKind.Mean => ReduceSingleAxis(handle, axes[0], keepDims, kind),
                ReductionKind.Max => ReduceSingleAxis(handle, axes[0], keepDims, kind),
                ReductionKind.Min => ReduceSingleAxis(handle, axes[0], keepDims, kind),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }

        Span<int> normalized = stackalloc int[axes.Length];

        for (var i = 0; i < axes.Length; i++)
            normalized[i] = NormalizeAxis(axes[i], Rank(handle));

        normalized.Sort();

        fixed (int* axesPtr = normalized)
        {
            return kind switch
            {
                ReductionKind.Sum => ReduceManyAxes(handle, axesPtr, normalized.Length, keepDims, kind),
                ReductionKind.Mean => ReduceManyAxes(handle, axesPtr, normalized.Length, keepDims, kind),
                ReductionKind.Max => ReduceManyAxes(handle, axesPtr, normalized.Length, keepDims, kind),
                ReductionKind.Min => ReduceManyAxes(handle, axesPtr, normalized.Length, keepDims, kind),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }
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

        if (DType(values) != DType(mask))
            throw new InvalidOperationException("MeanPool requires values and mask to share the same dtype.");

        Span<int> expandedShape = stackalloc int[3] { batch, tokens, 1 };
        var expanded = Reshape(mask, expandedShape);

        try
        {
            Span<int> broadcastShape = stackalloc int[3] { batch, tokens, hidden };
            var broadcast = Broadcast(expanded, broadcastShape);

            try
            {
                var weighted = Binary(values, broadcast, BinaryTensorOp.Multiply);

                try
                {
                    var weightedSum = Reduce(weighted, (Index)1, ReductionKind.Sum, false);

                    try
                    {
                        var maskSum = Reduce(broadcast, (Index)1, ReductionKind.Sum, false);

                        try
                        {
                            var epsilon = CreateCompatibleScalar(values, 1e-12f);

                            try
                            {
                                var denominator = Binary(maskSum, epsilon, BinaryTensorOp.Maximum);

                                try
                                {
                                    return Binary(weightedSum, denominator, BinaryTensorOp.Divide);
                                }
                                finally
                                {
                                    DisposeHandle(denominator);
                                }
                            }
                            finally
                            {
                                DisposeHandle(epsilon);
                            }
                        }
                        finally
                        {
                            DisposeHandle(maskSum);
                        }
                    }
                    finally
                    {
                        DisposeHandle(weightedSum);
                    }
                }
                finally
                {
                    DisposeHandle(weighted);
                }
            }
            finally
            {
                DisposeHandle(broadcast);
            }
        }
        finally
        {
            DisposeHandle(expanded);
        }
    }

    internal static int ResolveIndex(Index index, int length)
    {
        var resolved = index.GetOffset(length);

        if ((uint)resolved >= (uint)length)
            throw new ArgumentOutOfRangeException(nameof(index));

        return resolved;
    }

    internal static int NormalizeAxis(int axis, int rank)
    {
        if (axis < 0)
            axis += rank;

        if ((uint)axis >= (uint)rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        return axis;
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

    internal static int ReadInt32(MlxArrayHandle handle)
    {
        EnsureScalar(handle);

        if (DType(handle) != MlxDType.MlxInt32)
        {
            var casted = Cast(handle, MlxDType.MlxInt32);

            try
            {
                CheckStatus(MlxArray.ItemInt32(out var castedResult, casted), "item_int32");

                return castedResult;
            }
            finally
            {
                DisposeHandle(casted);
            }
        }

        CheckStatus(MlxArray.ItemInt32(out var result, handle), "item_int32");

        return result;
    }

    internal static long ReadInt64(MlxArrayHandle handle)
    {
        EnsureScalar(handle);

        if (DType(handle) != MlxDType.MlxInt64)
        {
            var casted = Cast(handle, MlxDType.MlxInt64);

            try
            {
                CheckStatus(MlxArray.ItemInt64(out var castedResult, casted), "item_int64");

                return castedResult;
            }
            finally
            {
                DisposeHandle(casted);
            }
        }

        CheckStatus(MlxArray.ItemInt64(out var result, handle), "item_int64");

        return result;
    }

    internal static Half ReadFloat16(MlxArrayHandle handle)
    {
        EnsureScalar(handle);

        if (DType(handle) != MlxDType.MlxFloat16)
        {
            var casted = Cast(handle, MlxDType.MlxFloat16);

            try
            {
                CheckStatus(MlxArray.ItemFloat16(out var castedResult, casted), "item_float16");

                return BitConverter.UInt16BitsToHalf(castedResult);
            }
            finally
            {
                DisposeHandle(casted);
            }
        }

        CheckStatus(MlxArray.ItemFloat16(out var result, handle), "item_float16");

        return BitConverter.UInt16BitsToHalf(result);
    }

    internal static float ReadFloat32(MlxArrayHandle handle)
    {
        EnsureScalar(handle);

        if (DType(handle) != MlxDType.MlxFloat32)
        {
            var casted = Cast(handle, MlxDType.MlxFloat32);

            try
            {
                CheckStatus(MlxArray.ItemFloat32(out var castedResult, casted), "item_float32");

                return castedResult;
            }
            finally
            {
                DisposeHandle(casted);
            }
        }

        CheckStatus(MlxArray.ItemFloat32(out var result, handle), "item_float32");

        return result;
    }

    internal static double ReadFloat64(MlxArrayHandle handle)
    {
        EnsureScalar(handle);

        if (DType(handle) != MlxDType.MlxFloat64)
        {
            var casted = Cast(handle, MlxDType.MlxFloat64);

            try
            {
                CheckStatus(MlxArray.ItemFloat64(out var castedResult, casted), "item_float64");

                return castedResult;
            }
            finally
            {
                DisposeHandle(casted);
            }
        }

        CheckStatus(MlxArray.ItemFloat64(out var result, handle), "item_float64");

        return result;
    }

    internal static bool ReadBool(MlxArrayHandle handle)
    {
        EnsureScalar(handle);

        if (DType(handle) != MlxDType.MlxBool)
        {
            var casted = Cast(handle, MlxDType.MlxBool);

            try
            {
                CheckStatus(MlxArray.ItemBool(out var castedResult, casted), "item_bool");

                return castedResult != 0;
            }
            finally
            {
                DisposeHandle(casted);
            }
        }

        CheckStatus(MlxArray.ItemBool(out var result, handle), "item_bool");

        return result != 0;
    }

    internal static float[][] ReadVectors(MlxArrayHandle handle)
    {
        if (Rank(handle) != 2)
            throw new InvalidOperationException("ReadVectors() requires a rank-2 tensor.");

        var values = DType(handle) == MlxDType.MlxFloat32 ? ReadFlatFloat32(handle) : ReadFlatFloat32Temporary(handle);

        var rows = Dim(handle, 0);
        var cols = Dim(handle, 1);
        var result = new float[rows][];

        for (var row = 0; row < rows; row++)
        {
            var vector = new float[cols];
            Array.Copy(values, row * cols, vector, 0, cols);
            result[row] = vector;
        }

        return result;
    }

    private static MlxStreamHandle DefaultStream()
    {
        CheckStatus(MlxDevice.GetDefaultDevice(out var device), "get_default_device");

        try
        {
            CheckStatus(MlxDevice.GetType(out var type, device), "get_default_device_type");

            return type switch
            {
                MlxDeviceType.MlxCpu => GetCachedDefaultStream(device, ref sCpuDefaultStream),
                MlxDeviceType.MlxGpu => GetCachedDefaultStream(device, ref sGpuDefaultStream),
                _ => throw new InvalidOperationException($"Unsupported MLX device type '{type}'."),
            };
        }
        finally
        {
            if (device.ctx != 0)
                CheckStatus(MlxDevice.Free(device), "free_default_device");
        }
    }

    private static MlxStreamHandle GetCachedDefaultStream(MlxDeviceHandle device, ref MlxStreamHandle cachedStream)
    {
        lock (defaultStreamSync)
        {
            CheckStatus(MlxStream.GetDefaultStream(out var currentStream, device), "get_default_stream");

            try
            {
                if (cachedStream.ctx == 0)
                {
                    cachedStream = currentStream;
                    currentStream = default;
                }
                else
                    CheckStatus(MlxStream.Set(ref cachedStream, currentStream), "set_cached_default_stream");

                return cachedStream;
            }
            finally
            {
                if (currentStream.ctx != 0)
                    CheckStatus(MlxStream.Free(currentStream), "free_current_default_stream");
            }
        }
    }

    private static MlxArrayHandle CastOwned(MlxArrayHandle handle, MlxDType dtype)
    {
        try
        {
            return Cast(handle, dtype);
        }
        finally
        {
            DisposeHandle(handle);
        }
    }

    private static MlxArrayHandle CreateScalarFromData<T>(ReadOnlySpan<T> values, MlxDType dtype) where T : unmanaged
    {
        ReadOnlySpan<int> shape = [];

        return CreateNewData(values, shape, dtype);
    }

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

    private static MlxArrayHandle TransposeLastTwo(MlxArrayHandle handle)
    {
        var rank = Rank(handle);

        if (rank < 2)
            throw new InvalidOperationException("Transpose requires rank >= 2.");

        Span<int> permutation = stackalloc int[rank];

        for (var i = 0; i < rank; i++)
            permutation[i] = i;

        (permutation[rank - 2], permutation[rank - 1]) = (permutation[rank - 1], permutation[rank - 2]);

        return Transpose(handle, permutation);
    }

    private static float[] ReadFlatFloat32(MlxArrayHandle handle)
    {
        Eval(handle);

        return ReadStrided(handle, MlxArray.DataFloat32(handle));
    }

    private static float[] ReadFlatFloat32Temporary(MlxArrayHandle handle)
    {
        var casted = Cast(handle, MlxDType.MlxFloat32);

        try
        {
            return ReadFlatFloat32(casted);
        }
        finally
        {
            DisposeHandle(casted);
        }
    }

    private static MlxArrayHandle ReduceSingleAxis(MlxArrayHandle handle, int axis, bool keepDims, ReductionKind kind)
    {
        switch (kind)
        {
            case ReductionKind.Sum:
                CheckStatus(MlxOps.SumAxis(out var sumResult, handle, axis, keepDims, DefaultStream()), "sum_axis");

                return sumResult;
            case ReductionKind.Mean:
                CheckStatus(MlxOps.MeanAxis(out var meanResult, handle, axis, keepDims, DefaultStream()), "mean_axis");

                return meanResult;
            case ReductionKind.Max:
                CheckStatus(MlxOps.MaxAxis(out var maxResult, handle, axis, keepDims, DefaultStream()), "max_axis");

                return maxResult;
            case ReductionKind.Min:
                CheckStatus(MlxOps.MinAxis(out var minResult, handle, axis, keepDims, DefaultStream()), "min_axis");

                return minResult;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }

    private static MlxArrayHandle ReduceManyAxes(MlxArrayHandle handle, int* axes, int axisCount, bool keepDims, ReductionKind kind)
    {
        switch (kind)
        {
            case ReductionKind.Sum:
                CheckStatus(MlxOps.SumAxes(out var sumResult, handle, axes, (nuint)axisCount, keepDims, DefaultStream()), "sum_axes");

                return sumResult;
            case ReductionKind.Mean:
                CheckStatus(MlxOps.MeanAxes(out var meanResult, handle, axes, (nuint)axisCount, keepDims, DefaultStream()), "mean_axes");

                return meanResult;
            case ReductionKind.Max:
                CheckStatus(MlxOps.MaxAxes(out var maxResult, handle, axes, (nuint)axisCount, keepDims, DefaultStream()), "max_axes");

                return maxResult;
            case ReductionKind.Min:
                CheckStatus(MlxOps.MinAxes(out var minResult, handle, axes, (nuint)axisCount, keepDims, DefaultStream()), "min_axes");

                return minResult;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
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

    private static void EnsureScalar(MlxArrayHandle handle)
    {
        if (Rank(handle) != 0 || Size(handle) != 1)
            throw new InvalidOperationException("Scalar extraction requires a rank-0 tensor.");
    }

    private static void CheckStatus(int status, string operation)
    {
        if (status != 0)
            throw new InvalidOperationException($"MLX operation '{operation}' failed with status code {status}.");
    }
}
