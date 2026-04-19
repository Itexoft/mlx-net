// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;
using Itexoft.Tensors.Internal;

namespace Itexoft.Tensors;

public readonly ref partial struct Tensor
{
    private readonly MlxArrayHandle handle;

    internal Tensor(MlxArrayHandle handle) => this.handle = handle;

    public static Tensor Scalar(bool value) => AdoptOwned(TensorRuntime.CreateScalar(value));

    public static Tensor Scalar(int value, MlxDType dtype = MlxDType.MlxInt32) => AdoptOwned(TensorRuntime.CreateScalar(value, dtype));

    public static Tensor Scalar(long value, MlxDType dtype = MlxDType.MlxInt64) => AdoptOwned(TensorRuntime.CreateScalar(value, dtype));

    public static Tensor Scalar(Half value, MlxDType dtype = MlxDType.MlxFloat16) => AdoptOwned(TensorRuntime.CreateScalar(value, dtype));

    public static Tensor Scalar(float value, MlxDType dtype = MlxDType.MlxFloat32) => AdoptOwned(TensorRuntime.CreateScalar(value, dtype));

    public static Tensor Scalar(double value, MlxDType dtype = MlxDType.MlxFloat64) => AdoptOwned(TensorRuntime.CreateScalar(value, dtype));

    public static Tensor From(ReadOnlySpan<bool> values, Shape shape) => Create(shape, values, TensorRuntime.Create);

    public static Tensor From(ReadOnlySpan<int> values, Shape shape) => Create(shape, values, TensorRuntime.Create);

    public static Tensor From(ReadOnlySpan<long> values, Shape shape) => Create(shape, values, TensorRuntime.Create);

    public static Tensor From(ReadOnlySpan<Half> values, Shape shape) => Create(shape, values, TensorRuntime.Create);

    public static Tensor From(ReadOnlySpan<float> values, Shape shape) => Create(shape, values, TensorRuntime.Create);

    public static Tensor From(ReadOnlySpan<double> values, Shape shape) => Create(shape, values, TensorRuntime.Create);

    public static Tensor Zeros(Shape shape, MlxDType dtype = MlxDType.MlxFloat32) => Create(shape, dims => TensorRuntime.Zeros(dims, dtype));

    public static Tensor Ones(Shape shape, MlxDType dtype = MlxDType.MlxFloat32) => Create(shape, dims => TensorRuntime.Ones(dims, dtype));

    public static Tensor Full(Shape shape, float value, MlxDType dtype = MlxDType.MlxFloat32) =>
        Create(shape, dims => TensorRuntime.Full(dims, value, dtype));

    public static Tensor Arange(int start, int stop, int step = 1, MlxDType dtype = MlxDType.MlxInt32) =>
        AdoptOwned(TensorRuntime.Arange(start, stop, step, dtype));

    public static Tensor Arange(float start, float stop, float step = 1f, MlxDType dtype = MlxDType.MlxFloat32) =>
        AdoptOwned(TensorRuntime.Arange(start, stop, step, dtype));

    public readonly ShapeView Shape => new(this.Borrow());

    public readonly int Rank => TensorRuntime.Rank(this.Borrow());

    public readonly MlxDType DType => TensorRuntime.DType(this.Borrow());

    public readonly int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public readonly Tensor T => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public readonly Tensor Abs => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Abs));

    public readonly Tensor Exp => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Exp));

    public readonly Tensor Erf => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Erf));

    public readonly Tensor Log => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Log));

    public readonly Tensor Sqrt => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Sqrt));

    public readonly Tensor Tanh => CreateOwned(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Tanh));

    public readonly Tensor this[AxisSelector selector0] => CreateOwned(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public readonly Tensor this[AxisSelector selector0, AxisSelector selector1] =>
        CreateOwned(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public readonly Tensor this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        CreateOwned(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public readonly Tensor this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        CreateOwned(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public readonly Tensor Reshape(scoped ReadOnlySpan<int> shape) => CreateOwned(TensorRuntime.Reshape(this.Borrow(), shape));

    public readonly Tensor ExpandDims(int axis) => CreateOwned(TensorRuntime.ExpandDims(this.Borrow(), axis));

    public readonly Tensor ExpandedDimension(int axis) => this.ExpandDims(axis);

    public readonly Tensor Squeeze(scoped ReadOnlySpan<int> axes) => CreateOwned(TensorRuntime.Squeeze(this.Borrow(), axes));

    public readonly Tensor Squeezed(scoped ReadOnlySpan<int> axes) => this.Squeeze(axes);

    public readonly Tensor Transpose(scoped ReadOnlySpan<int> axes) => CreateOwned(TensorRuntime.Transpose(this.Borrow(), axes));

    public readonly Tensor Transposed(scoped ReadOnlySpan<int> axes) => this.Transpose(axes);

    public readonly Tensor MoveAxis(int source, int destination) => CreateOwned(TensorRuntime.MoveAxis(this.Borrow(), source, destination));

    public readonly Tensor BroadcastTo(scoped ReadOnlySpan<int> shape) => CreateOwned(TensorRuntime.Broadcast(this.Borrow(), shape));

    public readonly Tensor Copy() => CreateOwned(TensorRuntime.Copy(this.Borrow()));

    public readonly Tensor Cast(MlxDType dtype) => CreateOwned(TensorRuntime.Cast(this.Borrow(), dtype));

    public readonly Tensor Matmul(Tensor other) => this.Binary(other, BinaryTensorOp.Matmul);

    public readonly Tensor Mm(Tensor other) => this.Matmul(other);

    public readonly Tensor TakeAlong(Tensor indices, int axis) => CreateOwned(TensorRuntime.TakeAlong(this.Borrow(), indices.Borrow(), axis));

    public readonly Tensor Where(Tensor whenTrue, Tensor whenFalse) =>
        CreateOwned(TensorRuntime.Where(this.Borrow(), whenTrue.Borrow(), whenFalse.Borrow()));

    public readonly Tensor Pow(float exponent) => CreateOwned(TensorRuntime.Pow(this.Borrow(), exponent));

    public readonly Tensor MeanPool(Tensor mask) => CreateOwned(TensorRuntime.MeanPool(this.Borrow(), mask.Borrow()));

    public readonly Tensor Softmax(Index axis) => CreateOwned(TensorRuntime.Softmax(this.Borrow(), axis));

    public readonly Tensor Sum(bool keepDims = false) => CreateOwned(TensorRuntime.Reduce(this.Borrow(), ReductionKind.Sum, keepDims));

    public readonly Tensor Sum(Index axis, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axis, ReductionKind.Sum, keepDims));

    public readonly Tensor Sum(Range axes, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axes, ReductionKind.Sum, keepDims));

    public readonly Tensor Mean(bool keepDims = false) => CreateOwned(TensorRuntime.Reduce(this.Borrow(), ReductionKind.Mean, keepDims));

    public readonly Tensor Mean(Index axis, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axis, ReductionKind.Mean, keepDims));

    public readonly Tensor Mean(Range axes, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axes, ReductionKind.Mean, keepDims));

    public readonly Tensor Max(bool keepDims = false) => CreateOwned(TensorRuntime.Reduce(this.Borrow(), ReductionKind.Max, keepDims));

    public readonly Tensor Max(Index axis, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axis, ReductionKind.Max, keepDims));

    public readonly Tensor Max(Range axes, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axes, ReductionKind.Max, keepDims));

    public readonly Tensor Max(Tensor other) => this.Binary(other, BinaryTensorOp.Maximum);

    public readonly Tensor Min(bool keepDims = false) => CreateOwned(TensorRuntime.Reduce(this.Borrow(), ReductionKind.Min, keepDims));

    public readonly Tensor Min(Index axis, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axis, ReductionKind.Min, keepDims));

    public readonly Tensor Min(Range axes, bool keepDims = false) =>
        CreateOwned(TensorRuntime.Reduce(this.Borrow(), axes, ReductionKind.Min, keepDims));

    public readonly Tensor Min(Tensor other) => this.Binary(other, BinaryTensorOp.Minimum);

    public readonly Tensor Eq(Tensor other) => this.Binary(other, BinaryTensorOp.Equal);

    public readonly Tensor Ne(Tensor other) => this.Binary(other, BinaryTensorOp.NotEqual);

    public readonly Tensor Lt(Tensor other) => this.Binary(other, BinaryTensorOp.Less);

    public readonly Tensor Le(Tensor other) => this.Binary(other, BinaryTensorOp.LessEqual);

    public readonly Tensor Gt(Tensor other) => this.Binary(other, BinaryTensorOp.Greater);

    public readonly Tensor Ge(Tensor other) => this.Binary(other, BinaryTensorOp.GreaterEqual);

    public readonly float[][] ReadVectors() => TensorRuntime.ReadVectors(this.Borrow());

    public readonly void Eval() => TensorRuntime.Eval(this.Borrow());

    public static explicit operator bool(Tensor value) => TensorRuntime.ReadBool(value.Borrow());

    public static explicit operator int(Tensor value) => TensorRuntime.ReadInt32(value.Borrow());

    public static explicit operator long(Tensor value) => TensorRuntime.ReadInt64(value.Borrow());

    public static explicit operator Half(Tensor value) => TensorRuntime.ReadFloat16(value.Borrow());

    public static explicit operator float(Tensor value) => TensorRuntime.ReadFloat32(value.Borrow());

    public static explicit operator double(Tensor value) => TensorRuntime.ReadFloat64(value.Borrow());

    internal readonly bool IsAlive => this.handle.ctx != 0;

    internal readonly Tensor ApplyReduction(ReductionSpec spec) => CreateOwned(
        spec.IsRange
            ? TensorRuntime.Reduce(this.Borrow(), spec.Axes, ReductionKind.Sum, spec.KeepDims)
            : TensorRuntime.Reduce(this.Borrow(), spec.Axis, ReductionKind.Sum, spec.KeepDims));

    internal static Tensor AdoptOwned(MlxArrayHandle handle)
    {
        if (handle.ctx == 0)
            throw new ArgumentException("Tensor handle must reference a live MLX array.", nameof(handle));

        return new Tensor(handle);
    }

    internal readonly MlxArrayHandle Borrow()
    {
        if (!this.IsAlive)
            throw new InvalidOperationException("Tensor does not reference a live MLX array.");

        return this.handle;
    }

    private static Tensor CreateOwned(MlxArrayHandle result) => AdoptOwned(result);

    private readonly Tensor Binary(Tensor other, BinaryTensorOp op) => CreateOwned(TensorRuntime.Binary(this.Borrow(), other.Borrow(), op));

    private readonly Tensor ScalarRight(int value, BinaryTensorOp op)
    {
        var scalar = TensorRuntime.CreateCompatibleScalar(this.Borrow(), value);

        try
        {
            return CreateOwned(TensorRuntime.Binary(this.Borrow(), scalar, op));
        }
        finally
        {
            TensorRuntime.DisposeHandle(scalar);
        }
    }

    private readonly Tensor ScalarRight(float value, BinaryTensorOp op)
    {
        var scalar = TensorRuntime.CreateCompatibleScalar(this.Borrow(), value);

        try
        {
            return CreateOwned(TensorRuntime.Binary(this.Borrow(), scalar, op));
        }
        finally
        {
            TensorRuntime.DisposeHandle(scalar);
        }
    }

    private readonly Tensor ScalarLeft(int value, BinaryTensorOp op)
    {
        var scalar = TensorRuntime.CreateCompatibleScalar(this.Borrow(), value);

        try
        {
            return CreateOwned(TensorRuntime.Binary(scalar, this.Borrow(), op));
        }
        finally
        {
            TensorRuntime.DisposeHandle(scalar);
        }
    }

    private readonly Tensor ScalarLeft(float value, BinaryTensorOp op)
    {
        var scalar = TensorRuntime.CreateCompatibleScalar(this.Borrow(), value);

        try
        {
            return CreateOwned(TensorRuntime.Binary(scalar, this.Borrow(), op));
        }
        finally
        {
            TensorRuntime.DisposeHandle(scalar);
        }
    }

    private readonly void RequireBoolTensor()
    {
        if (this.DType != MlxDType.MlxBool)
            throw new InvalidOperationException("Logical tensor operators require boolean tensors.");
    }

    private static Tensor Create<T>(Shape shape, ReadOnlySpan<T> values, TensorFactory<T> factory) where T : unmanaged
    {
        var buffer = shape.Rank <= 16 ? stackalloc int[shape.Rank] : new int[shape.Rank];
        shape.CopyTo(buffer);

        return AdoptOwned(factory(values, buffer));
    }

    private static Tensor Create(Shape shape, ShapeFactory factory)
    {
        var buffer = shape.Rank <= 16 ? stackalloc int[shape.Rank] : new int[shape.Rank];
        shape.CopyTo(buffer);

        return AdoptOwned(factory(buffer));
    }

    private delegate MlxArrayHandle TensorFactory<T>(ReadOnlySpan<T> values, ReadOnlySpan<int> shape) where T : unmanaged;

    private delegate MlxArrayHandle ShapeFactory(ReadOnlySpan<int> shape);
}
