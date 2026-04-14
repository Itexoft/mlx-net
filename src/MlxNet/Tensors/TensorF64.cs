// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly ref struct TensorF64 : IDisposable
{
    private readonly MlxArrayHandle handle;

    internal TensorF64(MlxArrayHandle handle) => this.handle = handle;

    public ShapeView Shape => new(this.Borrow());

    public int Rank => TensorRuntime.Rank(this.Borrow());

    public int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public TensorF64 T => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public TensorF64 Abs => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Abs));

    public TensorF64 Exp => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Exp));

    public TensorF64 Log => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Log));

    public TensorF64 Sqrt => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Sqrt));

    public TensorF64MeanProxy Mean => new(this);

    public TensorF64MeanKeepProxy MeanKeep => new(this);

    public TensorF64SoftmaxProxy Softmax => new(this);

    public TensorF64 this[AxisSelector selector0] => new(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public TensorF64 this[AxisSelector selector0, AxisSelector selector1] => new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public TensorF64 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public TensorF64 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public void Dispose() => TensorRuntime.Dispose(this.handle);

    internal readonly MlxArrayHandle Borrow() => this.handle.ctx != 0 ? this.handle : throw new ObjectDisposedException(nameof(TensorF64));

    internal TensorF64 ApplySum(ReductionSpec spec) => new(
        spec.IsRange
            ? TensorRuntime.Reduction(this.Borrow(), spec.Axes, ReductionKind.Sum, spec.KeepDims)
            : TensorRuntime.Reduction(this.Borrow(), spec.Axis, ReductionKind.Sum, spec.KeepDims));

    internal readonly TensorF64 ApplyMean(Index axis, bool keepDims) =>
        new(TensorRuntime.Reduction(this.Borrow(), axis, ReductionKind.Mean, keepDims));

    internal readonly TensorF64 ApplyMean(Range axes, bool keepDims) =>
        new(TensorRuntime.Reduction(this.Borrow(), axes, ReductionKind.Mean, keepDims));

    internal readonly TensorF64 ApplySoftmax(Index axis) => new(TensorRuntime.Softmax(this.Borrow(), axis));

    public TensorF64 Mm(TensorF64 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Matmul));

    public TensorF64 Max(TensorF64 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Maximum));

    public TensorF64 Max(double value) => new(TensorRuntime.ScalarMaximum(this.Borrow(), value));

    public static TensorF64 operator +(TensorF64 left, TensorF64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Add));

    public static TensorF64 operator -(TensorF64 left, TensorF64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Subtract));

    public static TensorF64 operator *(TensorF64 left, TensorF64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Multiply));

    public static TensorF64 operator /(TensorF64 left, TensorF64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Divide));

    public static TensorF64 operator -(TensorF64 value) => new(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.Negate));

    public static explicit operator double(TensorF64 value) => TensorRuntime.ReadFloat64(value.Borrow());

    public static explicit operator TensorF64(TensorBool value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat64));

    public static explicit operator TensorF64(TensorI32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat64));

    public static explicit operator TensorF64(TensorI64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat64));

    public static explicit operator TensorF64(TensorF16 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat64));

    public static explicit operator TensorF64(TensorF32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat64));
}
