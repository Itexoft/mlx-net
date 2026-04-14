// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly ref struct TensorF16 : IDisposable
{
    private readonly MlxArrayHandle handle;

    internal TensorF16(MlxArrayHandle handle) => this.handle = handle;

    public ShapeView Shape => new(this.Borrow());

    public int Rank => TensorRuntime.Rank(this.Borrow());

    public int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public TensorF16 T => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public TensorF16 Abs => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Abs));

    public TensorF16 Exp => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Exp));

    public TensorF16 Log => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Log));

    public TensorF16 Sqrt => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Sqrt));

    public TensorF16MeanProxy Mean => new(this);

    public TensorF16MeanKeepProxy MeanKeep => new(this);

    public TensorF16SoftmaxProxy Softmax => new(this);

    public TensorF16 this[AxisSelector selector0] => new(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public TensorF16 this[AxisSelector selector0, AxisSelector selector1] => new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public TensorF16 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public TensorF16 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public void Dispose() => TensorRuntime.Dispose(this.handle);

    internal readonly MlxArrayHandle Borrow() => this.handle.ctx != 0 ? this.handle : throw new ObjectDisposedException(nameof(TensorF16));

    internal TensorF16 ApplySum(ReductionSpec spec) => new(
        spec.IsRange
            ? TensorRuntime.Reduction(this.Borrow(), spec.Axes, ReductionKind.Sum, spec.KeepDims)
            : TensorRuntime.Reduction(this.Borrow(), spec.Axis, ReductionKind.Sum, spec.KeepDims));

    internal readonly TensorF16 ApplyMean(Index axis, bool keepDims) =>
        new(TensorRuntime.Reduction(this.Borrow(), axis, ReductionKind.Mean, keepDims));

    internal readonly TensorF16 ApplyMean(Range axes, bool keepDims) =>
        new(TensorRuntime.Reduction(this.Borrow(), axes, ReductionKind.Mean, keepDims));

    internal readonly TensorF16 ApplySoftmax(Index axis) => new(TensorRuntime.Softmax(this.Borrow(), axis));

    public TensorF16 Mm(TensorF16 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Matmul));

    public TensorF16 Max(TensorF16 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Maximum));

    public TensorF16 Max(Half value) => new(TensorRuntime.ScalarMaximum(this.Borrow(), value));

    public static TensorF16 operator +(TensorF16 left, TensorF16 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Add));

    public static TensorF16 operator -(TensorF16 left, TensorF16 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Subtract));

    public static TensorF16 operator *(TensorF16 left, TensorF16 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Multiply));

    public static TensorF16 operator /(TensorF16 left, TensorF16 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Divide));

    public static TensorF16 operator -(TensorF16 value) => new(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.Negate));

    public static explicit operator Half(TensorF16 value) => TensorRuntime.ReadFloat16(value.Borrow());

    public static explicit operator TensorF16(TensorBool value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat16));

    public static explicit operator TensorF16(TensorI32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat16));

    public static explicit operator TensorF16(TensorI64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat16));

    public static explicit operator TensorF16(TensorF32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat16));

    public static explicit operator TensorF16(TensorF64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat16));
}
