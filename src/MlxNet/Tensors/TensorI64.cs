// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly ref struct TensorI64
{
    private readonly MlxArrayHandle handle;

    internal TensorI64(MlxArrayHandle handle) => this.handle = handle;

    public ShapeView Shape => new(this.Borrow());

    public int Rank => TensorRuntime.Rank(this.Borrow());

    public int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public TensorI64 T => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public TensorI64 Abs => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Abs));

    public TensorI64 this[AxisSelector selector0] => new(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public TensorI64 this[AxisSelector selector0, AxisSelector selector1] => new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public TensorI64 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public TensorI64 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public void Dispose() => TensorRuntime.Dispose(this.handle);

    internal MlxArrayHandle Borrow() => this.handle.ctx != 0 ? this.handle : throw new ObjectDisposedException(nameof(TensorI64));

    internal TensorI64 ApplySum(ReductionSpec spec) => new(
        spec.IsRange
            ? TensorRuntime.Reduction(this.Borrow(), spec.Axes, ReductionKind.Sum, spec.KeepDims)
            : TensorRuntime.Reduction(this.Borrow(), spec.Axis, ReductionKind.Sum, spec.KeepDims));

    public TensorI64 Mm(TensorI64 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Matmul));

    public TensorI64 Max(TensorI64 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Maximum));

    public TensorI64 Max(long value) => new(TensorRuntime.ScalarMaximum(this.Borrow(), value));

    public static TensorI64 operator +(TensorI64 left, TensorI64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Add));

    public static TensorI64 operator -(TensorI64 left, TensorI64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Subtract));

    public static TensorI64 operator *(TensorI64 left, TensorI64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Multiply));

    public static TensorI64 operator /(TensorI64 left, TensorI64 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Divide));

    public static TensorI64 operator -(TensorI64 value) => new(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.Negate));

    public static explicit operator long(TensorI64 value) => TensorRuntime.ReadInt64(value.Borrow());

    public static explicit operator TensorI64(TensorBool value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt64));

    public static explicit operator TensorI64(TensorI32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt64));

    public static explicit operator TensorI64(TensorF16 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt64));

    public static explicit operator TensorI64(TensorF32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt64));

    public static explicit operator TensorI64(TensorF64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt64));
}
