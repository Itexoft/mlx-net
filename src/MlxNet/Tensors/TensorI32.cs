// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly ref struct TensorI32 : IDisposable
{
    private readonly MlxArrayHandle handle;

    internal TensorI32(MlxArrayHandle handle) => this.handle = handle;

    public ShapeView Shape => new(this.Borrow());

    public int Rank => TensorRuntime.Rank(this.Borrow());

    public int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public TensorI32 T => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public TensorI32 Abs => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Abs));

    public TensorI32 this[AxisSelector selector0] => new(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public TensorI32 this[AxisSelector selector0, AxisSelector selector1] => new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public TensorI32 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public TensorI32 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public void Dispose() => TensorRuntime.Dispose(this.handle);

    internal MlxArrayHandle Borrow() => this.handle.ctx != 0 ? this.handle : throw new ObjectDisposedException(nameof(TensorI32));

    internal TensorI32 ApplySum(ReductionSpec spec) => new(
        spec.IsRange
            ? TensorRuntime.Reduction(this.Borrow(), spec.Axes, ReductionKind.Sum, spec.KeepDims)
            : TensorRuntime.Reduction(this.Borrow(), spec.Axis, ReductionKind.Sum, spec.KeepDims));

    public TensorI32 Mm(TensorI32 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Matmul));

    public TensorI32 Max(TensorI32 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Maximum));

    public TensorI32 Max(int value) => new(TensorRuntime.ScalarMaximum(this.Borrow(), value));

    public static TensorI32 operator +(TensorI32 left, TensorI32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Add));

    public static TensorI32 operator -(TensorI32 left, TensorI32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Subtract));

    public static TensorI32 operator *(TensorI32 left, TensorI32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Multiply));

    public static TensorI32 operator /(TensorI32 left, TensorI32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Divide));

    public static TensorI32 operator -(TensorI32 value) => new(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.Negate));

    public static explicit operator int(TensorI32 value) => TensorRuntime.ReadInt32(value.Borrow());

    public static explicit operator TensorI32(TensorBool value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt32));

    public static explicit operator TensorI32(TensorI64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt32));

    public static explicit operator TensorI32(TensorF16 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt32));

    public static explicit operator TensorI32(TensorF32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt32));

    public static explicit operator TensorI32(TensorF64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxInt32));
}
