// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly ref struct TensorBool : IDisposable
{
    private readonly MlxArrayHandle handle;

    internal TensorBool(MlxArrayHandle handle) => this.handle = handle;

    public ShapeView Shape => new(this.Borrow());

    public int Rank => TensorRuntime.Rank(this.Borrow());

    public int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public TensorBool T => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public TensorBool this[AxisSelector selector0] => new(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public TensorBool this[AxisSelector selector0, AxisSelector selector1] => new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public TensorBool this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public TensorBool this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public void Dispose() => TensorRuntime.Dispose(this.handle);

    internal MlxArrayHandle Borrow() => this.handle.ctx != 0 ? this.handle : throw new ObjectDisposedException(nameof(TensorBool));

    public static TensorBool operator !(TensorBool value) => new(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.LogicalNot));

    public static TensorBool operator &(TensorBool left, TensorBool right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.LogicalAnd));

    public static TensorBool operator |(TensorBool left, TensorBool right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.LogicalOr));

    public static explicit operator bool(TensorBool value) => TensorRuntime.ReadBool(value.Borrow());

    public static explicit operator TensorBool(TensorI32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxBool));

    public static explicit operator TensorBool(TensorI64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxBool));

    public static explicit operator TensorBool(TensorF16 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxBool));

    public static explicit operator TensorBool(TensorF32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxBool));

    public static explicit operator TensorBool(TensorF64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxBool));
}
