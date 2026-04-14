// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly ref struct TensorF32 : IDisposable
{
    private readonly MlxArrayHandle handle;

    internal TensorF32(MlxArrayHandle handle) => this.handle = handle;

    public ShapeView Shape => new(this.Borrow());

    public int Rank => TensorRuntime.Rank(this.Borrow());

    public int Dim(int axis) => TensorRuntime.Dim(this.Borrow(), axis);

    public TensorF32 T => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.TransposeLastTwo));

    public TensorF32 Abs => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Abs));

    public TensorF32 Exp => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Exp));

    public TensorF32 Log => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Log));

    public TensorF32 Sqrt => new(TensorRuntime.Unary(this.Borrow(), UnaryTensorOp.Sqrt));

    public TensorF32MeanProxy Mean => new(this);

    public TensorF32MeanKeepProxy MeanKeep => new(this);

    public TensorF32SoftmaxProxy Softmax => new(this);

    public TensorF32 this[AxisSelector selector0] => new(TensorRuntime.Slice(this.Borrow(), [selector0]));

    public TensorF32 this[AxisSelector selector0, AxisSelector selector1] => new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1]));

    public TensorF32 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2]));

    public TensorF32 this[AxisSelector selector0, AxisSelector selector1, AxisSelector selector2, AxisSelector selector3] =>
        new(TensorRuntime.Slice(this.Borrow(), [selector0, selector1, selector2, selector3]));

    public void Dispose() => TensorRuntime.Dispose(this.handle);

    internal readonly MlxArrayHandle Borrow() => this.handle.ctx != 0 ? this.handle : throw new ObjectDisposedException(nameof(TensorF32));

    internal TensorF32 ApplySum(ReductionSpec spec) => new(
        spec.IsRange
            ? TensorRuntime.Reduction(this.Borrow(), spec.Axes, ReductionKind.Sum, spec.KeepDims)
            : TensorRuntime.Reduction(this.Borrow(), spec.Axis, ReductionKind.Sum, spec.KeepDims));

    internal readonly TensorF32 ApplyMean(Index axis, bool keepDims) =>
        new(TensorRuntime.Reduction(this.Borrow(), axis, ReductionKind.Mean, keepDims));

    internal readonly TensorF32 ApplyMean(Range axes, bool keepDims) =>
        new(TensorRuntime.Reduction(this.Borrow(), axes, ReductionKind.Mean, keepDims));

    internal readonly TensorF32 ApplySoftmax(Index axis) => new(TensorRuntime.Softmax(this.Borrow(), axis));

    public TensorF32 Mm(TensorF32 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Matmul));

    public TensorF32 Max(TensorF32 other) => new(TensorRuntime.Binary(this.Borrow(), other.Borrow(), BinaryTensorOp.Maximum));

    public TensorF32 Max(float value) => new(TensorRuntime.ScalarMaximum(this.Borrow(), value));

    public TensorF32 MeanPool(TensorF32 mask) => new(TensorRuntime.MeanPool(this.Borrow(), mask.Borrow()));

    public float[][] ReadVectors() => TensorRuntime.ReadVectorsFloat32(this.Borrow());

    public static TensorF32 operator +(TensorF32 left, TensorF32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Add));

    public static TensorF32 operator -(TensorF32 left, TensorF32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Subtract));

    public static TensorF32 operator *(TensorF32 left, TensorF32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Multiply));

    public static TensorF32 operator /(TensorF32 left, TensorF32 right) =>
        new(TensorRuntime.Binary(left.Borrow(), right.Borrow(), BinaryTensorOp.Divide));

    public static TensorF32 operator -(TensorF32 value) => new(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.Negate));

    public static explicit operator float(TensorF32 value) => TensorRuntime.ReadFloat32(value.Borrow());

    public static explicit operator TensorF32(TensorBool value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat32));

    public static explicit operator TensorF32(TensorI32 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat32));

    public static explicit operator TensorF32(TensorI64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat32));

    public static explicit operator TensorF32(TensorF16 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat32));

    public static explicit operator TensorF32(TensorF64 value) => new(TensorRuntime.Cast(value.Borrow(), MlxDType.MlxFloat32));
}
