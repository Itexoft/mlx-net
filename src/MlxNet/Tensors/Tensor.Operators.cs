// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using Itexoft.Tensors.Internal;

namespace Itexoft.Tensors;

public readonly ref partial struct Tensor
{
    public static Tensor operator +(Tensor left, Tensor right) => left.Binary(right, BinaryTensorOp.Add);

    public static Tensor operator -(Tensor left, Tensor right) => left.Binary(right, BinaryTensorOp.Subtract);

    public static Tensor operator *(Tensor left, Tensor right) => left.Binary(right, BinaryTensorOp.Multiply);

    public static Tensor operator /(Tensor left, Tensor right) => left.Binary(right, BinaryTensorOp.Divide);

    public static Tensor operator %(Tensor left, Tensor right) => left.Binary(right, BinaryTensorOp.Remainder);

    public static Tensor operator -(Tensor value) => CreateOwned(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.Negate));

    public static Tensor operator +(Tensor left, int right) => left.ScalarRight(right, BinaryTensorOp.Add);

    public static Tensor operator +(int left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Add);

    public static Tensor operator -(Tensor left, int right) => left.ScalarRight(right, BinaryTensorOp.Subtract);

    public static Tensor operator -(int left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Subtract);

    public static Tensor operator *(Tensor left, int right) => left.ScalarRight(right, BinaryTensorOp.Multiply);

    public static Tensor operator *(int left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Multiply);

    public static Tensor operator /(Tensor left, int right) => left.ScalarRight(right, BinaryTensorOp.Divide);

    public static Tensor operator /(int left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Divide);

    public static Tensor operator %(Tensor left, int right) => left.ScalarRight(right, BinaryTensorOp.Remainder);

    public static Tensor operator %(int left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Remainder);

    public static Tensor operator +(Tensor left, float right) => left.ScalarRight(right, BinaryTensorOp.Add);

    public static Tensor operator +(float left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Add);

    public static Tensor operator -(Tensor left, float right) => left.ScalarRight(right, BinaryTensorOp.Subtract);

    public static Tensor operator -(float left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Subtract);

    public static Tensor operator *(Tensor left, float right) => left.ScalarRight(right, BinaryTensorOp.Multiply);

    public static Tensor operator *(float left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Multiply);

    public static Tensor operator /(Tensor left, float right) => left.ScalarRight(right, BinaryTensorOp.Divide);

    public static Tensor operator /(float left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Divide);

    public static Tensor operator %(Tensor left, float right) => left.ScalarRight(right, BinaryTensorOp.Remainder);

    public static Tensor operator %(float left, Tensor right) => right.ScalarLeft(left, BinaryTensorOp.Remainder);

    public static Tensor operator ~(Tensor value)
    {
        value.RequireBoolTensor();

        return CreateOwned(TensorRuntime.Unary(value.Borrow(), UnaryTensorOp.LogicalNot));
    }

    public static Tensor operator !(Tensor value) => ~value;

    public static Tensor operator &(Tensor left, Tensor right)
    {
        left.RequireBoolTensor();
        right.RequireBoolTensor();

        return left.Binary(right, BinaryTensorOp.LogicalAnd);
    }

    public static Tensor operator |(Tensor left, Tensor right)
    {
        left.RequireBoolTensor();
        right.RequireBoolTensor();

        return left.Binary(right, BinaryTensorOp.LogicalOr);
    }

    public static Tensor operator ^(Tensor left, Tensor right)
    {
        left.RequireBoolTensor();
        right.RequireBoolTensor();

        return left.Binary(right, BinaryTensorOp.LogicalXor);
    }
}
