// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Tensors;
using NUnit.Framework;
using static Itexoft.Tensors.Reductions;

[TestFixture]
public sealed class TensorLayerTests
{
    [Test]
    public void From_Int32_PreservesShape_And_Values()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var tensor = Tensor.From([1, 2, 3, 4], (2, 2));
        TestHelpers.EvalArray(tensor.Borrow());
        Assert.That(TestHelpers.ShapeOf(tensor.Borrow()), Is.EqualTo(new[] { 2, 2 }));
        Assert.That(TestHelpers.ToInt32(tensor.Borrow()), Is.EqualTo(new[] { 1, 2, 3, 4 }));
    }

    [Test]
    public void Explicit_Tensor_Cast_Preserves_Data()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var source = Tensor.From([1, 2, 3, 4], (2, 2));
        using var casted = (TensorF32)source;

        TestHelpers.EvalArray(casted.Borrow());
        Assert.That(TestHelpers.ShapeOf(casted.Borrow()), Is.EqualTo(new[] { 2, 2 }));
        Assert.That(TestHelpers.ToFloat32(casted.Borrow()), Is.EqualTo(new[] { 1f, 2f, 3f, 4f }).Within(1e-6f));
    }

    [Test]
    public void Scalar_Cast_Reads_Scalar_And_Rejects_NonScalar()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var scalar = Tensor.From([42f], 1);
        var value = (float)scalar;
        Assert.That(value, Is.EqualTo(42f).Within(1e-6f));

        using var nonScalar = Tensor.From([1f, 2f], (1, 2));

        try
        {
            _ = (float)nonScalar;
            Assert.Fail("Non-scalar tensor cast must fail.");
        }
        catch (InvalidOperationException) { }
    }

    [Test]
    public void Arithmetic_And_Mm_Work_For_Float32()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var left = Tensor.From([1f, 2f, 3f, 4f], (2, 2));
        using var right = Tensor.From([5f, 6f, 7f, 8f], (2, 2));
        using var sum = left + right;
        using var product = left * right;
        using var mm = left.Mm(right);

        AssertFloatTensor(sum, new[] { 2, 2 }, new[] { 6f, 8f, 10f, 12f });
        AssertFloatTensor(product, new[] { 2, 2 }, new[] { 5f, 12f, 21f, 32f });
        AssertFloatTensor(mm, new[] { 2, 2 }, new[] { 19f, 22f, 43f, 50f });
    }

    [Test]
    public void Sum_And_SumKeep_Work_For_Single_Axis_And_Range()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var tensor = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], (2, 2, 2));
        using var sumLast = Sum[^1] + tensor;
        using var sumKeepLast = SumKeep[^1] + tensor;
        using var sumLeading = Sum[..^1] + tensor;

        AssertFloatTensor(sumLast, new[] { 2, 2 }, new[] { 3f, 7f, 11f, 15f });
        AssertFloatTensor(sumKeepLast, new[] { 2, 2, 1 }, new[] { 3f, 7f, 11f, 15f });
        AssertFloatTensor(sumLeading, new[] { 2 }, new[] { 16f, 20f });
    }

    [Test]
    public void Indexer_Performs_Selection_And_Slicing()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var tensor = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f], (2, 3));
        using var row = tensor[1];
        using var tail = tensor[..1, 1..];

        AssertFloatTensor(row, new[] { 3 }, new[] { 4f, 5f, 6f });
        AssertFloatTensor(tail, new[] { 1, 2 }, new[] { 2f, 3f });
    }

    [Test]
    public void Mean_Softmax_MeanPool_And_ReadVectors_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var tensor = Tensor.From([1f, 3f, 5f, 7f], (2, 2));
        using var mean = tensor.Mean[^1];
        using var logits = Tensor.From([1f, 2f, 3f], (1, 3));
        using var softmax = logits.Softmax[^1];
        using var values = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f], (1, 3, 2));
        using var mask = Tensor.From([1f, 1f, 0f], (1, 3));
        using var pooled = values.MeanPool(mask);

        AssertFloatTensor(mean, new[] { 2 }, new[] { 2f, 6f });
        AssertFloatTensor(softmax, new[] { 1, 3 }, new[] { 0.09003057f, 0.24472847f, 0.66524096f }, 1e-5f);

        var vectors = pooled.ReadVectors();
        Assert.That(vectors, Has.Length.EqualTo(1));
        Assert.That(vectors[0], Is.EqualTo(new[] { 2f, 3f }).Within(1e-6f));
    }

    [Test]
    public void Bool_Operators_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var left = Tensor.From([true, false, true, false], (2, 2));
        using var right = Tensor.From([true, true, false, false], (2, 2));
        using var inverted = !left;
        using var merged = inverted | right;
        using var result = (TensorI32)merged;

        AssertIntTensor(result, new[] { 2, 2 }, new[] { 1, 1, 0, 1 });
    }

    private static void AssertFloatTensor(TensorF32 tensor, int[] expectedShape, float[] expected, float tolerance = 1e-6f)
    {
        TestHelpers.EvalArray(tensor.Borrow());
        Assert.That(TestHelpers.ShapeOf(tensor.Borrow()), Is.EqualTo(expectedShape));
        Assert.That(TestHelpers.ToFloat32(tensor.Borrow()), Is.EqualTo(expected).Within(tolerance));
    }

    private static void AssertIntTensor(TensorI32 tensor, int[] expectedShape, int[] expected)
    {
        TestHelpers.EvalArray(tensor.Borrow());
        Assert.That(TestHelpers.ShapeOf(tensor.Borrow()), Is.EqualTo(expectedShape));
        Assert.That(TestHelpers.ToInt32(tensor.Borrow()), Is.EqualTo(expected));
    }
}
