// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Itexoft.Mlx;
using Itexoft.Tensors.CompilerServices;
using Itexoft.Tensors;
using NUnit.Framework;
using static Itexoft.Tensors.Reductions;

[TestFixture]
public sealed class TensorCoreTests
{
    [Test]
    public void StaticFactories_Arithmetic_And_ScalarCasts_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var left = Tensor.From([1f, 2f, 3f, 4f], (2, 2));
        var right = Tensor.From([5f, 6f, 7f, 8f], (2, 2));
        var sum = left + right;
        var product = left * right;
        var remainder = right % left;
        var shifted = left + 2f;
        var scaled = 2 * left;
        var mm = left.Matmul(right);
        var scalar = Tensor.Scalar(42f);
        var nonScalar = Tensor.From([42f], 1);

        AssertFloatTensor(sum, new[] { 2, 2 }, new[] { 6f, 8f, 10f, 12f });
        AssertFloatTensor(product, new[] { 2, 2 }, new[] { 5f, 12f, 21f, 32f });
        AssertFloatTensor(remainder, new[] { 2, 2 }, new[] { 0f, 0f, 1f, 0f });
        AssertFloatTensor(shifted, new[] { 2, 2 }, new[] { 3f, 4f, 5f, 6f });
        AssertFloatTensor(scaled, new[] { 2, 2 }, new[] { 2f, 4f, 6f, 8f });
        AssertFloatTensor(mm, new[] { 2, 2 }, new[] { 19f, 22f, 43f, 50f });
        Assert.That((float)scalar, Is.EqualTo(42f).Within(1e-6f));
        Assert.That(nonScalar.Rank, Is.EqualTo(1));
    }

    [Test]
    public void Reductions_Indexing_And_ShapeMethods_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var values = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], (2, 2, 2));
        var sumLast = Sum[^1] + values;
        var sumKeepLast = SumKeep[^1] + values;
        var matrix = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f], (2, 3));
        var row = matrix[1];
        var tail = matrix[..1, 1..];
        var moved = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f], (1, 2, 3)).MoveAxis(1, 2);
        var expanded = matrix.ExpandDims(0);
        var broadcast = expanded.BroadcastTo([4, 2, 3]);
        var transposed = broadcast.Transpose([1, 0, 2]);
        var squeezed = Tensor.From([1f, 2f, 3f], (1, 3, 1)).Squeeze([0, 2]);

        AssertFloatTensor(sumLast, new[] { 2, 2 }, new[] { 3f, 7f, 11f, 15f });
        AssertFloatTensor(sumKeepLast, new[] { 2, 2, 1 }, new[] { 3f, 7f, 11f, 15f });
        AssertFloatTensor(row, new[] { 3 }, new[] { 4f, 5f, 6f });
        AssertFloatTensor(tail, new[] { 1, 2 }, new[] { 2f, 3f });
        AssertFloatTensor(moved, new[] { 1, 3, 2 }, new[] { 1f, 4f, 2f, 5f, 3f, 6f });
        AssertShape(expanded, 1, 2, 3);
        AssertShape(broadcast, 4, 2, 3);
        AssertShape(transposed, 2, 4, 3);
        AssertFloatTensor(squeezed, new[] { 3 }, new[] { 1f, 2f, 3f });
    }

    [Test]
    public void Softmax_Where_TakeAlong_And_MeanPool_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var rows = Tensor.Arange(0, 3).ExpandDims(1);
        var cols = Tensor.Arange(0, 3).ExpandDims(0);
        var condition = rows.Lt(cols);
        var ones = Tensor.Ones((3, 3));
        var zeros = Tensor.Zeros((3, 3));
        var selected = condition.Where(ones, zeros);
        var logits = Tensor.From([1f, 2f, 3f], (1, 3));
        var softmax = logits.Softmax(^1);
        var values = Tensor.From([10f, 20f, 30f, 40f, 50f, 60f], (2, 3));
        var indices = Tensor.From([2, 1, 0, 0, 2, 1], (2, 3));
        var gathered = values.TakeAlong(indices, 1);
        var hidden = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f], (1, 3, 2));
        var mask = Tensor.From([1f, 1f, 0f], (1, 3));
        var pooled = hidden.MeanPool(mask);

        AssertFloatTensor(selected, new[] { 3, 3 }, new[] { 0f, 1f, 1f, 0f, 0f, 1f, 0f, 0f, 0f });
        AssertFloatTensor(softmax, new[] { 1, 3 }, new[] { 0.09003057f, 0.24472848f, 0.66524094f }, 1e-5f);
        AssertFloatTensor(gathered, new[] { 2, 3 }, new[] { 30f, 20f, 10f, 40f, 60f, 50f });

        var vectors = pooled.ReadVectors();
        Assert.That(vectors, Has.Length.EqualTo(1));
        Assert.That(vectors[0], Is.EqualTo(new[] { 2f, 3f }).Within(1e-6f));
    }

    [Test]
    public void UnaryElementwise_Copy_Pow_And_TransposeProperty_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var values = Tensor.From([-1f, 0f, 1f, 4f], (2, 2));
        var abs = values.Abs;
        var zeros = Tensor.Zeros((2, 2));
        var exp = zeros.Exp;
        var erf = zeros.Erf;
        var ones = Tensor.Ones((2, 2));
        var log = ones.Log;
        var squareRoots = Tensor.From([1f, 4f, 9f, 16f], (2, 2)).Sqrt;
        var tanh = zeros.Tanh;
        var copy = values.Copy();
        var squared = Tensor.From([1f, 2f, 3f, 4f], (2, 2)).Pow(2f);
        var transposed = Tensor.From([1f, 2f, 3f, 4f], (2, 2)).T;

        AssertFloatTensor(abs, new[] { 2, 2 }, new[] { 1f, 0f, 1f, 4f });
        AssertFloatTensor(exp, new[] { 2, 2 }, new[] { 1f, 1f, 1f, 1f });
        AssertFloatTensor(erf, new[] { 2, 2 }, new[] { 0f, 0f, 0f, 0f });
        AssertFloatTensor(log, new[] { 2, 2 }, new[] { 0f, 0f, 0f, 0f }, 1e-5f);
        AssertFloatTensor(squareRoots, new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f });
        AssertFloatTensor(tanh, new[] { 2, 2 }, new[] { 0f, 0f, 0f, 0f }, 1e-6f);
        AssertFloatTensor(copy, new[] { 2, 2 }, new[] { -1f, 0f, 1f, 4f });
        AssertFloatTensor(squared, new[] { 2, 2 }, new[] { 1f, 4f, 9f, 16f });
        AssertFloatTensor(transposed, new[] { 2, 2 }, new[] { 1f, 3f, 2f, 4f });
    }

    [Test]
    public void FloatScalar_WithIntegerTensor_PromotesInsteadOfTruncating()
    {
        TestHelpers.RequireNativeOrIgnore();

        var integers = Tensor.From([1, 2, 4, 9], (2, 2));
        var shifted = integers + 0.5f;
        var divided = integers / 0.5f;
        var roots = integers.Pow(0.5f);

        AssertFloatTensor(shifted, new[] { 2, 2 }, new[] { 1.5f, 2.5f, 4.5f, 9.5f });
        AssertFloatTensor(divided, new[] { 2, 2 }, new[] { 2f, 4f, 8f, 18f });
        AssertFloatTensor(roots, new[] { 2, 2 }, new[] { 1f, MathF.Sqrt(2f), 2f, 3f }, 1e-5f);
    }

    [Test]
    public void BoolOperators_And_Comparisons_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var left = Tensor.From([true, false, true, false], (2, 2));
        var right = Tensor.From([true, true, false, false], (2, 2));
        var inverted = ~left;
        var merged = inverted | right;
        var xor = left ^ right;
        var numbers = Tensor.From([1f, 2f, 3f, 4f], (2, 2));
        var bounds = Tensor.From([2f, 2f, 2f, 2f], (2, 2));
        var greater = numbers.Gt(bounds).Cast(MlxDType.MlxInt32);
        var equal = numbers.Eq(bounds).Cast(MlxDType.MlxInt32);

        AssertIntTensor(merged.Cast(MlxDType.MlxInt32), new[] { 2, 2 }, new[] { 1, 1, 0, 1 });
        AssertIntTensor(xor.Cast(MlxDType.MlxInt32), new[] { 2, 2 }, new[] { 0, 1, 1, 0 });
        AssertIntTensor(greater, new[] { 2, 2 }, new[] { 0, 0, 1, 1 });
        AssertIntTensor(equal, new[] { 2, 2 }, new[] { 0, 1, 0, 0 });
    }

    [Test]
    public void Wrap_DoesNotStealRawHandle_And_CopyByValue_IsSafe()
    {
        TestHelpers.RequireNativeOrIgnore();

        var raw = CreateFloatArray([1f, 2f, 3f, 4f], [2, 2]);

        try
        {
            AssertWrapBorrowedSemantics(raw);
        }
        finally
        {
            if (raw.ctx != 0)
                MlxArray.Free(raw);
        }
    }

    [Test]
    public void Copy_CreatesIndependentTensorValue()
    {
        TestHelpers.RequireNativeOrIgnore();

        var source = Tensor.From([1f, 2f, 3f, 4f], (2, 2));
        var copy = source.Copy();

        AssertFloatTensor(copy, new[] { 2, 2 }, new[] { 1f, 2f, 3f, 4f });
        AssertFloatTensor(source + copy, new[] { 2, 2 }, new[] { 2f, 4f, 6f, 8f });
    }

    [Test]
    public void PublicTensorSurface_ExposesOnlyStructs_And_NoOwnershipApi()
    {
        var assembly = typeof(MlxArrayHandle).Assembly;
        var tensorType = assembly.GetType("Itexoft.Tensors.Tensor", true)!;
        var exported = new List<Type>();
        foreach (var type in assembly.GetExportedTypes())
        {
            if (type.Namespace is null)
                continue;

            if (!type.Namespace.StartsWith("Itexoft.Tensors", StringComparison.Ordinal))
                continue;

            if (type.Namespace == "Itexoft.Tensors.CompilerServices")
                continue;

            exported.Add(type);
        }

        Assert.That(exported, Is.Not.Empty);

        var allAreStructs = true;
        foreach (var type in exported)
        {
            if (type.IsClass || type.IsInterface)
            {
                allAreStructs = false;
                break;
            }
        }

        Assert.That(allAreStructs, Is.True);
        Assert.That(tensorType.IsByRefLike, Is.True);

        var members = tensorType.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static);
        var names = new string[members.Length];
        for (var i = 0; i < members.Length; i++)
            names[i] = members[i].Name;

        Assert.That(names, Does.Not.Contain("Dispose"));
        Assert.That(names, Does.Not.Contain("Borrow"));
        Assert.That(names, Does.Not.Contain("Handle"));
        Assert.That(names, Does.Not.Contain("Owner"));
        Assert.That(names, Does.Not.Contain("Alias"));
        Assert.That(names, Does.Not.Contain("Detach"));
        Assert.That(names, Does.Not.Contain("Adopt"));
    }

    [Test]
    public void PublicAssembly_DoesNotExpose_MlxNnNamespace()
    {
        var assembly = typeof(MlxArrayHandle).Assembly;
        var exported = assembly.GetExportedTypes();

        var hasNnTypes = false;
        foreach (var type in exported)
        {
            if (type.Namespace == "Itexoft.Mlx.Nn"
                || (type.Namespace is not null
                    && type.Namespace.StartsWith("Itexoft.Mlx.Nn.", StringComparison.Ordinal)))
            {
                hasNnTypes = true;
                break;
            }
        }

        Assert.That(hasNnTypes, Is.False);
        Assert.That(assembly.GetType("Itexoft.Mlx.Nn.Module", false), Is.Null);
        Assert.That(assembly.GetType("Itexoft.Mlx.Nn.IUnaryLayer", false), Is.Null);
        Assert.That(assembly.GetType("Itexoft.Mlx.Nn.ValueAndGrad", false), Is.Null);
    }

    private static unsafe MlxArrayHandle CreateFloatArray(float[] values, int[] shape)
    {
        fixed (float* data = values)
        fixed (int* dims = shape)
            return MlxArray.NewData(data, dims, shape.Length, MlxDType.MlxFloat32);
    }

    private static void AssertWrapBorrowedSemantics(MlxArrayHandle raw)
    {
        var wrappedA = TensorCompiler.WrapBorrowed(raw);
        var wrappedB = TensorCompiler.WrapBorrowed(raw);
        var copiedValue = wrappedA;
        var doubled = copiedValue + wrappedB;

        AssertFloatTensor(doubled, new[] { 2, 2 }, new[] { 2f, 4f, 6f, 8f });

        TestHelpers.EvalArray(raw);
        Assert.That(TestHelpers.ShapeOf(raw), Is.EqualTo(new[] { 2, 2 }));
        Assert.That(TestHelpers.ToFloat32(raw), Is.EqualTo(new[] { 1f, 2f, 3f, 4f }).Within(1e-6f));
    }

    private static void AssertShape(Tensor tensor, params int[] expected)
    {
        Assert.That(tensor.Rank, Is.EqualTo(expected.Length));
        Assert.That(tensor.Shape.Rank, Is.EqualTo(expected.Length));

        for (var i = 0; i < expected.Length; i++)
            Assert.That(tensor.Shape[i], Is.EqualTo(expected[i]));
    }

    private static void AssertFloatTensor(Tensor tensor, int[] expectedShape, float[] expected, float tolerance = 1e-6f)
    {
        AssertShape(tensor, expectedShape);
        Assert.That(ReadFlatFloat(tensor), Is.EqualTo(expected).Within(tolerance));
    }

    private static void AssertIntTensor(Tensor tensor, int[] expectedShape, int[] expected)
    {
        AssertShape(tensor, expectedShape);
        Assert.That(ReadFlatInt(tensor), Is.EqualTo(expected));
    }

    private static float[] ReadFlatFloat(Tensor tensor)
    {
        tensor.Eval();
        var result = new List<float>();
        CollectFloats(tensor, result);

        return result.ToArray();
    }

    private static int[] ReadFlatInt(Tensor tensor)
    {
        tensor.Eval();
        var result = new List<int>();
        CollectInts(tensor, result);

        return result.ToArray();
    }

    private static void CollectFloats(Tensor tensor, List<float> values)
    {
        if (tensor.Rank == 0)
        {
            values.Add((float)tensor);

            return;
        }

        for (var i = 0; i < tensor.Dim(0); i++)
            CollectFloats(tensor[i], values);
    }

    private static void CollectInts(Tensor tensor, List<int> values)
    {
        if (tensor.Rank == 0)
        {
            values.Add((int)tensor);

            return;
        }

        for (var i = 0; i < tensor.Dim(0); i++)
            CollectInts(tensor[i], values);
    }
}
