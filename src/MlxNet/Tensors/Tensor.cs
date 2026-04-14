// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Tensors;

public static class Tensor
{
    public static TensorBool From(ReadOnlySpan<bool> values, Shape shape) => new(TensorRuntime.Create(values, shape.Dimensions));

    public static TensorBool From(ReadOnlySpan<bool> values, int d0)
    {
        Span<int> shape = stackalloc int[1] { d0 };

        return new(TensorRuntime.Create(values, shape));
    }

    public static TensorBool From(ReadOnlySpan<bool> values, (int d0, int d1) shape)
    {
        Span<int> dimensions = stackalloc int[2] { shape.d0, shape.d1 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorBool From(ReadOnlySpan<bool> values, (int d0, int d1, int d2) shape)
    {
        Span<int> dimensions = stackalloc int[3] { shape.d0, shape.d1, shape.d2 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorBool From(ReadOnlySpan<bool> values, (int d0, int d1, int d2, int d3) shape)
    {
        Span<int> dimensions = stackalloc int[4] { shape.d0, shape.d1, shape.d2, shape.d3 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorI32 From(ReadOnlySpan<int> values, Shape shape) => new(TensorRuntime.Create(values, shape.Dimensions));

    public static TensorI32 From(ReadOnlySpan<int> values, int d0)
    {
        Span<int> shape = stackalloc int[1] { d0 };

        return new(TensorRuntime.Create(values, shape));
    }

    public static TensorI32 From(ReadOnlySpan<int> values, (int d0, int d1) shape)
    {
        Span<int> dimensions = stackalloc int[2] { shape.d0, shape.d1 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorI32 From(ReadOnlySpan<int> values, (int d0, int d1, int d2) shape)
    {
        Span<int> dimensions = stackalloc int[3] { shape.d0, shape.d1, shape.d2 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorI32 From(ReadOnlySpan<int> values, (int d0, int d1, int d2, int d3) shape)
    {
        Span<int> dimensions = stackalloc int[4] { shape.d0, shape.d1, shape.d2, shape.d3 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorI64 From(ReadOnlySpan<long> values, Shape shape) => new(TensorRuntime.Create(values, shape.Dimensions));

    public static TensorI64 From(ReadOnlySpan<long> values, int d0)
    {
        Span<int> shape = stackalloc int[1] { d0 };

        return new(TensorRuntime.Create(values, shape));
    }

    public static TensorI64 From(ReadOnlySpan<long> values, (int d0, int d1) shape)
    {
        Span<int> dimensions = stackalloc int[2] { shape.d0, shape.d1 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorI64 From(ReadOnlySpan<long> values, (int d0, int d1, int d2) shape)
    {
        Span<int> dimensions = stackalloc int[3] { shape.d0, shape.d1, shape.d2 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorI64 From(ReadOnlySpan<long> values, (int d0, int d1, int d2, int d3) shape)
    {
        Span<int> dimensions = stackalloc int[4] { shape.d0, shape.d1, shape.d2, shape.d3 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF16 From(ReadOnlySpan<Half> values, Shape shape) => new(TensorRuntime.Create(values, shape.Dimensions));

    public static TensorF16 From(ReadOnlySpan<Half> values, int d0)
    {
        Span<int> shape = stackalloc int[1] { d0 };

        return new(TensorRuntime.Create(values, shape));
    }

    public static TensorF16 From(ReadOnlySpan<Half> values, (int d0, int d1) shape)
    {
        Span<int> dimensions = stackalloc int[2] { shape.d0, shape.d1 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF16 From(ReadOnlySpan<Half> values, (int d0, int d1, int d2) shape)
    {
        Span<int> dimensions = stackalloc int[3] { shape.d0, shape.d1, shape.d2 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF16 From(ReadOnlySpan<Half> values, (int d0, int d1, int d2, int d3) shape)
    {
        Span<int> dimensions = stackalloc int[4] { shape.d0, shape.d1, shape.d2, shape.d3 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF32 From(ReadOnlySpan<float> values, Shape shape) => new(TensorRuntime.Create(values, shape.Dimensions));

    public static TensorF32 From(ReadOnlySpan<float> values, int d0)
    {
        Span<int> shape = stackalloc int[1] { d0 };

        return new(TensorRuntime.Create(values, shape));
    }

    public static TensorF32 From(ReadOnlySpan<float> values, (int d0, int d1) shape)
    {
        Span<int> dimensions = stackalloc int[2] { shape.d0, shape.d1 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF32 From(ReadOnlySpan<float> values, (int d0, int d1, int d2) shape)
    {
        Span<int> dimensions = stackalloc int[3] { shape.d0, shape.d1, shape.d2 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF32 From(ReadOnlySpan<float> values, (int d0, int d1, int d2, int d3) shape)
    {
        Span<int> dimensions = stackalloc int[4] { shape.d0, shape.d1, shape.d2, shape.d3 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF64 From(ReadOnlySpan<double> values, Shape shape) => new(TensorRuntime.Create(values, shape.Dimensions));

    public static TensorF64 From(ReadOnlySpan<double> values, int d0)
    {
        Span<int> shape = stackalloc int[1] { d0 };

        return new(TensorRuntime.Create(values, shape));
    }

    public static TensorF64 From(ReadOnlySpan<double> values, (int d0, int d1) shape)
    {
        Span<int> dimensions = stackalloc int[2] { shape.d0, shape.d1 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF64 From(ReadOnlySpan<double> values, (int d0, int d1, int d2) shape)
    {
        Span<int> dimensions = stackalloc int[3] { shape.d0, shape.d1, shape.d2 };

        return new(TensorRuntime.Create(values, dimensions));
    }

    public static TensorF64 From(ReadOnlySpan<double> values, (int d0, int d1, int d2, int d3) shape)
    {
        Span<int> dimensions = stackalloc int[4] { shape.d0, shape.d1, shape.d2, shape.d3 };

        return new(TensorRuntime.Create(values, dimensions));
    }
}
