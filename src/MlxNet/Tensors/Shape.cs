// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Tensors;

public readonly ref struct Shape
{
    private readonly ReadOnlySpan<int> dimensions;

    public Shape(ReadOnlySpan<int> dimensions)
    {
        if (dimensions.Length == 0)
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(dimensions));

        this.dimensions = dimensions;
    }

    public Shape(int d0) => this.dimensions = new[] { d0 };

    public Shape(int d0, int d1) => this.dimensions = new[] { d0, d1 };

    public Shape(int d0, int d1, int d2) => this.dimensions = new[] { d0, d1, d2 };

    public Shape(int d0, int d1, int d2, int d3) => this.dimensions = new[] { d0, d1, d2, d3 };

    public int Rank => this.dimensions.Length;

    public int this[int index] => this.dimensions[index];

    internal ReadOnlySpan<int> Dimensions => this.dimensions;

    public static implicit operator Shape(int d0) => new(d0);

    public static implicit operator Shape((int d0, int d1) value) => new(value.d0, value.d1);

    public static implicit operator Shape((int d0, int d1, int d2) value) => new(value.d0, value.d1, value.d2);

    public static implicit operator Shape((int d0, int d1, int d2, int d3) value) => new(value.d0, value.d1, value.d2, value.d3);
}
