// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Tensors;

public readonly struct Shape
{
    private readonly int d0;
    private readonly int d1;
    private readonly int d2;
    private readonly int d3;
    private readonly int[]? dimensions;

    public Shape(scoped ReadOnlySpan<int> dimensions)
    {
        if (dimensions.Length == 0)
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(dimensions));

        this.Rank = dimensions.Length;

        ValidateDimensions(dimensions);

        if (dimensions.Length <= 4)
        {
            this.d0 = dimensions.Length > 0 ? dimensions[0] : 0;
            this.d1 = dimensions.Length > 1 ? dimensions[1] : 0;
            this.d2 = dimensions.Length > 2 ? dimensions[2] : 0;
            this.d3 = dimensions.Length > 3 ? dimensions[3] : 0;
            this.dimensions = null;
        }
        else
        {
            this.d0 = 0;
            this.d1 = 0;
            this.d2 = 0;
            this.d3 = 0;
            this.dimensions = dimensions.ToArray();
        }
    }

    public Shape(int d0)
    {
        ValidateDimension(d0);

        this.Rank = 1;
        this.d0 = d0;
        this.d1 = 0;
        this.d2 = 0;
        this.d3 = 0;
        this.dimensions = null;
    }

    public Shape(int d0, int d1)
    {
        ValidateDimension(d0);
        ValidateDimension(d1);

        this.Rank = 2;
        this.d0 = d0;
        this.d1 = d1;
        this.d2 = 0;
        this.d3 = 0;
        this.dimensions = null;
    }

    public Shape(int d0, int d1, int d2)
    {
        ValidateDimension(d0);
        ValidateDimension(d1);
        ValidateDimension(d2);

        this.Rank = 3;
        this.d0 = d0;
        this.d1 = d1;
        this.d2 = d2;
        this.d3 = 0;
        this.dimensions = null;
    }

    public Shape(int d0, int d1, int d2, int d3)
    {
        ValidateDimension(d0);
        ValidateDimension(d1);
        ValidateDimension(d2);
        ValidateDimension(d3);

        this.Rank = 4;
        this.d0 = d0;
        this.d1 = d1;
        this.d2 = d2;
        this.d3 = d3;
        this.dimensions = null;
    }

    public int Rank { get; }

    public int this[int index]
    {
        get
        {
            if ((uint)index >= (uint)this.Rank)
                throw new ArgumentOutOfRangeException(nameof(index));

            if (this.dimensions is { } dimensions)
                return dimensions[index];

            return index switch
            {
                0 => this.d0,
                1 => this.d1,
                2 => this.d2,
                3 => this.d3,
                _ => throw new ArgumentOutOfRangeException(nameof(index)),
            };
        }
    }

    internal void CopyTo(Span<int> destination)
    {
        if (destination.Length < this.Rank)
            throw new ArgumentException("Destination span is shorter than the shape rank.", nameof(destination));

        if (this.dimensions is { } dimensions)
        {
            dimensions.AsSpan().CopyTo(destination);

            return;
        }

        if (this.Rank > 0)
            destination[0] = this.d0;

        if (this.Rank > 1)
            destination[1] = this.d1;

        if (this.Rank > 2)
            destination[2] = this.d2;

        if (this.Rank > 3)
            destination[3] = this.d3;
    }

    public static implicit operator Shape(int d0) => new(d0);

    public static implicit operator Shape((int d0, int d1) value) => new(value.d0, value.d1);

    public static implicit operator Shape((int d0, int d1, int d2) value) => new(value.d0, value.d1, value.d2);

    public static implicit operator Shape((int d0, int d1, int d2, int d3) value) => new(value.d0, value.d1, value.d2, value.d3);

    private static void ValidateDimensions(ReadOnlySpan<int> dimensions)
    {
        foreach (var dimension in dimensions)
            ValidateDimension(dimension);
    }

    private static void ValidateDimension(int dimension)
    {
        if (dimension < 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), "Shape dimensions cannot be negative.");
    }
}
