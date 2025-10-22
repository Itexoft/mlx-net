// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Represents a scalar or pair of integers used in convolution- and pooling-related APIs.
/// </summary>
public readonly struct IntPair(int first, int second)
{
    public int First { get; } = first;

    public int Second { get; } = second;

    public int[] Values => [this.First, this.Second];

    public static implicit operator IntPair(int value) => new(value, value);

    public static implicit operator IntPair((int first, int second) value) => new(value.first, value.second);

    public void Deconstruct(out int first, out int second)
    {
        first = this.First;
        second = this.Second;
    }
}

/// <summary>
/// Represents a scalar or triple of integers used in convolution- and pooling-related APIs.
/// </summary>
public readonly struct IntTriple(int first, int second, int third)
{
    public int First { get; } = first;

    public int Second { get; } = second;

    public int Third { get; } = third;

    public int[] Values => [this.First, this.Second, this.Third];

    public static implicit operator IntTriple(int value) => new(value, value, value);

    public static implicit operator IntTriple((int first, int second, int third) value)
        => new(value.first, value.second, value.third);

    public void Deconstruct(out int first, out int second, out int third)
    {
        first = this.First;
        second = this.Second;
        third = this.Third;
    }
}

/// <summary>
/// Represents either a single float or an array of floats.
/// </summary>
public readonly struct FloatOrArray
{
    private readonly float _value;
    private readonly float[]? _values;

    public FloatOrArray(float value)
    {
        this._value = value;
        this._values = null;
    }

    public FloatOrArray(float[] values)
    {
        this._value = 0f;
        this._values = values ?? throw new ArgumentNullException(nameof(values));
    }

    public FloatOrArray(ReadOnlySpan<float> values)
    {
        this._value = 0f;
        this._values = values.ToArray();
    }

    public int Count => this._values?.Length ?? 1;

    public float[] AsArray(int dimensions)
    {
        if (this._values is null)
        {
            var result = new float[dimensions];
            Array.Fill(result, this._value);

            return result;
        }

        if (this._values.Length != dimensions)
            throw new ArgumentException("Scale factor count does not match dimensionality.");

        return this._values;
    }

    public static implicit operator FloatOrArray(float value) => new(value);

    public static implicit operator FloatOrArray(float[] values) => new(values);

    public static implicit operator FloatOrArray(ReadOnlySpan<float> values) => new(values);
}

/// <summary>
/// Quantization strategies supported by high-level layers.
/// </summary>
public enum QuantizationMode
{
    Affine,
    Mxfp4
}