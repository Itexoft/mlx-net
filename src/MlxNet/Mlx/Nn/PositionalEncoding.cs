// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Rotary positional encoding as described in RoFormer.
/// </summary>
public sealed class Rope(int dimensions, bool traditional = false, float @base = 10_000f, float scale = 1f)
    : Module, IUnaryLayer
{
    public MlxArrayHandle Forward(MlxArrayHandle input) => this.Forward(input, 0);

    public MlxArrayHandle Forward(MlxArrayHandle input, int offset)
    {
        var shape = input.Shape();

        if (shape.Length < 2)
            throw new ArgumentException("RoPE expects the last two dimensions to represent sequence and features.");

        var seq = shape[^2];
        var feature = shape[^1];
        var batch = 1;
        for (var i = 0; i < shape.Length - 2; i++)
            batch *= shape[i];

        var reshaped = input.Reshape(batch, seq, feature);
        var optionalBase = new MlxOptionalFloat { value = @base, has_value = 1 };

        var status = MlxFast.Rope(
            out var rope,
            reshaped,
            dimensions,
            traditional,
            optionalBase,
            scale,
            offset,
            default,
            TensorUtilities.DefaultStream());
        TensorUtilities.CheckStatus(status, "rope");

        var restored = rope.Reshape(shape);

        MlxArray.Free(rope);
        MlxArray.Free(reshaped);

        return restored;
    }
}