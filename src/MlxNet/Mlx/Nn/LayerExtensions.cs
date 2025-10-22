// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Convenience helpers for working with <see cref="IUnaryLayer"/>.
/// </summary>
public static class LayerExtensions
{
    /// <summary>
    /// Invokes the unary layer.
    /// </summary>
    public static MlxArrayHandle Invoke(this IUnaryLayer layer, MlxArrayHandle input)
        => layer.Forward(input);
}