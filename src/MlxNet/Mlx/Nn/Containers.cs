// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Collections.Generic;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Composes multiple unary layers that are executed sequentially.
/// </summary>
public class Sequential : Module, IUnaryLayer
{
    private readonly List<IUnaryLayer> layers = [];

    public Sequential(IEnumerable<IUnaryLayer> layers)
    {
        var index = 0;

        foreach (var layer in layers)
        {
            this.layers.Add(layer);

            if (layer is Module module)
                this.RegisterModule(index.ToString(), module);

            index++;
        }
    }

    public Sequential(params IUnaryLayer[] layers) : this((IEnumerable<IUnaryLayer>)layers) { }

    public IReadOnlyList<IUnaryLayer> Layers => this.layers;

    public MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var current = input;

        foreach (var layer in this.layers)
            current = layer.Forward(current);

        return current;
    }
}
