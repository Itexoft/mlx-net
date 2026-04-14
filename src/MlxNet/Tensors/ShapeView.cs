// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Tensors;

public readonly unsafe ref struct ShapeView
{
    private readonly int* shape;

    internal ShapeView(MlxArrayHandle handle)
    {
        this.Rank = checked((int)MlxArray.Ndim(handle));
        this.shape = MlxArray.Shape(handle);
    }

    public int Rank { get; }

    public int this[int index]
    {
        get
        {
            if ((uint)index >= (uint)this.Rank)
                throw new ArgumentOutOfRangeException(nameof(index));

            return this.shape[index];
        }
    }
}
