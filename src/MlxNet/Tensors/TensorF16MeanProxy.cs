// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Tensors;

public readonly ref struct TensorF16MeanProxy
{
    private readonly TensorF16 tensor;

    internal TensorF16MeanProxy(TensorF16 tensor) => this.tensor = tensor;

    public TensorF16 this[Index axis] => this.tensor.ApplyMean(axis, false);

    public TensorF16 this[Range axes] => this.tensor.ApplyMean(axes, false);
}
