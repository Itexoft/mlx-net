// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Tensors;

public readonly struct AxisSelector
{
    private readonly Index index;
    private readonly Range range;

    private AxisSelector(Index index)
    {
        this.index = index;
        this.range = default;
        this.IsIndex = true;
    }

    private AxisSelector(Range range)
    {
        this.index = default;
        this.range = range;
        this.IsIndex = false;
    }

    internal bool IsIndex { get; }

    internal Index Index => this.IsIndex ? this.index : throw new InvalidOperationException("Selector does not contain a single index.");

    internal Range Range => !this.IsIndex ? this.range : throw new InvalidOperationException("Selector does not contain a range.");

    public static implicit operator AxisSelector(int index) => new(new Index(index));

    public static implicit operator AxisSelector(Index index) => new(index);

    public static implicit operator AxisSelector(Range range) => new(range);
}
