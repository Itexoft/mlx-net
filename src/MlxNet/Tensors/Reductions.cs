// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Tensors;

public static class Reductions
{
    public static ReductionOperation Sum { get; } = new(keepDims: false);

    public static ReductionOperation SumKeep { get; } = new(keepDims: true);
}

public readonly struct ReductionOperation
{
    private readonly bool keepDims;

    internal ReductionOperation(bool keepDims) => this.keepDims = keepDims;

    public ReductionSpec this[Index axis] => new(this.keepDims, axis);

    public ReductionSpec this[Range axes] => new(this.keepDims, axes);
}

public readonly struct ReductionSpec
{
    private readonly Index axis;
    private readonly Range axes;

    internal ReductionSpec(bool keepDims, Index axis)
    {
        this.axis = axis;
        this.axes = default;
        this.KeepDims = keepDims;
        this.IsRange = false;
    }

    internal ReductionSpec(bool keepDims, Range axes)
    {
        this.axis = default;
        this.axes = axes;
        this.KeepDims = keepDims;
        this.IsRange = true;
    }

    internal bool KeepDims { get; }

    internal bool IsRange { get; }

    internal Index Axis => !this.IsRange ? this.axis : throw new InvalidOperationException("Reduction spec stores a range.");

    internal Range Axes => this.IsRange ? this.axes : throw new InvalidOperationException("Reduction spec stores a single axis.");

    public static TensorF16 operator +(ReductionSpec spec, TensorF16 tensor) => tensor.ApplySum(spec);

    public static TensorF32 operator +(ReductionSpec spec, TensorF32 tensor) => tensor.ApplySum(spec);

    public static TensorF64 operator +(ReductionSpec spec, TensorF64 tensor) => tensor.ApplySum(spec);

    public static TensorI32 operator +(ReductionSpec spec, TensorI32 tensor) => tensor.ApplySum(spec);

    public static TensorI64 operator +(ReductionSpec spec, TensorI64 tensor) => tensor.ApplySum(spec);
}
