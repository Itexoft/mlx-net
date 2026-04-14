// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Represents a persistent non-trainable tensor that participates in a module's state.
/// </summary>
public sealed class ModuleBuffer : IDisposable
{
    private MlxArrayHandle value;

    internal ModuleBuffer(string name, MlxArrayHandle value)
    {
        this.Name = name;
        this.value = value;
    }

    /// <summary>
    /// Gets the hierarchical name of the buffer.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the underlying tensor handle.
    /// </summary>
    public MlxArrayHandle Value => this.value;

    /// <inheritdoc />
    public void Dispose()
    {
        if (!TensorUtilities.IsNull(this.value))
        {
            MlxArray.Free(this.value);
            this.value = default;
        }
    }

    /// <summary>
    /// Replaces the stored value and optionally disposes the current handle.
    /// </summary>
    public void SetValue(MlxArrayHandle value, bool disposeCurrent = true)
    {
        if (disposeCurrent && !TensorUtilities.IsNull(this.value))
            MlxArray.Free(this.value);

        this.value = value;
    }
}
