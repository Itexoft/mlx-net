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
    private MlxArrayHandle _value;

    internal ModuleBuffer(string name, MlxArrayHandle value)
    {
        this.Name = name;
        this._value = value;
    }

    /// <summary>
    /// Gets the hierarchical name of the buffer.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the underlying tensor handle.
    /// </summary>
    public MlxArrayHandle Value => this._value;

    /// <summary>
    /// Replaces the stored value and optionally disposes the current handle.
    /// </summary>
    public void SetValue(MlxArrayHandle value, bool disposeCurrent = true)
    {
        if (disposeCurrent && !TensorUtilities.IsNull(this._value))
            MlxArray.Free(this._value);

        this._value = value;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!TensorUtilities.IsNull(this._value))
        {
            MlxArray.Free(this._value);
            this._value = default;
        }
    }
}