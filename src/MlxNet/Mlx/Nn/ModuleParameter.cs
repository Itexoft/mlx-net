// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Represents a named tensor parameter that belongs to a <see cref="Module"/>.
/// </summary>
public sealed class ModuleParameter : IDisposable
{
    private MlxArrayHandle _value;

    internal ModuleParameter(string name, MlxArrayHandle value, bool trainable)
    {
        this.Name = name;
        this._value = value;
        this.Trainable = trainable;
    }

    /// <summary>
    /// Gets the logical name of the parameter within its owning module hierarchy.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets or sets a flag indicating whether automatic differentiation should produce gradients for this parameter.
    /// </summary>
    public bool Trainable { get; internal set; }

    /// <summary>
    /// Gets the underlying MLX array handle for the parameter.
    /// </summary>
    public MlxArrayHandle Value => this._value;

    /// <summary>
    /// Replaces the current MLX array backing this parameter.
    /// </summary>
    /// <param name="handle">Handle to the new array. Ownership is transferred to the parameter.</param>
    /// <param name="disposeCurrent">
    /// When <c>true</c>, the previously owned handle is released via <see cref="MlxArray.Free(MlxArrayHandle)"/>.
    /// </param>
    public void SetValue(MlxArrayHandle handle, bool disposeCurrent = true)
    {
        if (disposeCurrent && !TensorUtilities.IsNull(this._value))
            MlxArray.Free(this._value);

        this._value = handle;
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