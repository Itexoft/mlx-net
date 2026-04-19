// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.ComponentModel;
using Itexoft.Mlx;
using Itexoft.Tensors.Internal;

namespace Itexoft.Tensors.CompilerServices;

[EditorBrowsable(EditorBrowsableState.Never)]
public static class TensorCompiler
{
    public static Tensor AdoptOwned(MlxArrayHandle handle) => Tensor.AdoptOwned(handle);

    public static Tensor WrapBorrowed(MlxArrayHandle handle) => Tensor.AdoptOwned(TensorRuntime.RetainHandle(handle));

    public static MlxArrayHandle Borrow(scoped ref Tensor tensor) => tensor.Borrow();

    public static MlxArrayHandle RetainBorrowed(Tensor tensor) => TensorRuntime.RetainHandle(tensor.Borrow());

    public static MlxArrayHandle TakeOwned(ref Tensor tensor)
    {
        var moved = tensor;
        tensor = default;

        return moved.IsAlive ? moved.Borrow() : default;
    }

    public static void Release(ref Tensor tensor)
    {
        if (!tensor.IsAlive)
            return;

        TensorRuntime.DisposeHandle(tensor.Borrow());
        tensor = default;
    }
}
