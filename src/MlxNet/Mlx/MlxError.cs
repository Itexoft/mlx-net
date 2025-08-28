// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxError
{
    /// <summary>Installs a custom error handler callback invoked on MLX errors.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_set_error_handler")]
    public static partial void SetErrorHandler(
        delegate* unmanaged[Cdecl]<sbyte*, void*, void> handler,
        void* data,
        delegate* unmanaged[Cdecl]<void*, void> dtor
    );
}