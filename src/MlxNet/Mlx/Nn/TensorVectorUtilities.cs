// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;

namespace Itexoft.Mlx.Nn;

internal static class TensorVectorUtilities
{
    internal static MlxVectorArrayHandle Create(ReadOnlySpan<MlxArrayHandle> arrays)
    {
        var vector = MlxVector.ArrayNew();
        try
        {
            foreach (var array in arrays)
            {
                var status = MlxVector.ArrayAppendValue(vector, array);
                TensorUtilities.CheckStatus(status, "vector_array_append_value");
            }

            return vector;
        }
        catch
        {
            MlxVector.ArrayFree(vector);

            throw;
        }
    }

    internal static MlxVectorArrayHandle Create(params MlxArrayHandle[] arrays)
        => Create(arrays.AsSpan());

    internal static MlxArrayHandle[] Consume(MlxVectorArrayHandle vector)
    {
        var size = (int)MlxVector.ArraySize(vector);
        var result = new MlxArrayHandle[size];
        for (var i = 0; i < size; i++)
        {
            var status = MlxVector.ArrayGet(out result[i], vector, (nuint)i);
            TensorUtilities.CheckStatus(status, "vector_array_get");
        }

        MlxVector.ArrayFree(vector);

        return result;
    }
}