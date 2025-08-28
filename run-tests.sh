#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

test_projects=()
while IFS= read -r p; do
  test_projects+=("$p")
done < <(find "$SCRIPT_DIR/src" -type f \( -name '*Tests.csproj' -o -name '*.Tests.csproj' -o -name '*Test.csproj' -o -name '*.Test.csproj' \) | sort)


artifacts="$PWD/test-artifacts"
mkdir -p "$artifacts"
trap '{ [ -n "${GITHUB_OUTPUT:-}" ] && { echo "paths<<EOF" ; printf "%s\n" "$artifacts" ; echo "EOF" ; } >> "$GITHUB_OUTPUT" ; }' EXIT INT SIGTERM


if [ "${#test_projects[@]}" -gt 0 ]; then
  for p in "${test_projects[@]}"; do
    # name="$(basename "${p%.*}")"
    # export DYLD_PRINT_LIBRARIES=1
    # export DYLD_PRINT_LIBRARIES_POST_LAUNCH=1
    # export DYLD_PRINT_RPATHS=1
    # export DYLD_PRINT_INITIALIZERS=1
    # export DYLD_PRINT_SEARCHING=1
    # export DYLD_PRINT_OPTS=1
    # export DYLD_PRINT_ENV=1
    # export DOTNET_EnableDiagnostics=1
    # export COMPlus_ReadyToRun=0
    # export COREHOST_TRACE=1
    # export COREHOST_TRACEFILE="$artifacts/$name.corehost.log"
    # export COMPlus_EnableEventPipe=1
    # export COMPlus_EventPipeConfig="Microsoft-Windows-DotNETRuntime:0x4c14fccbd:5,Microsoft-DotNETCore-SampleProfiler:0x1:5"
    # export COMPlus_EventPipeOutputPath="$artifacts/$name.nettrace"
    # export MTL_DEBUG_LAYER=
    #ulimit -c unlimited || true
    dotnet test "$p" -c Release \
      --diag "$artifacts/$(basename "${p%.*}").vstest.diag.log" \
      --logger "trx;LogFileName=$(basename "${p%.*}").trx" \
      --results-directory "$artifacts" \
      --blame-crash --blame-hang --blame-hang-timeout 60s
  done
  exit 0
fi

echo "::error::No test projects or solutions found"
exit 1