#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT SIGTERM

GHP="$TMP/gh-pick.sh" && curl -fsSL "https://raw.githubusercontent.com/Itexoft/devops/refs/heads/master/gh-pick.sh" -o "$GHP" && chmod +x "$GHP"

ARGS=("-c" "Release" "$SCRIPT_DIR/src/MlxNet/MlxNet.csproj" "-o" "$SCRIPT_DIR/nuget")

if [ -n "${P12_BASE64-}" ] && [ -n "${P12_BASE64// }" ]; then
  SNK="$TMP/strongname.snk"
  CCR=$("$GHP" "@master" "lib/cert-converter.sh")
  "$CCR" "$P12_BASE64" snk "$SNK"
  ARGS+=("/p:SignAssembly=true" "/p:PublicSign=false" "/p:AssemblyOriginatorKeyFile=$SNK" "--cert=$P12_BASE64")
fi

mkdir -p "$SCRIPT_DIR/nuget"

DSP=$("$GHP" "@master" "lib/dotnet-sign-pack.sh")
"$DSP" "${ARGS[@]}"