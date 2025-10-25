#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

MACOSX_MIN_VER=$(tr -d '[:space:]' < "$SCRIPT_DIR/../../MACOSX.VERSION")

BIN_DIR="$SCRIPT_DIR/../../mlx-bin"
BUILD_DIR="$SCRIPT_DIR/../../build"
BUILD_CACHE_DIR="$SCRIPT_DIR/build-cache"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "macOS required for clang shim linking" >&2
  exit 1
fi

mkdir -p "$BUILD_CACHE_DIR"

CONFIGURATION="${Configuration:-Release}"
RUNTIME_IDENTIFIER="${RuntimeIdentifier:-osx-arm64}"
PUBLISH_DIR="${PUBLISH_DIR:-$BUILD_DIR}"

echo "PUBLISH_DIR=$PUBLISH_DIR"

SDKROOT_DEFAULT="$(/usr/bin/xcrun --show-sdk-path 2>/dev/null || true)"
if [ -n "$SDKROOT_DEFAULT" ]; then export SDKROOT="$SDKROOT_DEFAULT"; fi
if [ -z "${SDKROOT:-}" ] && [ -d /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk ]; then 
  export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";
fi

ExtraExports="$BUILD_CACHE_DIR/MLX.NativeAOT.exports"
# account for exported symbols whose C names already start with an underscore
/usr/bin/nm -gU -j \
  "$BIN_DIR/libmlx.a" \
  "$BIN_DIR/libmlxc.a" \
  "$BIN_DIR/libgguflib.a" 2>/dev/null \
| /usr/bin/grep -E '^_+mlx_|^_+mlxc_|^_+gguf_' \
| /usr/bin/sed -E 's/[[:space:]]+$//' \
| /usr/bin/sort -u > "$ExtraExports"

MACOSX_TARGET="arm64-apple-macos${MACOSX_MIN_VER%%.*}"

CLANG_REAL="$(/usr/bin/xcrun --find clang || command -v clang || echo /usr/bin/clang)"
CLANGXX_REAL="$(/usr/bin/xcrun --find clang++ || command -v clang++ || echo /usr/bin/clang++)"

clang++ -c -std=c++17 -fobjc-arc -fPIC -target "$MACOSX_TARGET" -mmacosx-version-min="$MACOSX_MIN_VER" ${SDKROOT:+-isysroot "$SDKROOT"} "$SCRIPT_DIR/metal_embed.mm" -o "$BUILD_CACHE_DIR/metal_embed.o"

export PROJECT_DIR="$SCRIPT_DIR"
export MACOSX_MIN_VER
export MACOSX_TARGET
export BIN_DIR
export BUILD_CACHE_DIR
export Configuration="$CONFIGURATION"
export RuntimeIdentifier="$RUNTIME_IDENTIFIER"
export ExtraExports
export CLANG_REAL
export CLANGXX_REAL
export SDKROOT

cat > "$BUILD_CACHE_DIR/clang-shim" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

real="$CLANG_REAL"; [ "${0##*/}" = clang++ ] && real="$CLANGXX_REAL"
argv=( "$@" )
for a in "${argv[@]}"; do 
  [ "$a" = -c ] && exec "$real" ${SDKROOT:+-isysroot "$SDKROOT"} "-mmacosx-version-min=$MACOSX_MIN_VER" "${argv[@]}"; 
done

asm=; need=
for a in "${argv[@]}"; do
  [ "$need" ] && { asm="${a##*/}"; need=; continue; }
  case $a in 
    -o) need=1 ;; 
    -o*) asm="${a#-o}"; asm="${asm##*/}" ;; 
  esac
done

extra=()
extra+=("-Wl,-force_load,$BIN_DIR/libmlx.a")
extra+=("-Wl,-force_load,$BIN_DIR/libmlxc.a")
extra+=("-Wl,-force_load,$BIN_DIR/libgguflib.a")
extra+=("-Wl,-map,$BUILD_CACHE_DIR/link.map")
extra+=("-lc++")
extra+=("-Wl,-framework,Metal")
extra+=("-Wl,-framework,Foundation")
extra+=("-Wl,-framework,Accelerate")
extra+=("-Wl,-install_name,$asm")
extra+=("-Wl,-exported_symbols_list,$ExtraExports")
extra+=("-mmacosx-version-min=$MACOSX_MIN_VER")
extra+=("-Wl,-sectcreate,__DATA,__mlx_metallib,$BIN_DIR/mlx.metallib")
extra+=("$BUILD_CACHE_DIR/metal_embed.o")
extra+=("--target=$MACOSX_TARGET")

[ "$Configuration" = "Release" ] && extra+=("-Wl,-dead_strip")
[ "$Configuration" = "Debug" ] && extra+=("-g")

filtered=()
for a in "${argv[@]}"; do
  case "$a" in
    -ldl) continue ;;
    --target=*) continue ;;
    -ld_classic) continue ;;
  esac
  filtered+=("$a")
done
printf '%q ' "$real" ${SDKROOT:+-isysroot "$SDKROOT"} "${filtered[@]}" "${extra[@]}" > "$BUILD_CACHE_DIR/last-clang-link.txt"
exec "$real" ${SDKROOT:+-isysroot "$SDKROOT"} "${filtered[@]}" "${extra[@]}"
EOF
chmod +x "$BUILD_CACHE_DIR/clang-shim"
ln -sf "$BUILD_CACHE_DIR/clang-shim" "$BUILD_CACHE_DIR/clang"
ln -sf "$BUILD_CACHE_DIR/clang-shim" "$BUILD_CACHE_DIR/clang++"


export PATH="$BUILD_CACHE_DIR:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

run_dotnet() { dotnet "$@"; }

run_dotnet --diagnostics publish "$SCRIPT_DIR" \
  -c "$CONFIGURATION" \
  -r "$RUNTIME_IDENTIFIER" \
  -o "$PUBLISH_DIR" \
  -p:SkipRunBuildSh=true \
  --no-build \
  -bl:"$BUILD_CACHE_DIR/msbuild.binlog" || true

if [ -f "$BUILD_CACHE_DIR/msbuild.binlog" ] && command -v dotnet >/dev/null 2>&1; then
  dotnet msbuild "$BUILD_CACHE_DIR/msbuild.binlog" -noconlog "-flp:v=diag;logfile=$BUILD_CACHE_DIR/msbuild.binlog.txt" || true
fi
