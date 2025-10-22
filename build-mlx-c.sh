#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH

MLX_C_DIR="$SCRIPT_DIR/external/mlx-c"
BUILD_DIR="$SCRIPT_DIR/mlx-c-build"
BIN_DIR="$SCRIPT_DIR/mlx-bin"
MACOSX_VER_FILE="$SCRIPT_DIR/MACOSX.VERSION"
VERSION_FILE="$SCRIPT_DIR/VERSION"
EXPECTED_AR_COUNT=3

is_true() {
  case "$1" in
    1|true|TRUE|True|yes|YES|on|ON) return 0;;
    *) return 1;;
  esac
}

normalize_version_base() {
  local version="${1%%-*}"
  local major minor patch
  IFS='.' read -r major minor patch _ <<< "$version"
  major=${major:-0}
  minor=${minor:-0}
  patch=${patch:-0}
  printf '%s.%s.%s' "$((10#$major))" "$((10#$minor))" "$((10#$patch))"
}

ALLOW_VERSION_MISMATCH=${ALLOW_VERSION_MISMATCH:-false}
CMD=""
while [ $# -gt 0 ]; do
  case "$1" in
    upgrade)
      CMD="upgrade"
      ;;
    --allow-version-mismatch)
      ALLOW_VERSION_MISMATCH=true
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [ ! -f "$MLX_C_DIR/CMakeLists.txt" ]; then echo "CMakeLists.txt not found in $MLX_C_DIR" >&2; exit 1; fi
VER="$(sed -nE 's/.*GIT_TAG[[:space:]]+v([0-9]+(\.[0-9]+)*).*/\1/p' "$MLX_C_DIR/CMakeLists.txt" | head -n 1)"
if [ -z "$VER" ]; then echo "failed to extract MLX version from CMakeLists.txt" >&2; exit 1; fi

if [ ! -f "$MACOSX_VER_FILE" ]; then echo "MACOSX.VERSION not found" >&2; exit 1; fi
MACOSX_MIN_VER="$(tr -d '[:space:]' < "$MACOSX_VER_FILE")"
if [ -z "$MACOSX_MIN_VER" ]; then echo "MACOSX.VERSION is empty" >&2; exit 1; fi

if [ ! -f "$VERSION_FILE" ]; then echo "VERSION file not found: $VERSION_FILE; run with 'upgrade' to set" >&2; exit 1; fi
CUR_VER_FILE="$(tr -d '[:space:]' < "$VERSION_FILE")"

if [ -z "$CUR_VER_FILE" ] && [ "$CMD" != "upgrade" ]; then
  echo "VERSION file is empty; run with 'upgrade' to set" >&2
  exit 1
fi

if [ "$CMD" = "upgrade" ]; then
  CUR_VER="$VER"
else
  if [ "$(normalize_version_base "$CUR_VER_FILE")" != "$(normalize_version_base "$VER")" ]; then
    if is_true "$ALLOW_VERSION_MISMATCH"; then
      echo "warning: version mismatch allowed (VERSION=$CUR_VER_FILE, MLX_C=$VER)" >&2
    else
      echo "version mismatch: $CUR_VER_FILE != $VER; run with 'upgrade' to update or pass --allow-version-mismatch" >&2
      exit 1
    fi
  fi

  CUR_VER="$CUR_VER_FILE"
fi

CC="${CC:-/usr/bin/clang}"
CXX="${CXX:-/usr/bin/clang++}"
MACOSX_TARGET="arm64-apple-macos${MACOSX_MIN_VER%%.*}"
CFLAGS="-mmacosx-version-min=$MACOSX_MIN_VER -target $MACOSX_TARGET"
CXXFLAGS="-mmacosx-version-min=$MACOSX_MIN_VER -target $MACOSX_TARGET"
LDFLAGS="-mmacosx-version-min=$MACOSX_MIN_VER -target $MACOSX_TARGET"

export CC CXX CFLAGS CXXFLAGS LDFLAGS

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

trap '[ -n "${BUILD_DIR:-}" ] && [ -d "$BUILD_DIR" ] && rm -rf "$BUILD_DIR"' EXIT INT SIGTERM

METAL_KERNELS_DIR="$BUILD_DIR/_deps/mlx-build/mlx/backend/metal/kernels"
METAL_LIB="$METAL_KERNELS_DIR/mlx.metallib"

cmake -S "$MLX_C_DIR" -B "$BUILD_DIR" \
  -DMLX_METAL_PATH="$METAL_KERNELS_DIR" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="$CFLAGS" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS"

cmake --build "$BUILD_DIR"

AR_FILES_ALL="$(find "$BUILD_DIR" -type f -name '*.a' | sort || true)"
AR_COUNT="$(printf "%s\n" "$AR_FILES_ALL" | sed '/^$/d' | wc -l | tr -d '[:space:]')"
if [ "$AR_COUNT" -lt "$EXPECTED_AR_COUNT" ]; then echo "expected at least $EXPECTED_AR_COUNT static libs, got $AR_COUNT" >&2; exit 1; fi

rm -rf "$BIN_DIR"
mkdir -p "$BIN_DIR"
echo "BIN_DIR=$BIN_DIR"

cp "$METAL_LIB" "$BIN_DIR/"
printf "%s\n" "$AR_FILES_ALL" | head -n "$EXPECTED_AR_COUNT" | while IFS= read -r f; do
  [ -n "$f" ] && cp "$f" "$BIN_DIR/"
done

BIN_COUNT="$(find "$BIN_DIR" -type f | wc -l | tr -d '[:space:]')"
if [ "$BIN_COUNT" -ne $((EXPECTED_AR_COUNT + 1)) ]; then echo "unexpected files count in mlx-bin: $BIN_COUNT" >&2; exit 1; fi

printf '%s\n' "$CUR_VER" > "$VERSION_FILE"
