#!/usr/bin/env bash
set -e

# ==============================================================================
# IDENTI-SKIN Android APK One-Click Build Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Setting up build environment..."
export JAVA_HOME="${JAVA_HOME:-/home/eney/.jdks/jbr-21.0.11}"
export ANDROID_HOME="${ANDROID_HOME:-/home/eney/Android/Sdk}"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"

GRADLE_BIN="/home/eney/.gradle/wrapper/dists/gradle-8.14.3-all/10utluxaxniiv4wxiphsi49nj/gradle-8.14.3/bin/gradle"

echo "    JAVA_HOME:    $JAVA_HOME"
echo "    ANDROID_HOME: $ANDROID_HOME"

echo "==> Syncing web assets from frontend to Android..."
npx cap copy android

echo "==> Compiling APK with Gradle (assembleDebug)..."
cd "$SCRIPT_DIR/android"
"$GRADLE_BIN" assembleDebug

echo "==> Staging APK to output directory..."
mkdir -p "$SCRIPT_DIR/output"
cp "$SCRIPT_DIR/android/app/build/outputs/apk/debug/app-debug.apk" "$SCRIPT_DIR/output/identi-skin-debug.apk"

echo "=============================================================================="
echo " BUILD SUCCESSFUL!"
echo " APK output: $SCRIPT_DIR/output/identi-skin-debug.apk"
ls -lh "$SCRIPT_DIR/output/identi-skin-debug.apk"
echo "=============================================================================="
