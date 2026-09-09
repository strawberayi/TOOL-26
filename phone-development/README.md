# IDENTI-SKIN — Phone Development

This directory contains the mobile application source, build pipeline, and compiled Android Package (APK) for **IDENTI-SKIN** (PUP CCIS Thesis 2026).

---

## Quick Start — Pre-built APK

The compiled debug APK is located at:
```
phone-development/output/identi-skin-debug.apk
```
* **Package Name**: `ph.edu.pup.ccis.identiskin`
* **Target SDK**: Android 34 / 35 (Android 14 / 15)
* **Minimum SDK**: Android 22 (Android 5.1 Lollipop+)
* **Architecture**: Universal (arm64-v8a, armeabi-v7a, x86, x86_64)

### Installing onto a Connected Device or Emulator via ADB
```bash
/home/eney/Android/Sdk/platform-tools/adb install -r output/identi-skin-debug.apk
```
Or simply transfer `identi-skin-debug.apk` to your Android device and tap to install.

---

## One-Click Rebuild

Whenever you modify any frontend HTML/CSS/JS in `../frontend`, rebuild the APK with:
```bash
cd phone-development
./build-apk.sh
```

This script:
1. Syncs web assets from `../frontend` into the native Android assets directory.
2. Compiles the Java/Kotlin and Android resources using Gradle.
3. Outputs the fresh APK into `phone-development/output/identi-skin-debug.apk`.

---

## Opening in Android Studio

You can open the native Android project in Android Studio:
1. Launch Android Studio:
   ```bash
   /home/eney/android-studio/bin/studio.sh
   ```
2. Choose **Open** and select:
   `/home/eney/Documents/TOOL2026/TOOL-26/phone-development/android`
3. Run or debug directly on connected devices or virtual emulators.

---

## Directory Structure

```
phone-development/
├── output/
│   └── identi-skin-debug.apk    # Ready-to-install Android APK
├── build-apk.sh                 # One-click rebuild script
├── android/                     # Full native Android Studio / Gradle project
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── AndroidManifest.xml   # App permissions (Camera, Storage, Internet)
│   │   │   ├── assets/public/        # Packaged web frontend assets
│   │   │   └── java/                 # Native MainActivity (Capacitor WebView bridge)
│   │   └── build.gradle
│   ├── build.gradle
│   └── gradle.properties
├── capacitor.config.json        # Capacitor config linking to ../frontend
├── package.json                 # Mobile dependencies (@capacitor/core, @capacitor/android)
├── scripts/                     # Mobile-specific UI and layout update utilities
└── README.md                    # This documentation
```
