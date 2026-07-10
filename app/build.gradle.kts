plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

val releaseStorePath = providers.gradleProperty("LIGHTASR_KEYSTORE_FILE").orNull
    ?: System.getenv("LIGHTASR_KEYSTORE_FILE")
val releaseStorePassword = providers.gradleProperty("LIGHTASR_KEYSTORE_PASSWORD").orNull
    ?: System.getenv("LIGHTASR_KEYSTORE_PASSWORD")
val releaseKeyAlias = providers.gradleProperty("LIGHTASR_KEY_ALIAS").orNull
    ?: System.getenv("LIGHTASR_KEY_ALIAS")
val releaseKeyPassword = providers.gradleProperty("LIGHTASR_KEY_PASSWORD").orNull
    ?: System.getenv("LIGHTASR_KEY_PASSWORD")
val releaseSigningConfigured = listOf(
    releaseStorePath,
    releaseStorePassword,
    releaseKeyAlias,
    releaseKeyPassword,
).all { !it.isNullOrBlank() }

android {
    namespace = "com.threemountain.lightasr"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.threemountain.lightasr"
        minSdk = 23
        targetSdk = 34
        versionCode = 2
        versionName = "0.2.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    signingConfigs {
        if (releaseSigningConfigured) {
            create("release") {
                storeFile = file(releaseStorePath!!)
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".vadtest"
        }
        release {
            isDebuggable = false
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(project(":sherpa_onnx"))
}
