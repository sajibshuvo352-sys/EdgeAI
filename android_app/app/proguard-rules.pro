# Proguard rules for Edge AI Benchmark
-keep class org.tensorflow.lite.** { *; }
-keepattributes *Annotation*
-dontwarn org.tensorflow.lite.**
