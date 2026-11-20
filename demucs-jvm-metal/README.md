# demucs-jvm-metal

Optional Apple GPU backend for [demucs-jvm](../demucs-jvm). Put it on the classpath next to `demucs-jvm`; on a Mac
with Metal it takes over matrix multiplies (MPS), attention (MPS matrix multiply + softmax) and the elementwise
GELU/GLU/gating kernels (Metal shaders compiled at startup). Elsewhere it reports itself unavailable and demucs-jvm
stays on its pure-Java path.

Pure Java (FFM + Objective-C runtime), no native code to build. Requires JDK 25+ and
`--enable-native-access=ALL-UNNAMED` to avoid the JDK's warning.
