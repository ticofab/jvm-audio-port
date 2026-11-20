# demucs-jvm-metal

Optional Apple GPU backend for [demucs-jvm](../demucs-jvm). Put it on the classpath next to `demucs-jvm`; on a Mac
with Metal it takes over matrix multiplies, attention, norms and the elementwise GELU/GLU/gating work, with Metal
kernels of its own compiled at startup, and implements the device API for BS-Roformer's transformer stack.
Elsewhere it reports itself unavailable and demucs-jvm stays on its pure-Java path.

It used Metal Performance Shaders at first, but MPS's matrix multiply returned NaN tiles from clean inputs depending
on how buffers sat in memory, and its softmax was slow; the kernels here check their bounds and are exact to float
rounding against PyTorch.

Pure Java (FFM + Objective-C runtime), no native code to build. Requires JDK 25+ and
`--enable-native-access=ALL-UNNAMED` to avoid the JDK's warning.
