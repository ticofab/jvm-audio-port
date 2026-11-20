package io.github.ticofab.demucs;

import java.util.function.DoubleConsumer;

/** Separates one stem out of a whole stereo song; what {@link SeparationProcessor} runs. */
@FunctionalInterface
public interface StemSeparator {

  /**
   * {@code mix} is {@code [2][n]} at 44.1 kHz; returns the stem as {@code [2][n]}. {@code progress} receives the
   * completed fraction, 0 to 1.
   */
  float[][] separate(float[][] mix, DoubleConsumer progress);
}
