package io.github.ticofab.demucs;

import java.util.Arrays;

/** A dense, row-major fp32 array with a shape. */
public record Tensor(float[] data, int[] shape) {

  public int dim(int i) {
    return shape[i < 0 ? shape.length + i : i];
  }

  @Override
  public String toString() {
    return "Tensor" + Arrays.toString(shape);
  }
}
