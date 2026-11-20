package io.github.ticofab.demucs;

import java.util.Arrays;
import java.util.concurrent.Executor;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;
import java.util.function.DoubleConsumer;

/**
 * A {@link Flow.Processor} that separates one source (e.g. bass) out of a stereo stream.
 *
 * <p>Input elements are interleaved stereo frames ({@code L0, R0, L1, R1, ...}) at 44.1 kHz, in chunks of any size;
 * output elements are the separated source in the same format, in chunks of {@code outputChunkFrames}.
 *
 * <p>Demucs normalizes by statistics of the whole song, so this processor collects the full input before separating;
 * output starts once the input completes. Separation runs on {@code executor}, never on the caller's thread, and
 * output respects downstream demand.
 */
public final class SeparationProcessor implements Flow.Processor<float[], float[]> {

  private final Separator separator;
  private final int sourceIndex;
  private final int outputChunkFrames;
  private final Executor executor;
  private final DoubleConsumer progress;
  private final SubmissionPublisher<float[]> out;

  private Flow.Subscription upstream;
  private float[] buffer = new float[1 << 20];
  private int size;

  /**
   * @param source name of the source to emit, one of {@link Separator#sources()}
   * @param progress receives the separation's completed fraction, from {@code executor}'s thread
   */
  public SeparationProcessor(Separator separator, String source, int outputChunkFrames, Executor executor,
                             DoubleConsumer progress) {
    this.separator = separator;
    this.sourceIndex = separator.sources().indexOf(source);
    if (sourceIndex < 0) throw new IllegalArgumentException("unknown source " + source + "; have " + separator.sources());
    this.outputChunkFrames = outputChunkFrames;
    this.executor = executor;
    this.progress = progress;
    this.out = new SubmissionPublisher<>(executor, Flow.defaultBufferSize());
  }

  @Override
  public void subscribe(Flow.Subscriber<? super float[]> subscriber) {
    out.subscribe(subscriber);
  }

  @Override
  public void onSubscribe(Flow.Subscription subscription) {
    if (upstream != null) {
      subscription.cancel();
      return;
    }
    upstream = subscription;
    // The whole song is needed before separating, so take everything upstream offers.
    subscription.request(Long.MAX_VALUE);
  }

  @Override
  public void onNext(float[] frames) {
    if ((frames.length & 1) != 0) {
      upstream.cancel();
      out.closeExceptionally(new IllegalArgumentException("odd number of samples in a stereo chunk"));
      return;
    }
    if (size + frames.length > buffer.length) buffer = Arrays.copyOf(buffer, Math.max(buffer.length * 2, size + frames.length));
    System.arraycopy(frames, 0, buffer, size, frames.length);
    size += frames.length;
  }

  @Override
  public void onError(Throwable error) {
    out.closeExceptionally(error);
  }

  @Override
  public void onComplete() {
    float[] input = buffer;
    int samples = size;
    buffer = null;
    executor.execute(() -> {
      try {
        int n = samples / 2;
        float[][] mix = new float[2][n];
        for (int i = 0; i < n; i++) {
          mix[0][i] = input[2 * i];
          mix[1][i] = input[2 * i + 1];
        }
        float[][] source = separator.separate(mix, progress)[sourceIndex];
        for (int start = 0; start < n; start += outputChunkFrames) {
          int len = Math.min(outputChunkFrames, n - start);
          float[] chunk = new float[2 * len];
          for (int i = 0; i < len; i++) {
            chunk[2 * i] = source[0][start + i];
            chunk[2 * i + 1] = source[1][start + i];
          }
          out.submit(chunk); // blocks while the subscriber's buffer is full: backpressure
        }
        out.close();
      } catch (Throwable t) {
        out.closeExceptionally(t);
      }
    });
  }
}
