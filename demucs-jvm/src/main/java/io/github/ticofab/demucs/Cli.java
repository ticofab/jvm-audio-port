package io.github.ticofab.demucs;

import java.nio.file.Path;
import java.util.Random;

/**
 * {@code java -cp demucs-jvm.jar io.github.ticofab.demucs.Cli <model.th> <in.wav> <out.wav> [source] [shifts]}:
 * separates one source of a 44.1 kHz WAV file.
 */
public final class Cli {

  private Cli() {}

  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      System.err.println("Usage: Cli <model.th> <in.wav> <out.wav> [source=bass] [shifts=1]");
      System.exit(2);
    }
    String source = args.length > 3 ? args[3] : "bass";
    int shifts = args.length > 4 ? Integer.parseInt(args[4]) : 1;
    long t0 = System.nanoTime();
    HTDemucs model = HTDemucs.load(Path.of(args[0]));
    float[][] mix = Wav.read(Path.of(args[1]));
    System.err.printf("loaded model in %.1fs; %d frames (%.1fs of audio)%n", (System.nanoTime() - t0) / 1e9,
        mix[0].length, mix[0].length / 44100.0);
    long t1 = System.nanoTime();
    Separator separator = new Separator(model, shifts, new Random());
    float[][][] out = separator.separate(mix, f -> System.err.printf("\r%3.0f%%  %.0fs", f * 100, (System.nanoTime() - t1) / 1e9));
    System.err.printf("%nseparated in %.1fs%n", (System.nanoTime() - t1) / 1e9);
    Wav.write(Path.of(args[2]), out[model.sources().indexOf(source)]);
  }
}
