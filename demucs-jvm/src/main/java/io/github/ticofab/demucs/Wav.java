package io.github.ticofab.demucs;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/** Minimal WAV I/O: reads 16-bit, 24-bit or 32-bit float PCM, writes 32-bit float. Stereo or mono (duplicated). */
public final class Wav {

  private Wav() {}

  /** Returns {@code [2][frames]}. */
  public static float[][] read(Path file) throws IOException {
    ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
    if (b.getInt(0) != 0x46464952 || b.getInt(8) != 0x45564157) throw new IOException("not a WAV file: " + file);
    int pos = 12, format = 0, channels = 0, bits = 0, dataOff = -1, dataLen = 0;
    while (pos + 8 <= b.limit()) {
      int id = b.getInt(pos), len = b.getInt(pos + 4);
      if (id == 0x20746d66) { // "fmt "
        format = b.getShort(pos + 8) & 0xffff;
        channels = b.getShort(pos + 10);
        bits = b.getShort(pos + 22);
        if (format == 0xfffe) format = b.getShort(pos + 32) & 0xffff; // WAVE_FORMAT_EXTENSIBLE sub-format
      } else if (id == 0x61746164) { // "data"
        dataOff = pos + 8;
        dataLen = len < 0 || dataOff + len > b.limit() ? b.limit() - dataOff : len; // streamed WAVs may lie
        break;
      }
      pos += 8 + len + (len & 1);
    }
    if (dataOff < 0 || channels < 1 || channels > 2) throw new IOException("unsupported WAV layout: " + file);
    int bytes = bits / 8, frames = dataLen / (bytes * channels);
    float[][] out = new float[2][frames];
    for (int f = 0; f < frames; f++) {
      for (int c = 0; c < channels; c++) {
        int o = dataOff + (f * channels + c) * bytes;
        float v = switch (format * 100 + bits) {
          case 116 -> b.getShort(o) / 32768f;
          case 124 -> ((b.get(o) & 0xff) | (b.get(o + 1) & 0xff) << 8 | b.get(o + 2) << 16) / 8388608f;
          case 332 -> b.getFloat(o);
          default -> throw new IOException("unsupported WAV sample format " + format + "/" + bits + " bits");
        };
        out[c][f] = v;
      }
      if (channels == 1) out[1][f] = out[0][f];
    }
    return out;
  }

  /** Writes {@code [2][frames]} as 32-bit float stereo at 44.1 kHz. */
  public static void write(Path file, float[][] audio) throws IOException {
    int frames = audio[0].length, dataLen = frames * 8;
    ByteBuffer b = ByteBuffer.allocate(44 + dataLen).order(ByteOrder.LITTLE_ENDIAN);
    b.putInt(0x46464952).putInt(36 + dataLen).putInt(0x45564157);
    b.putInt(0x20746d66).putInt(16).putShort((short) 3).putShort((short) 2).putInt(HTDemucs.SAMPLE_RATE)
        .putInt(HTDemucs.SAMPLE_RATE * 8).putShort((short) 8).putShort((short) 32);
    b.putInt(0x61746164).putInt(dataLen);
    for (int f = 0; f < frames; f++) b.putFloat(audio[0][f]).putFloat(audio[1][f]);
    Files.write(file, b.array());
  }
}
