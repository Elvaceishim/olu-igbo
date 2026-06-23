"""
Numerical parity harness for the on-device mel spectrogram.

Ground truth is the same transformers WhisperFeatureExtractor the model was
trained with (and that legacy/mel_server.py formerly served off-device). We check:
  1. the CURRENT Kotlin algorithm (computeLogMel) ported to numpy  -> shows the drift
  2. a FIXED algorithm (power spectrum + center/reflect pad + Slaney mel) -> should match
  3. a from-scratch Slaney filterbank vs the extractor's own mel_filters
If 2 and 3 match to tight tolerance, the Kotlin port is a straight transcription.
"""

import numpy as np
from transformers import WhisperProcessor

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
N_SAMPLES = 480000  # 30s
N_FRAMES = 3000

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
fe = proc.feature_extractor
MEL_GT = np.asarray(fe.mel_filters)  # (n_freqs, n_mels) = (201, 80)


def ground_truth(audio):
    return fe(audio, sampling_rate=SR, return_tensors="np").input_features[0]  # (80, 3000)


def hann_periodic(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


# ---- 1. current Kotlin algorithm, mirrored ----------------------------------
def build_tri_filterbank():
    n_freqs = N_FFT // 2 + 1
    hz2mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mmin, mmax = hz2mel(0.0), hz2mel(SR / 2.0)
    pts = np.array([mel2hz(mmin + i * (mmax - mmin) / (N_MELS + 1)) for i in range(N_MELS + 2)])
    bins = np.array([i * SR / N_FFT for i in range(n_freqs)])
    filt = np.zeros((N_MELS, n_freqs))
    for m in range(N_MELS):
        lo, ctr, hi = pts[m], pts[m + 1], pts[m + 2]
        for f in range(n_freqs):
            fr = bins[f]
            if fr < lo or fr > hi:
                filt[m, f] = 0.0
            elif fr <= ctr:
                filt[m, f] = (fr - lo) / (ctr - lo)
            else:
                filt[m, f] = (hi - fr) / (hi - ctr)
    return filt


def kotlin_current(audio):
    padded = np.zeros(N_SAMPLES)
    n = min(len(audio), N_SAMPLES)
    padded[:n] = audio[:n]
    window = hann_periodic(N_FFT)
    melf = build_tri_filterbank()
    num_frames = (N_SAMPLES - N_FFT) // HOP + 1  # 2998, no centering
    mel = np.zeros((N_MELS, N_FRAMES))
    maxv = -np.inf
    for frame in range(num_frames):
        s = frame * HOP
        spec = np.fft.rfft(padded[s:s + N_FFT] * window)
        mag = np.sqrt(spec.real ** 2 + spec.imag ** 2)  # bug: magnitude, not power
        if frame < N_FRAMES:
            for m in range(N_MELS):
                lv = np.log10(max((melf[m] * mag).sum(), 1e-10))
                mel[m, frame] = lv
                maxv = max(maxv, lv)
    mel = np.maximum(mel, maxv - 8.0)
    return (mel + 4.0) / 4.0


# ---- 2. fixed algorithm -----------------------------------------------------
def mel_fixed(audio, mel_filters):
    x = np.zeros(N_SAMPLES)
    n = min(len(audio), N_SAMPLES)
    x[:n] = audio[:n]
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")          # center=True
    window = hann_periodic(N_FFT)
    nf = 1 + (len(xp) - N_FFT) // HOP                    # 3001
    frames = np.stack([xp[i * HOP:i * HOP + N_FFT] * window for i in range(nf)])
    spec = np.fft.rfft(frames, axis=1)                  # (3001, 201)
    power = (spec.real ** 2 + spec.imag ** 2)[:-1]       # power, drop last -> (3000, 201)
    mel_spec = np.maximum(power @ mel_filters, 1e-10)    # (3000, 80)
    log_spec = np.log10(mel_spec)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).T                   # (80, 3000)


# ---- 3. from-scratch Slaney filterbank --------------------------------------
def mel_filterbank_slaney():
    n_freqs = N_FFT // 2 + 1
    fftfreqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    f_sp = 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz2mel(f):
        f = np.asarray(f, float)
        m = f / f_sp
        hi = f >= min_log_hz
        m = np.where(hi, min_log_mel + np.log(np.where(hi, f, min_log_hz) / min_log_hz) / logstep, m)
        return m

    def mel2hz(m):
        m = np.asarray(m, float)
        f = f_sp * m
        hi = m >= min_log_mel
        f = np.where(hi, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)
        return f

    mel_pts = np.linspace(hz2mel(0.0), hz2mel(SR / 2.0), N_MELS + 2)
    freq_pts = mel2hz(mel_pts)
    fdiff = np.diff(freq_pts)
    ramps = np.subtract.outer(freq_pts, fftfreqs)
    w = np.zeros((N_MELS, n_freqs))
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (freq_pts[2:N_MELS + 2] - freq_pts[:N_MELS])
    w *= enorm[:, None]
    return w.T  # (n_freqs, n_mels) to match fe.mel_filters


def report(name, a, b):
    d = np.abs(a - b)
    print(f"  {name:28s} max={d.max():.3e}  mean={d.mean():.3e}")


rng = np.random.default_rng(0)
signals = {
    "sine440": np.sin(2 * np.pi * 440 * np.arange(SR * 4) / SR),
    "chirp": np.sin(2 * np.pi * (200 + 600 * np.arange(SR * 4) / (SR * 4)) * np.arange(SR * 4) / SR),
    "noise": rng.standard_normal(SR * 4) * 0.1,
}

print("=== filterbank: from-scratch Slaney vs extractor.mel_filters ===")
report("slaney filterbank", mel_filterbank_slaney(), MEL_GT)

print("\n=== mel parity vs transformers ground truth ===")
for name, sig in signals.items():
    sig = sig.astype(np.float32)
    gt = ground_truth(sig)
    print(f"[{name}]")
    report("current kotlin port", kotlin_current(sig), gt)
    report("fixed (extractor filters)", mel_fixed(sig, MEL_GT), gt)
    report("fixed (own slaney filters)", mel_fixed(sig, mel_filterbank_slaney()), gt)
