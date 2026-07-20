"""Strictly causal audio features for end-of-turn detection.

The public interface in this module deliberately accepts ``pause_start`` but
never ``pause_end``.  ``extract_causal_features`` first crops the waveform to
audio strictly preceding that timestamp, then computes every representation
from the crop.  This is the inference-time causality contract for the project.

The features are organised around four complementary signals:

* energy cessation and its recent trajectory;
* speaker-relative pitch and voicing behaviour;
* final active-speech timing; and
* lightweight spectral dynamics.

They are summarised over several trailing windows rather than treating a
single instantaneous frame as an end-of-turn decision.  The module contains no
labels, pause durations, file-duration features, or future-audio processing.
"""
from __future__ import annotations

import wave

import numpy as np


FRAME_MS = 25
HOP_MS = 10
PITCH_FRAME_MS = 40
CONTEXT_S = 1.5
MULTISCALE_WINDOWS_S = (0.10, 0.25, 0.50, 1.00, 1.50)
SPECTRAL_WINDOWS_S = (0.50, 1.50)
EPS = 1e-8


def load_wav(path):
    """Load a PCM WAV as mono float32 audio without third-party I/O packages.

    The assignment data are 16 kHz PCM WAV files.  Supporting the common PCM
    sample widths here keeps the feature path within the official dependency
    rules and avoids changing behaviour based on any future part of a turn.
    """
    with wave.open(path, "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError(f"unsupported compressed WAV: {path}")
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        triples = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            triples[:, 0].astype(np.int32)
            | (triples[:, 1].astype(np.int32) << 8)
            | (triples[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        x = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width {sample_width}: {path}")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def speech_before(x, sr, pause_start, window_s=CONTEXT_S):
    """Return at most ``window_s`` seconds ending strictly at ``pause_start``."""
    end = min(len(x), max(0, int(np.floor(float(pause_start) * sr))))
    start = max(0, end - int(round(window_s * sr)))
    return np.asarray(x[start:end], dtype=np.float32)


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Create non-padded frames whose samples all lie in the supplied prefix."""
    frame_length = int(round(sr * frame_ms / 1000))
    hop_length = int(round(sr * hop_ms / 1000))
    if len(x) < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)
    count = 1 + (len(x) - frame_length) // hop_length
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None]
    return np.asarray(x, dtype=np.float32)[indices]


def frame_energy_db(x, sr):
    """Short-time RMS energy per frame in dB."""
    fr = frames(x, sr)
    if len(fr) == 0:
        return np.empty(0, dtype=np.float32)
    rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Estimate one frame's F0 with autocorrelation, or return 0 if unvoiced."""
    frame = np.asarray(frame, dtype=np.float32) - np.mean(frame)
    if len(frame) == 0 or np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    low_lag = int(sr / fmax)
    high_lag = min(int(sr / fmin), len(ac) - 1)
    if high_lag <= low_lag:
        return 0.0
    lag = low_lag + int(np.argmax(ac[low_lag:high_lag]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=PITCH_FRAME_MS, hop_ms=HOP_MS):
    """Per-frame F0 in Hz, with zero marking unvoiced or silent frames."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(frame, sr) for frame in fr], dtype=np.float32)


def frame_spectral_features(x, sr):
    """Compute causal per-frame spectral shape and change features.

    The output arrays use the same 25 ms / 10 ms frame geometry as energy.
    Spectral flux compares each frame with the immediately preceding frame,
    which is also entirely in the observed audio prefix.
    """
    fr = frames(x, sr)
    if len(fr) == 0:
        empty = np.empty(0, dtype=np.float32)
        return {"centroid": empty, "flatness": empty, "flux": empty, "zcr": empty}

    window = np.hanning(fr.shape[1]).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(fr * window[None, :], axis=1)) + EPS
    power = magnitude**2
    frequencies = np.fft.rfftfreq(fr.shape[1], d=1.0 / sr).astype(np.float32)
    centroid = (power @ frequencies) / (power.sum(axis=1) + EPS)
    flatness = np.exp(np.mean(np.log(power), axis=1)) / (np.mean(power, axis=1) + EPS)

    normalized = magnitude / (magnitude.sum(axis=1, keepdims=True) + EPS)
    flux = np.zeros(len(fr), dtype=np.float32)
    if len(fr) > 1:
        flux[1:] = np.sqrt(np.mean(np.diff(normalized, axis=0) ** 2, axis=1))

    zcr = np.mean(fr[:, 1:] * fr[:, :-1] < 0.0, axis=1)
    return {
        "centroid": centroid.astype(np.float32),
        "flatness": flatness.astype(np.float32),
        "flux": flux.astype(np.float32),
        "zcr": zcr.astype(np.float32),
    }


def _tail(values: np.ndarray, window_s: float, hop_s: float) -> np.ndarray:
    """Return a trailing time window from a frame sequence."""
    if len(values) == 0:
        return values
    count = max(1, int(np.ceil(window_s / hop_s)))
    return values[-count:]


def _linear_slope(values: np.ndarray, hop_s: float) -> float:
    """Least-squares slope per second, returning zero for unavailable input."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    time = np.arange(len(values), dtype=np.float64) * hop_s
    centered_time = time - time.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= EPS:
        return 0.0
    return float(np.dot(centered_time, values - values.mean()) / denominator)


def _masked_slope(values: np.ndarray, valid: np.ndarray, hop_s: float) -> float:
    """Slope for valid frames only, preserving their original timing."""
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return 0.0
    time = indices.astype(np.float64) * hop_s
    observed = np.asarray(values, dtype=np.float64)[indices]
    centered_time = time - time.mean()
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= EPS:
        return 0.0
    return float(np.dot(centered_time, observed - observed.mean()) / denominator)


def _last_true_run(mask: np.ndarray):
    """Return (last-run length, frames since its end) for a boolean sequence."""
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return 0, len(mask)
    end = int(indices[-1])
    start = end
    while start > 0 and mask[start - 1]:
        start -= 1
    return end - start + 1, len(mask) - 1 - end


def _robust_relative(value: float, reference: np.ndarray):
    """Return value relative to a causal median/IQR baseline and its validity."""
    reference = np.asarray(reference, dtype=np.float64)
    if len(reference) < 5:
        return 0.0, 0.0
    q25, median, q75 = np.percentile(reference, [25, 50, 75])
    scale = max(float(q75 - q25), 1e-3)
    return float(np.clip((value - median) / scale, -20.0, 20.0)), 1.0


def _window_label(window_s: float) -> str:
    return f"{int(round(window_s * 1000)):04d}ms"


def feature_names() -> list[str]:
    """Return stable names for ``extract_causal_features`` columns."""
    names = ["context_s", "context_is_short"]
    for window_s in MULTISCALE_WINDOWS_S:
        label = _window_label(window_s)
        names.extend(
            [
                f"energy_{label}_mean_db",
                f"energy_{label}_std_db",
                f"energy_{label}_slope_db_s",
                f"energy_{label}_end_minus_start_db",
            ]
        )
    names.extend(
        [
            "energy_final_relative_to_context",
            "energy_final_minus_previous_250ms_db",
            "energy_active_fraction",
            "energy_last_active_run_s",
            "energy_gap_after_last_active_s",
        ]
    )
    for window_s in MULTISCALE_WINDOWS_S:
        label = _window_label(window_s)
        names.extend(
            [
                f"voicing_{label}_fraction",
                f"f0_{label}_slope_semitones_s",
                f"f0_{label}_range_semitones",
            ]
        )
    names.extend(
        [
            "f0_final_relative_to_context",
            "f0_final_minus_previous_250ms_semitones",
            "f0_final_std_semitones",
            "f0_reference_available",
            "f0_last_voiced_run_s",
            "f0_gap_after_last_voiced_s",
        ]
    )
    for window_s in SPECTRAL_WINDOWS_S:
        label = _window_label(window_s)
        names.extend(
            [
                f"spectral_centroid_{label}_slope_hz_s",
                f"spectral_flatness_{label}_mean",
                f"spectral_flux_{label}_mean",
                f"zcr_{label}_mean",
            ]
        )
    return names


def extract_causal_features(x, sr, pause_start, context_s=CONTEXT_S):
    """Extract a 56-dimensional terminality representation for one pause.

    Only ``x[0 : pause_start]`` is consulted.  The returned representation uses
    zero-valued fallbacks plus explicit availability features when an early
    pause lacks enough past speech for a reliable estimate.
    """
    segment = speech_before(x, sr, pause_start, window_s=context_s)
    hop_s = HOP_MS / 1000.0
    values: list[float] = []

    def add(value: float) -> None:
        values.append(float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)))

    context_duration = len(segment) / float(sr) if sr else 0.0
    add(context_duration)
    add(float(context_duration < context_s - hop_s))

    energy = frame_energy_db(segment, sr)
    for window_s in MULTISCALE_WINDOWS_S:
        tail = _tail(energy, window_s, hop_s)
        if len(tail) == 0:
            add(0.0)
            add(0.0)
            add(0.0)
            add(0.0)
        else:
            add(float(np.mean(tail)))
            add(float(np.std(tail)))
            add(_linear_slope(tail, hop_s))
            add(float(tail[-1] - tail[0]))

    final_energy = _tail(energy, 0.25, hop_s)
    excluded = max(1, int(np.ceil(0.25 / hop_s)))
    energy_reference = energy[:-excluded] if len(energy) > excluded + 4 else energy
    final_energy_mean = float(np.mean(final_energy)) if len(final_energy) else 0.0
    energy_relative, _ = _robust_relative(final_energy_mean, energy_reference)
    add(energy_relative)
    previous_energy = energy[-2 * excluded : -excluded] if len(energy) >= 2 * excluded else np.empty(0)
    add(final_energy_mean - float(np.mean(previous_energy)) if len(previous_energy) else 0.0)

    if len(energy):
        active_threshold = float(np.max(energy) - 30.0)
        active = energy >= active_threshold
        active_run, active_gap = _last_true_run(active)
        add(float(np.mean(active)))
        add(active_run * hop_s)
        add(active_gap * hop_s)
    else:
        add(0.0)
        add(0.0)
        add(0.0)

    f0_hz = f0_contour(segment, sr)
    voiced = f0_hz > 0.0
    f0_st = np.zeros_like(f0_hz, dtype=np.float32)
    if np.any(voiced):
        f0_st[voiced] = 12.0 * np.log2(f0_hz[voiced])

    for window_s in MULTISCALE_WINDOWS_S:
        tail = _tail(f0_st, window_s, hop_s)
        valid = _tail(voiced, window_s, hop_s)
        if len(tail) == 0:
            add(0.0)
            add(0.0)
            add(0.0)
        else:
            add(float(np.mean(valid)))
            add(_masked_slope(tail, valid, hop_s))
            add(float(np.ptp(tail[valid])) if np.any(valid) else 0.0)

    final_f0 = _tail(f0_st, 0.25, hop_s)
    final_voiced = _tail(voiced, 0.25, hop_s)
    final_f0_values = final_f0[final_voiced]
    f0_reference = f0_st[:-excluded][voiced[:-excluded]] if len(f0_st) > excluded + 4 else f0_st[voiced]
    final_f0_mean = float(np.mean(final_f0_values)) if len(final_f0_values) else 0.0
    f0_relative, f0_reference_available = _robust_relative(final_f0_mean, f0_reference)
    add(f0_relative)

    previous_f0 = f0_st[-2 * excluded : -excluded]
    previous_voiced = voiced[-2 * excluded : -excluded]
    previous_f0_values = previous_f0[previous_voiced]
    add(final_f0_mean - float(np.mean(previous_f0_values)) if len(previous_f0_values) else 0.0)
    add(float(np.std(final_f0_values)) if len(final_f0_values) else 0.0)
    add(f0_reference_available)
    voiced_run, voiced_gap = _last_true_run(voiced)
    add(voiced_run * hop_s)
    add(voiced_gap * hop_s)

    spectral = frame_spectral_features(segment, sr)
    for window_s in SPECTRAL_WINDOWS_S:
        centroid = _tail(spectral["centroid"], window_s, hop_s)
        flatness = _tail(spectral["flatness"], window_s, hop_s)
        flux = _tail(spectral["flux"], window_s, hop_s)
        zcr = _tail(spectral["zcr"], window_s, hop_s)
        add(_linear_slope(centroid, hop_s))
        add(float(np.mean(flatness)) if len(flatness) else 0.0)
        add(float(np.mean(flux)) if len(flux) else 0.0)
        add(float(np.mean(zcr)) if len(zcr) else 0.0)

    vector = np.asarray(values, dtype=np.float32)
    expected = len(feature_names())
    if len(vector) != expected:
        raise RuntimeError(f"feature count mismatch: got {len(vector)}, expected {expected}")
    return vector
