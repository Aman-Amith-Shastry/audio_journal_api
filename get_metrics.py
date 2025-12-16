import numpy as np
import librosa

def compute_speech_metrics(y, sr=16000):
    import numpy as np
    import librosa

    # ----- 1. Speech Rate (syllables/sec approximation) -----
    # Detect voiced segments
    voiced_segments = librosa.effects.split(y, top_db=30)
    total_voiced_time = sum([(end - start) / sr for start, end in voiced_segments])

    # Detect onsets (proxy for syllables)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    speech_rate = len(onsets) / total_voiced_time if total_voiced_time > 0 else 0.0

    # ----- 2. Loudness (RMS energy mean) -----
    rms = librosa.feature.rms(y=y)[0]
    loudness_mean = float(np.mean(rms))

    # ----- 3. Pitch (F0 fundamental frequency) -----
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(f0)) if len(f0) else 0.0

    # ----- 4. Speech Energy Variability -----
    energy_variability = float(np.std(rms))

    return {
        "speech_rate": float(speech_rate),
        "loudness_mean": loudness_mean,
        "pitch_mean": pitch_mean,
        "energy_variability": energy_variability
    }