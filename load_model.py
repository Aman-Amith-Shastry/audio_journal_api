import requests
import io
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment

def load_audio_waveform_from_url(url, target_sr=16000):
    # Download the file in memory
    print(f"Url: ", url)
    audio_bytes = requests.get(url).content
    # Decode using soundfile (uses libsndfile + ffmpeg for m4a)
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
    wav_in_memory = io.BytesIO()
    audio_segment.export(wav_in_memory, format="wav")
    wav_in_memory.seek(0) # Reset stream position
    data, samplerate = sf.read(wav_in_memory)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Ensure float32
    data = data.astype(np.float32)

    # Resample ONLY if needed, using fast mode
    if samplerate != target_sr:
        data = librosa.resample(
            data,
            orig_sr=samplerate,
            target_sr=target_sr,
            res_type="kaiser_fast"
        )
    return data