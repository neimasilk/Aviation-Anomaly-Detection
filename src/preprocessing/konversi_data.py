import os
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import noisereduce as nr

# ============ KONFIGURASI ============
RAW_AUDIO_PATH = "data/data_voice/audio3.wav"
PROCESSED_TEXT_DIR = "processed_data/"
PROCESSED_FEATURES_DIR = "processed_data/"
CHUNK_LENGTH_MS = 30000  # 30 detik per chunk
SAMPLE_RATE = 16000
LANGUAGE = "en"  # "id" untuk bahasa Indonesia

# Pastikan direktori ada
os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
os.makedirs(PROCESSED_FEATURES_DIR, exist_ok=True)

# ============ FUNGSI PEMERIKSAAN ============
def validate_audio(audio):
    if audio is None:
        raise ValueError("Gagal membaca file audio. Pastikan formatnya WAV dan path benar.")
    if len(audio) == 0:
        raise ValueError("File audio kosong atau durasi 0 detik.")

# ============ FUNGSI PREPROCESSING ============
def preprocess_audio(y, sr):
    """Lakukan noise reduction dan normalisasi"""
    # Noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    # Normalisasi volume
    y_normalized = librosa.util.normalize(y_denoised)
    
    # Filter frekuensi suara manusia (300-3400 Hz)
    y_filtered = librosa.effects.preemphasis(y_normalized)
    
    return y_filtered

# ============ PEMROSESAN UTAMA ============
def main():
    # Load model Whisper
    model = whisper.load_model("base")
    
    # Load audio dengan error handling
    try:
        audio = AudioSegment.from_wav(RAW_AUDIO_PATH)
        validate_audio(audio)
    except Exception as e:
        raise RuntimeError(f"Error loading audio: {e}")
    
    # Split audio dengan validasi
    chunks = make_chunks(audio, CHUNK_LENGTH_MS)
    if not chunks:
        raise ValueError("Tidak ada chunk yang dihasilkan. Cek durasi audio.")
        
    # Proses tiap chunk
    for i, chunk in enumerate(chunks):
        # Skip chunk kosong
        if chunk is None or len(chunk) == 0:
            print(f"Chunk {i} kosong, dilewati")
            continue
        
        # Ekspor chunk ke file sementara
        chunk_path = f"data/raw/temp_chunk_{i}.wav"
        try:
            chunk.export(chunk_path, format="wav")
        except Exception as e:
            print(f"Gagal menyimpan chunk {i}: {e}")
            continue
        
        # Load audio dengan librosa
        try:
            y, sr = librosa.load(chunk_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Gagal membaca chunk {i} dengan librosa: {e}")
            os.remove(chunk_path)
            continue
        
        # Preprocessing
        try:
            y_processed = preprocess_audio(y, sr)
        except Exception as e:
            print(f"Error preprocessing chunk {i}: {e}")
            y_processed = y  # Gunakan audio asli jika gagal
        
        # Transkripsi dengan Whisper
        try:
            result = model.transcribe(
                chunk_path,
                language=LANGUAGE,
                verbose=False
            )
            text = result["text"].strip()
            
            # Simpan teks
            with open(f"{PROCESSED_TEXT_DIR}/transcript_{i}.txt", "w") as f:
                f.write(text)
                
        except Exception as e:
            print(f"Error transkripsi chunk {i}: {e}")
            text = "[ERROR_TRANSCRIPTION]"
        
        # Ekstraksi fitur
        try:
            # Ekstrak MFCC
            mfcc = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=13)
            
            # Ekstrak Spectrogram
            spectrogram = np.abs(librosa.stft(y_processed))
            
            # Simpan fitur
            np.save(f"{PROCESSED_FEATURES_DIR}/mfcc_{i}.npy", mfcc)
            np.save(f"{PROCESSED_FEATURES_DIR}/spectrogram_{i}.npy", spectrogram)
            
        except Exception as e:
            print(f"Error ekstraksi fitur chunk {i}: {e}")
        
        # Hapus file chunk sementara
        try:
            os.remove(chunk_path)
        except:
            pass

if __name__ == "__main__":
    main()