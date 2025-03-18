import os
import whisper
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf



# 2. Konversi Audio ke Teks menggunakan Whisper
def audio_to_text(audio_path, model_name='base'):
    # Load model
    model = whisper.load_model(model_name)
    
    # Transcribe audio
    result = model.transcribe(audio_path)
    
    # Save text result
    text_output_path = 'processed_data/transcript.txt'
    with open(text_output_path, 'w') as f:
        f.write(result['text'])
    
    print(f"Transkrip berhasil disimpan di {text_output_path}")
    return result['text']

# 3. Analisis Fitur Audio dengan Librosa
def extract_audio_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)  # Pertahankan sampling rate asli
    
    # Ekstraksi MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Ekstraksi Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Simpan fitur
    feature_output_path = 'data/processed/audio_features.npy'
    np.save(feature_output_path, {
        'mfcc': mfcc,
        'spectrogram': D,
        'sample_rate': sr
    })
    
    # Visualisasi Spectrogram (opsional)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig('data/processed/spectrogram.png')
    plt.close()
    
    print(f"Fitur audio berhasil disimpan di {feature_output_path}")
    return mfcc, D

# 4. Main Pipeline
def main():
    # Path ke file audio asli
    audio_file = 'data/data_voice/audio3.wav'
    
    # Step 1: Konversi Audio ke Teks
    print("Memulai proses konversi speech-to-text...")
    transcript = audio_to_text(audio_file)
    
    # Step 2: Ekstraksi Fitur Audio
    print("\nMemulai ekstraksi fitur audio...")
    mfcc, spectrogram = extract_audio_features(audio_file)
    
    print("\nProses preprocessing selesai!")
