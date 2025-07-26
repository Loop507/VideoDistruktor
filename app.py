import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
import random
import cv2
import subprocess
import shutil
from scipy.io import wavfile
from scipy import signal
import librosa
import soundfile as sf

# Configurazione della pagina
st.set_page_config(page_title="VideoDistruktor by loop507", layout="centered")

# Modifica del titolo con caratteri più piccoli per "by loop507"
st.markdown("<h1>🎬🔥 VideoDistruktor <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica un video e genera versioni glitchate: VHS, Distruttivo, Noise, Combinato, Broken TV o Random! **Ora con audio glitch!**")

# Controlla se ffmpeg è disponibile
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# File uploader per video
uploaded_file = st.file_uploader("📁 Carica un video", type=["mp4", "avi", "mov", "mkv"])

def frame_to_pil(frame):
    """Converte frame OpenCV (BGR) in PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_frame(pil_img):
    """Converte PIL Image (RGB) in frame OpenCV (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Funzioni degli effetti audio ---
def extract_audio(video_path):
    """Estrae l'audio dal video usando ffmpeg"""
    audio_path = tempfile.mktemp(suffix='.wav')
    try:
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path, '-y']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.warning("⚠️ Impossibile estrarre l'audio. Il video potrebbe non avere traccia audio.")
            return None
        return audio_path
    except Exception as e:
        st.warning(f"⚠️ Errore nell'estrazione audio: {e}")
        return None

def glitch_audio_vhs(audio, sr, intensity=1.0, wow_flutter=1.0, tape_hiss=1.0):
    """Effetto audio VHS con wow&flutter e tape hiss"""
    try:
        audio_out = audio.copy()
        
        # Wow & Flutter (modulazione di pitch)
        if wow_flutter > 0:
            flutter_freq = 0.5 + (2.0 * wow_flutter)  # Frequenza modulazione
            flutter_depth = 0.02 * wow_flutter  # Profondità modulazione
            time_vec = np.arange(len(audio_out)) / sr
            modulation = np.sin(2 * np.pi * flutter_freq * time_vec) * flutter_depth
            
            # Applica modulazione (simulazione di pitch shift)
            for i in range(len(audio_out)):
                if i > 0:
                    shift_samples = int(modulation[i] * sr * 0.001)  # Micro-shift
                    if abs(shift_samples) < len(audio_out) - i:
                        audio_out[i] = audio_out[i + shift_samples] if shift_samples < 0 else audio_out[max(0, i - shift_samples)]
        
        # Tape Hiss (rumore ad alta frequenza)
        if tape_hiss > 0:
            hiss_intensity = 0.005 * tape_hiss
            hiss = np.random.normal(0, hiss_intensity, len(audio_out))
            # Filtro passa-alto per simulare il hiss delle cassette
            b, a = signal.butter(4, 2000, 'highpass', fs=sr)
            hiss_filtered = signal.filtfilt(b, a, hiss)
            audio_out += hiss_filtered
        
        # Compressione e saturazione tipica del VHS
        if intensity > 0:
            compression_ratio = 1.0 + (2.0 * intensity)
            audio_out = np.tanh(audio_out * compression_ratio) / compression_ratio
            
        return np.clip(audio_out, -1.0, 1.0)
    except Exception as e:
        st.warning(f"Errore effetto VHS audio: {e}")
        return audio

def glitch_audio_destructive(audio, sr, chaos_level=1.0, skip_prob=1.0, reverse_prob=1.0):
    """Effetto audio distruttivo con skip, reverse e chaos"""
    try:
        audio_out = audio.copy()
        chunk_size = int(sr * 0.05)  # Chunk da 50ms
        
        i = 0
        while i < len(audio_out):
            chunk_end = min(i + chunk_size, len(audio_out))
            current_chunk = audio_out[i:chunk_end]
            
            # Skip casuale (simula salti del nastro)
            if random.random() < (0.05 * skip_prob):
                # Sostituisce il chunk con silenzio o ripete il precedente
                if random.random() < 0.5 and i > chunk_size:
                    audio_out[i:chunk_end] = audio_out[i-chunk_size:i-chunk_size+(chunk_end-i)]
                else:
                    audio_out[i:chunk_end] = 0
            
            # Reverse casuale
            elif random.random() < (0.03 * reverse_prob):
                audio_out[i:chunk_end] = current_chunk[::-1]
            
            # Chaos (distorsione estrema)
            elif random.random() < (0.02 * chaos_level):
                chaos_factor = 1.0 + (random.uniform(0, 5) * chaos_level)
                audio_out[i:chunk_end] = np.clip(current_chunk * chaos_factor, -1.0, 1.0)
                audio_out[i:chunk_end] = np.tanh(audio_out[i:chunk_end])  # Saturazione
            
            i += chunk_size
        
        return audio_out
    except Exception as e:
        st.warning(f"Errore effetto distruttivo audio: {e}")
        return audio

def glitch_audio_noise(audio, sr, noise_intensity=1.0, digital_artifacts=1.0, bit_crush=1.0):
    """Effetto audio noise con artefatti digitali e bit crushing"""
    try:
        audio_out = audio.copy()
        
        # Noise classico
        if noise_intensity > 0:
            noise_level = 0.01 * noise_intensity
            noise = np.random.normal(0, noise_level, len(audio_out))
            audio_out += noise
        
        # Artefatti digitali (dropouts)
        if digital_artifacts > 0:
            dropout_prob = 0.001 * digital_artifacts
            for i in range(len(audio_out)):
                if random.random() < dropout_prob:
                    # Dropout di durata variabile
                    dropout_length = random.randint(1, int(sr * 0.01))  # Fino a 10ms
                    end_idx = min(i + dropout_length, len(audio_out))
                    audio_out[i:end_idx] = 0
        
        # Bit Crushing (riduzione risoluzione)
        if bit_crush > 0:
            # Riduce i bit di risoluzione
            bits = max(1, int(16 - (12 * bit_crush)))  # Da 16 bit a 4 bit
            scale = 2**(bits-1)
            audio_out = np.round(audio_out * scale) / scale
        
        return np.clip(audio_out, -1.0, 1.0)
    except Exception as e:
        st.warning(f"Errore effetto noise audio: {e}")
        return audio

def glitch_audio_broken_tv(audio, sr, static_intensity=1.0, channel_separation=1.0, frequency_drift=1.0):
    """Effetto audio broken TV con static, separazione canali e drift"""
    try:
        audio_out = audio.copy()
        
        # Static (rumore bianco intermittente)
        if static_intensity > 0:
            static_prob = 0.02 * static_intensity
            static_level = 0.1 * static_intensity
            
            for i in range(0, len(audio_out), int(sr * 0.1)):  # Ogni 100ms
                if random.random() < static_prob:
                    static_length = random.randint(int(sr * 0.01), int(sr * 0.1))  # 10-100ms
                    end_idx = min(i + static_length, len(audio_out))
                    static_noise = np.random.uniform(-static_level, static_level, end_idx - i)
                    
                    if len(audio_out.shape) > 1:  # Stereo
                        for ch in range(audio_out.shape[1]):
                            audio_out[i:end_idx, ch] = static_noise
                    else:  # Mono
                        audio_out[i:end_idx] = static_noise
        
        # Separazione canali (simula problemi di connessione)
        if channel_separation > 0 and len(audio_out.shape) > 1:
            separation_prob = 0.01 * channel_separation
            
            for i in range(0, len(audio_out), int(sr * 0.2)):
                if random.random() < separation_prob:
                    # Uno dei canali va in mute per un periodo
                    mute_length = random.randint(int(sr * 0.05), int(sr * 0.3))
                    end_idx = min(i + mute_length, len(audio_out))
                    channel_to_mute = random.randint(0, 1)
                    audio_out[i:end_idx, channel_to_mute] = 0
        
        # Frequency Drift (simula deriva della frequenza di campionamento)
        if frequency_drift > 0:
            drift_amount = 0.02 * frequency_drift  # Fino al 2%
            time_vec = np.arange(len(audio_out)) / sr
            drift = np.sin(2 * np.pi * 0.1 * time_vec) * drift_amount  # Drift lento
            
            # Applica il drift (approssimazione)
            for i in range(1, len(audio_out)):
                drift_samples = int(drift[i] * sr * 0.001)
                if abs(drift_samples) < len(audio_out) - i:
                    if len(audio_out.shape) > 1:  # Stereo
                        for ch in range(audio_out.shape[1]):
                            if drift_samples > 0:
                                audio_out[i, ch] = audio_out[max(0, i - drift_samples), ch]
                            elif drift_samples < 0:
                                audio_out[i, ch] = audio_out[min(len(audio_out)-1, i - drift_samples), ch]
                    else:  # Mono
                        if drift_samples > 0:
                            audio_out[i] = audio_out[max(0, i - drift_samples)]
                        elif drift_samples < 0:
                            audio_out[i] = audio_out[min(len(audio_out)-1, i - drift_samples)]
        
        return np.clip(audio_out, -1.0, 1.0)
    except Exception as e:
        st.warning(f"Errore effetto broken TV audio: {e}")
        return audio

def process_audio_glitch(audio_path, effect_type, params):
    """Processa l'audio con l'effetto scelto"""
    try:
        # Carica l'audio
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # Assicurati che sia nel formato corretto
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)
        elif len(audio.shape) == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        
        processed_audio = audio.copy()
        
        # Applica l'effetto appropriato
        if effect_type == 'vhs':
            intensity, wow_flutter, tape_hiss = params
            if len(audio.shape) > 1:
                for ch in range(audio.shape[1]):
                    processed_audio[:, ch] = glitch_audio_vhs(audio[:, ch], sr, intensity, wow_flutter, tape_hiss)
            else:
                processed_audio = glitch_audio_vhs(audio.flatten(), sr, intensity, wow_flutter, tape_hiss)
                
        elif effect_type == 'distruttivo':
            chaos_level, skip_prob, reverse_prob = params
            if len(audio.shape) > 1:
                for ch in range(audio.shape[1]):
                    processed_audio[:, ch] = glitch_audio_destructive(audio[:, ch], sr, chaos_level, skip_prob, reverse_prob)
            else:
                processed_audio = glitch_audio_destructive(audio.flatten(), sr, chaos_level, skip_prob, reverse_prob)
                
        elif effect_type == 'noise':
            noise_intensity, digital_artifacts, bit_crush = params
            if len(audio.shape) > 1:
                for ch in range(audio.shape[1]):
                    processed_audio[:, ch] = glitch_audio_noise(audio[:, ch], sr, noise_intensity, digital_artifacts, bit_crush)
            else:
                processed_audio = glitch_audio_noise(audio.flatten(), sr, noise_intensity, digital_artifacts, bit_crush)
                
        elif effect_type == 'broken_tv':
            static_intensity, channel_separation, frequency_drift = params
            processed_audio = glitch_audio_broken_tv(audio, sr, static_intensity, channel_separation, frequency_drift)
            
        elif effect_type == 'combined':
            # Applica effetti combinati
            current_audio = audio.copy()
            
            if params.get("apply_vhs") and len(audio.shape) > 1:
                for ch in range(current_audio.shape[1]):
                    current_audio[:, ch] = glitch_audio_vhs(
                        current_audio[:, ch], sr,
                        params.get("vhs_intensity", 1.0),
                        params.get("vhs_wow_flutter", 1.0),
                        params.get("vhs_tape_hiss", 1.0)
                    )
            elif params.get("apply_vhs"):
                current_audio = glitch_audio_vhs(
                    current_audio.flatten(), sr,
                    params.get("vhs_intensity", 1.0),
                    params.get("vhs_wow_flutter", 1.0),
                    params.get("vhs_tape_hiss", 1.0)
                )
                
            if params.get("apply_distruttivo") and len(current_audio.shape) > 1:
                for ch in range(current_audio.shape[1]):
                    current_audio[:, ch] = glitch_audio_destructive(
                        current_audio[:, ch], sr,
                        params.get("dest_chaos_level", 1.0),
                        params.get("dest_skip_prob", 1.0),
                        params.get("dest_reverse_prob", 1.0)
                    )
            elif params.get("apply_distruttivo"):
                current_audio = glitch_audio_destructive(
                    current_audio.flatten(), sr,
                    params.get("dest_chaos_level", 1.0),
                    params.get("dest_skip_prob", 1.0),
                    params.get("dest_reverse_prob", 1.0)
                )
                
            if params.get("apply_noise") and len(current_audio.shape) > 1:
                for ch in range(current_audio.shape[1]):
                    current_audio[:, ch] = glitch_audio_noise(
                        current_audio[:, ch], sr,
                        params.get("noise_intensity", 1.0),
                        params.get("noise_digital_artifacts", 1.0),
                        params.get("noise_bit_crush", 1.0)
                    )
            elif params.get("apply_noise"):
                current_audio = glitch_audio_noise(
                    current_audio.flatten(), sr,
                    params.get("noise_intensity", 1.0),
                    params.get("noise_digital_artifacts", 1.0),
                    params.get("noise_bit_crush", 1.0)
                )
                
            if params.get("apply_broken_tv"):
                current_audio = glitch_audio_broken_tv(
                    current_audio, sr,
                    params.get("tv_static_intensity", 1.0),
                    params.get("tv_channel_separation", 1.0),
                    params.get("tv_frequency_drift", 1.0)
                )
                
            processed_audio = current_audio
            
        elif effect_type == 'random':
            # Effetto casuale
            random_level = params[0] if params else 1.0
            effects = [
                ('vhs', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level)),
                ('distruttivo', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level)),
                ('noise', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level)),
                ('broken_tv', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level))
            ]
            
            chosen_effect, chosen_params = random.choice(effects)
            return process_audio_glitch(audio_path, chosen_effect, chosen_params)
        
        # Salva l'audio processato
        output_audio_path = tempfile.mktemp(suffix='.wav')
        sf.write(output_audio_path, processed_audio, sr)
        
        return output_audio_path
        
    except Exception as e:
        st.warning(f"Errore nel processing audio: {e}")
        return audio_path  # Ritorna l'audio originale in caso di errore

def combine_video_audio(video_path, audio_path, output_path):
    """Combina video e audio usando ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest',
            output_path, '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Errore nella combinazione video/audio: {result.stderr}")
            return False
        return True
    except Exception as e:
        st.error(f"Errore nella combinazione video/audio: {e}")
        return False

# --- Funzioni degli effetti video (mantenute dal codice originale) ---
def glitch_vhs_frame(frame, intensity=1.0, scanline_freq=1.0, color_shift=1.0):
    try:
        arr = frame.copy()
        h, w, _ = arr.shape
        base_shift = int(5 + (15 * intensity))
        for y in range(0, h, int(max(1, 2 * (2 - scanline_freq)))):
            shift = int(base_shift * np.sin(y * 0.1 * scanline_freq))
            if shift != 0:
                arr[y:y+1, :, :] = np.roll(arr[y:y+1, :, :], shift, axis=1)
            if random.random() < (0.05 + 0.05 * intensity):
                noise_amount = int(5 + (10 * intensity))
                arr[y:y+1, :, :] = np.clip(arr[y:y+1, :, :] + np.random.randint(-noise_amount, noise_amount, (1, w, 3)), 0, 255)
        b, g, r = cv2.split(arr)
        shift_val = int(5 * color_shift)
        r = np.roll(r, random.randint(-shift_val, shift_val), axis=1)
        b = np.roll(b, random.randint(-shift_val, shift_val), axis=1)
        return cv2.merge([b, g, r])
    except Exception as e:
        return frame

def glitch_distruttivo_frame(frame, block_size=1.0, num_blocks=1.0, displacement=1.0):
    try:
        arr = frame.copy()
        h, w, _ = arr.shape
        if w < 20 or h < 20:
            return frame
        max_total_blocks = min(10, w * h // 5000)
        total_blocks = int(max(1, max_total_blocks * num_blocks))
        for i in range(total_blocks):
            max_w_block = int(min(w // 8, 20 + 20 * block_size))
            max_h_block = int(min(h // 8, 20 + 20 * block_size))
            w_block = random.randint(min(3, max_w_block), max_w_block)
            h_block = random.randint(min(3, max_h_block), max_h_block)

            x = random.randint(0, max(0, w - w_block -1))
            y = random.randint(0, max(0, h - h_block -1))

            max_disp = int(min(w//10, h//10, 10 + 10 * displacement))
            dx = random.randint(-max_disp, max_disp)
            dy = random.randint(-max_disp, max_disp)

            x_new = np.clip(x + dx, 0, w - w_block)
            y_new = np.clip(y + dy, 0, h - h_block)

            if h_block > 0 and w_block > 0 and y + h_block <= h and x + w_block <= w:
                block = arr[y:y+h_block, x:x+w_block].copy()
                if y_new + h_block <= h and x_new + w_block <= w:
                    arr[y_new:y_new+h_block, x_new:x_new+w_block] = block
        return arr
    except Exception as e:
        return frame

def glitch_noise_frame(frame, noise_intensity=1.0, coverage=1.0, chaos=1.0):
    try:
        arr = frame.copy().astype(np.int16)
        h, w, _ = arr.shape
        base_intensity = int(10 + (40 * noise_intensity))

        if random.random() < coverage:
            if chaos < 0.4:
                num_bands = int(2 + (5 * coverage))
                for _ in range(num_bands):
                    start_y = random.randint(0, h-1)
                    band_height = int(1 + (10 * noise_intensity))
                    end_y = min(start_y + band_height, h)
                    band_noise = np.random.randint(-base_intensity, base_intensity, (end_y - start_y, w, 3))
                    arr[start_y:end_y] += band_noise
            elif chaos < 0.8:
                num_pixels = int(w * h * 0.005 * coverage)
                for _ in range(num_pixels):
                    x = random.randint(0, w-1)
                    y = random.randint(0, h-1)
                    pixel_noise = np.random.randint(-base_intensity, base_intensity, 3)
                    arr[y, x] += pixel_noise
            else:
                general_intensity = int(base_intensity * 0.5)
                if random.random() < 0.2:
                    noise_block_h = int(h * (0.1 + 0.2 * coverage))
                    if noise_block_h > 0:
                        start_y = random.randint(0, h - noise_block_h)
                        arr[start_y:start_y+noise_block_h] += np.random.randint(-general_intensity, general_intensity, (noise_block_h, w, 3))

        if chaos > 0.5:
            channel = random.randint(0, 2)
            multiplier = random.uniform(0.8, 1.2)
            arr[:,:,channel] = np.clip(arr[:,:,channel] * multiplier, 0, 255)

        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception as e:
        return frame

def glitch_broken_tv_frame(frame, shift_intensity=1.0, line_height=1.0, flicker_prob=1.0):
    try:
        arr = frame.copy()
        h, w, _ = arr.shape

        min_line_h = max(1, int(1 + (10 * (1 - line_height))))
        max_line_h = max(1, int(20 * line_height))

        y = 0
        while y < h:
            current_line_h = random.randint(min_line_h, max_line_h)

            if random.random() < shift_intensity:
                max_shift = int(10 + (25 * shift_intensity))
                shift_amount = random.randint(-max_shift, max_shift)

                end_y = min(y + current_line_h, h)

                if shift_amount != 0 and (end_y - y) > 0:
                    arr[y:end_y, :, :] = np.roll(arr[y:end_y, :, :], shift_amount, axis=1)

            if random.random() < (0.05 + 0.15 * flicker_prob):
                noise_amount = int(5 + (20 * flicker_prob))

                noise_start_y = random.randint(y, min(y + current_line_h - 1, h - 1))
                noise_end_y = min(noise_start_y + int(current_line_h * random.uniform(0.2, 0.8)), h)

                if noise_end_y > noise_start_y:
                    arr[noise_start_y:noise_end_y, :, :] = np.clip(
                        arr[noise_start_y:noise_end_y, :, :] + np.random.randint(-noise_amount, noise_amount, (noise_end_y - noise_start_y, w, 3)),
                        0, 255
                    )

            y += current_line_h

        return arr
    except Exception as e:
        return frame

def process_video(video_path, effect_type, params, max_frames=None, include_audio=True):
    """Processa il video con l'effetto scelto, includendo l'audio glitch se richiesto."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il video. Potrebbe essere danneggiato o non supportato.")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is None or max_frames == 0:
        actual_total_frames = total_frames
    else:
        actual_total_frames = min(total_frames, max_frames)

    # Video temporaneo senza audio
    temp_video_path = tempfile.mktemp(suffix='_no_audio.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error("❌ Impossibile inizializzare VideoWriter. Controlla i codec o i permessi di scrittura.")
            cap.release()
            return None

        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Processa i frame video
        while cap.isOpened() and frame_count < actual_total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = frame
            try:
                if effect_type == 'vhs':
                    processed_frame = glitch_vhs_frame(frame, *params)
                elif effect_type == 'distruttivo':
                    processed_frame = glitch_distruttivo_frame(frame, *params)
                elif effect_type == 'noise':
                    processed_frame = glitch_noise_frame(frame, *params)
                elif effect_type == 'broken_tv':
                    processed_frame = glitch_broken_tv_frame(frame, *params)
                elif effect_type == 'combined':
                    # Applica effetti combinati sui frame
                    current_frame = frame
                    if params.get("apply_vhs"):
                        current_frame = glitch_vhs_frame(
                            current_frame,
                            params.get("vhs_intensity", 1.0),
                            params.get("vhs_scanline_freq", 1.0),
                            params.get("vhs_color_shift", 1.0)
                        )
                    if params.get("apply_distruttivo"):
                        current_frame = glitch_distruttivo_frame(
                            current_frame,
                            params.get("dest_block_size", 1.0),
                            params.get("dest_num_blocks", 1.0),
                            params.get("dest_displacement", 1.0)
                        )
                    if params.get("apply_noise"):
                        current_frame = glitch_noise_frame(
                            current_frame,
                            params.get("noise_intensity", 1.0),
                            params.get("noise_coverage", 1.0),
                            params.get("noise_chaos", 1.0)
                        )
                    if params.get("apply_broken_tv"):
                        current_frame = glitch_broken_tv_frame(
                            current_frame,
                            params.get("tv_shift_intensity", 1.0),
                            params.get("tv_line_height", 1.0),
                            params.get("tv_flicker_prob", 1.0)
                        )
                    processed_frame = current_frame
                elif effect_type == 'random':
                    # Applica un effetto casuale
                    random_level = params[0] if params else 1.0
                    effects = [
                        ('vhs', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level)),
                        ('distruttivo', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level)),
                        ('noise', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level)),
                        ('broken_tv', (random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level, random.uniform(0.5, 1.5) * random_level))
                    ]
                    chosen_effect, chosen_params = random.choice(effects)
                    if chosen_effect == 'vhs':
                        processed_frame = glitch_vhs_frame(frame, *chosen_params)
                    elif chosen_effect == 'distruttivo':
                        processed_frame = glitch_distruttivo_frame(frame, *chosen_params)
                    elif chosen_effect == 'noise':
                        processed_frame = glitch_noise_frame(frame, *chosen_params)
                    elif chosen_effect == 'broken_tv':
                        processed_frame = glitch_broken_tv_frame(frame, *chosen_params)

            except Exception as e:
                st.warning(f"⚠️ Errore nel processing del frame {frame_count}: {e}")
                processed_frame = frame

            out.write(processed_frame)
            frame_count += 1

            # Aggiorna progress bar
            progress = frame_count / actual_total_frames
            progress_bar.progress(progress)
            status_text.text(f"🎬 Processando frame {frame_count}/{actual_total_frames} ({progress*100:.1f}%)")

        cap.release()
        out.release()

        # Se include_audio è True, processa anche l'audio
        final_output_path = tempfile.mktemp(suffix='.mp4')
        
        if include_audio and check_ffmpeg():
            status_text.text("🎵 Processando audio...")
            
            # Estrai l'audio dal video originale
            audio_path = extract_audio(video_path)
            
            if audio_path:
                # Processa l'audio con l'effetto corrispondente
                audio_params = []
                audio_effect_type = effect_type
                
                if effect_type == 'vhs':
                    audio_params = params  # intensity, scanline_freq -> wow_flutter, color_shift -> tape_hiss
                elif effect_type == 'distruttivo':
                    audio_params = params  # block_size -> chaos_level, num_blocks -> skip_prob, displacement -> reverse_prob
                elif effect_type == 'noise':
                    audio_params = params  # noise_intensity, coverage -> digital_artifacts, chaos -> bit_crush
                elif effect_type == 'broken_tv':
                    audio_params = params  # shift_intensity -> static_intensity, line_height -> channel_separation, flicker_prob -> frequency_drift
                elif effect_type == 'combined':
                    audio_params = params  # Passa tutti i parametri
                elif effect_type == 'random':
                    audio_params = params  # Passa il livello random
                
                processed_audio_path = process_audio_glitch(audio_path, audio_effect_type, audio_params)
                
                # Combina video processato con audio processato
                if combine_video_audio(temp_video_path, processed_audio_path, final_output_path):
                    status_text.text("✅ Video e audio processati con successo!")
                    
                    # Pulizia file temporanei
                    try:
                        os.unlink(temp_video_path)
                        os.unlink(audio_path)
                        os.unlink(processed_audio_path)
                    except:
                        pass
                    
                    return final_output_path
                else:
                    st.warning("⚠️ Errore nella combinazione audio/video. Ritorno solo il video.")
                    return temp_video_path
            else:
                st.info("ℹ️ Nessun audio trovato nel video o errore nell'estrazione. Processando solo il video.")
                return temp_video_path
        else:
            # Se non include_audio o ffmpeg non disponibile, ritorna solo il video
            if not check_ffmpeg():
                st.warning("⚠️ FFmpeg non disponibile. Audio glitch disabilitato.")
            return temp_video_path

    except Exception as e:
        st.error(f"❌ Errore durante il processing del video: {e}")
        cap.release()
        if 'out' in locals():
            out.release()
        return None

# Interfaccia Streamlit principale
if uploaded_file is not None:
    # Controlla ffmpeg per l'audio
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        st.warning("⚠️ FFmpeg non è disponibile. Gli effetti audio saranno disabilitati. Solo gli effetti video funzioneranno.")
    
    # Opzioni audio
    include_audio = st.checkbox("🎵 Includi effetti audio glitch", value=ffmpeg_available, disabled=not ffmpeg_available)
    if not ffmpeg_available and include_audio:
        st.info("ℹ️ Per abilitare gli effetti audio, installa FFmpeg sul sistema.")

    # Salva il file caricato
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Selettore dell'effetto
    effect_type = st.selectbox(
        "🎭 Scegli l'effetto glitch:",
        ["vhs", "distruttivo", "noise", "combined", "broken_tv", "random"],
        format_func=lambda x: {
            "vhs": "📼 VHS Glitch",
            "distruttivo": "💥 Distruttivo",
            "noise": "📺 Noise",
            "combined": "🌟 Combinato",
            "broken_tv": "📻 Broken TV",
            "random": "🎲 Random"
        }[x]
    )

    # Parametri specifici per ogni effetto
    params = {}
    
    if effect_type == 'vhs':
        st.subheader("📼 Parametri VHS")
        col1, col2, col3 = st.columns(3)
        with col1:
            vhs_intensity = st.slider("Intensità generale", 0.1, 3.0, 1.0, 0.1)
        with col2:
            scanline_freq = st.slider("Frequenza scanline", 0.1, 3.0, 1.0, 0.1)
        with col3:
            color_shift = st.slider("Color shift", 0.1, 3.0, 1.0, 0.1)
        
        params = (vhs_intensity, scanline_freq, color_shift)
        
        if include_audio:
            st.subheader("🎵 Parametri Audio VHS")
            col1, col2, col3 = st.columns(3)
            with col1:
                wow_flutter = st.slider("Wow & Flutter", 0.1, 3.0, 1.0, 0.1)
            with col2:
                tape_hiss = st.slider("Tape Hiss", 0.1, 3.0, 1.0, 0.1)

    elif effect_type == 'distruttivo':
        st.subheader("💥 Parametri Distruttivo")
        col1, col2, col3 = st.columns(3)
        with col1:
            block_size = st.slider("Dimensione blocchi", 0.1, 3.0, 1.0, 0.1)
        with col2:
            num_blocks = st.slider("Numero blocchi", 0.1, 3.0, 1.0, 0.1)
        with col3:
            displacement = st.slider("Spostamento", 0.1, 3.0, 1.0, 0.1)
        
        params = (block_size, num_blocks, displacement)
        
        if include_audio:
            st.subheader("🎵 Parametri Audio Distruttivo")
            col1, col2, col3 = st.columns(3)
            with col1:
                chaos_level = st.slider("Livello chaos", 0.1, 3.0, 1.0, 0.1)
            with col2:
                skip_prob = st.slider("Probabilità skip", 0.1, 3.0, 1.0, 0.1)
            with col3:
                reverse_prob = st.slider("Probabilità reverse", 0.1, 3.0, 1.0, 0.1)

    elif effect_type == 'noise':
        st.subheader("📺 Parametri Noise")
        col1, col2, col3 = st.columns(3)
        with col1:
            noise_intensity = st.slider("Intensità noise", 0.1, 3.0, 1.0, 0.1)
        with col2:
            coverage = st.slider("Copertura", 0.1, 3.0, 1.0, 0.1)
        with col3:
            chaos = st.slider("Chaos", 0.1, 3.0, 1.0, 0.1)
        
        params = (noise_intensity, coverage, chaos)
        
        if include_audio:
            st.subheader("🎵 Parametri Audio Noise")
            col1, col2, col3 = st.columns(3)
            with col1:
                digital_artifacts = st.slider("Artefatti digitali", 0.1, 3.0, 1.0, 0.1)
            with col2:
                bit_crush = st.slider("Bit Crushing", 0.1, 3.0, 1.0, 0.1)

    elif effect_type == 'broken_tv':
        st.subheader("📻 Parametri Broken TV")
        col1, col2, col3 = st.columns(3)
        with col1:
            shift_intensity = st.slider("Intensità shift", 0.1, 3.0, 1.0, 0.1)
        with col2:
            line_height = st.slider("Altezza linee", 0.1, 3.0, 1.0, 0.1)
        with col3:
            flicker_prob = st.slider("Probabilità flicker", 0.1, 3.0, 1.0, 0.1)
        
        params = (shift_intensity, line_height, flicker_prob)
        
        if include_audio:
            st.subheader("🎵 Parametri Audio Broken TV")
            col1, col2, col3 = st.columns(3)
            with col1:
                static_intensity = st.slider("Intensità static", 0.1, 3.0, 1.0, 0.1)
            with col2:
                channel_separation = st.slider("Separazione canali", 0.1, 3.0, 1.0, 0.1)
            with col3:
                frequency_drift = st.slider("Drift frequenza", 0.1, 3.0, 1.0, 0.1)

    elif effect_type == 'combined':
        st.subheader("🌟 Parametri Combinato")
        
        # Selettore degli effetti da combinare
        st.write("Seleziona gli effetti da combinare:")
        apply_vhs = st.checkbox("📼 VHS", value=True)
        apply_distruttivo = st.checkbox("💥 Distruttivo", value=True)
        apply_noise = st.checkbox("📺 Noise", value=True)
        apply_broken_tv = st.checkbox("📻 Broken TV", value=True)
        
        params = {
            "apply_vhs": apply_vhs,
            "apply_distruttivo": apply_distruttivo,
            "apply_noise": apply_noise,
            "apply_broken_tv": apply_broken_tv
        }
        
        # Parametri specifici per ogni effetto attivo
        if apply_vhs:
            st.write("**Parametri VHS:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                params["vhs_intensity"] = st.slider("VHS Intensità", 0.1, 3.0, 1.0, 0.1, key="vhs_int")
            with col2:
                params["vhs_scanline_freq"] = st.slider("VHS Scanline", 0.1, 3.0, 1.0, 0.1, key="vhs_scan")
            with col3:
                params["vhs_color_shift"] = st.slider("VHS Color", 0.1, 3.0, 1.0, 0.1, key="vhs_color")
            
            if include_audio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["vhs_wow_flutter"] = st.slider("VHS Wow&Flutter", 0.1, 3.0, 1.0, 0.1, key="vhs_wow")
                with col2:
                    params["vhs_tape_hiss"] = st.slider("VHS Tape Hiss", 0.1, 3.0, 1.0, 0.1, key="vhs_hiss")
        
        if apply_distruttivo:
            st.write("**Parametri Distruttivo:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                params["dest_block_size"] = st.slider("Dest Block Size", 0.1, 3.0, 1.0, 0.1, key="dest_block")
            with col2:
                params["dest_num_blocks"] = st.slider("Dest Num Blocks", 0.1, 3.0, 1.0, 0.1, key="dest_num")
            with col3:
                params["dest_displacement"] = st.slider("Dest Displacement", 0.1, 3.0, 1.0, 0.1, key="dest_disp")
            
            if include_audio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["dest_chaos_level"] = st.slider("Dest Chaos", 0.1, 3.0, 1.0, 0.1, key="dest_chaos")
                with col2:
                    params["dest_skip_prob"] = st.slider("Dest Skip", 0.1, 3.0, 1.0, 0.1, key="dest_skip")
                with col3:
                    params["dest_reverse_prob"] = st.slider("Dest Reverse", 0.1, 3.0, 1.0, 0.1, key="dest_rev")
        
        if apply_noise:
            st.write("**Parametri Noise:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                params["noise_intensity"] = st.slider("Noise Intensity", 0.1, 3.0, 1.0, 0.1, key="noise_int")
            with col2:
                params["noise_coverage"] = st.slider("Noise Coverage", 0.1, 3.0, 1.0, 0.1, key="noise_cov")
            with col3:
                params["noise_chaos"] = st.slider("Noise Chaos", 0.1, 3.0, 1.0, 0.1, key="noise_chaos")
            
            if include_audio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["noise_digital_artifacts"] = st.slider("Noise Artifacts", 0.1, 3.0, 1.0, 0.1, key="noise_art")
                with col2:
                    params["noise_bit_crush"] = st.slider("Noise Bit Crush", 0.1, 3.0, 1.0, 0.1, key="noise_bit")
        
        if apply_broken_tv:
            st.write("**Parametri Broken TV:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                params["tv_shift_intensity"] = st.slider("TV Shift", 0.1, 3.0, 1.0, 0.1, key="tv_shift")
            with col2:
                params["tv_line_height"] = st.slider("TV Line Height", 0.1, 3.0, 1.0, 0.1, key="tv_line")
            with col3:
                params["tv_flicker_prob"] = st.slider("TV Flicker", 0.1, 3.0, 1.0, 0.1, key="tv_flick")
            
            if include_audio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    params["tv_static_intensity"] = st.slider("TV Static", 0.1, 3.0, 1.0, 0.1, key="tv_static")
                with col2:
                    params["tv_channel_separation"] = st.slider("TV Channel Sep", 0.1, 3.0, 1.0, 0.1, key="tv_chan")
                with col3:
                    params["tv_frequency_drift"] = st.slider("TV Freq Drift", 0.1, 3.0, 1.0, 0.1, key="tv_drift")

    elif effect_type == 'random':
        st.subheader("🎲 Parametri Random")
        random_level = st.slider("Livello di casualità", 0.1, 3.0, 1.0, 0.1)
        params = (random_level,)

    # Limita frame per video lunghi
    max_frames = st.number_input("🎬 Limite frame (0 = nessun limite)", min_value=0, max_value=10000, value=0)
    
    # Bottone per processare
    if st.button("🚀 Processa Video"):
        if not any([effect_type != 'combined' or any(params.values()) if isinstance(params, dict) else True]):
            st.warning("⚠️ Seleziona almeno un effetto per la modalità combinata!")
        else:
            with st.spinner("🔥 Processando il video..."):
                result_path = process_video(video_path, effect_type, params, max_frames, include_audio)
                
                if result_path:
                    # Mostra il video risultante
                    st.success("✅ Video processato con successo!")
                    
                    with open(result_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        
                        # Download button
                        effect_name = {
                            "vhs": "VHS",
                            "distruttivo": "Distruttivo", 
                            "noise": "Noise",
                            "combined": "Combinato",
                            "broken_tv": "BrokenTV",
                            "random": "Random"
                        }[effect_type]
                        
                        filename = f"glitched_{effect_name}_{uploaded_file.name}"
                        st.download_button(
                            label="📥 Scarica video glitchato",
                            data=video_bytes,
                            file_name=filename,
                            mime="video/mp4"
                        )
                    
                    # Pulizia
                    try:
                        os.unlink(result_path)
                    except:
                        pass
                else:
                    st.error("❌ Errore durante il processing del video.")
    
    # Pulizia file temporaneo
    try:
        os.unlink(video_path)
    except:
        pass

else:
    st.info("👆 Carica un video per iniziare!")

# Footer
st.markdown("---")
st.markdown("🎬 **VideoDistruktor by loop507** - Trasforma i tuoi video in opere d'arte glitched!")
if not check_ffmpeg():
    st.markdown("⚠️ *Per abilitare gli effetti audio, installa FFmpeg sul sistema*")
