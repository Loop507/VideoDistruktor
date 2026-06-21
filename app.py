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
import time

# Configurazione della pagina
st.set_page_config(page_title="VideoDistruktor by loop507", layout="centered")

# Modifica del titolo con caratteri più piccoli per "by loop507"
st.markdown("<h1>🎬🔥 VideoDistruktor <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica un video e genera versioni glitchate: VHS, Distruttivo, Noise, Combinato, Broken TV o Random! **Audio glitch su video e file audio separati (mp3/wav/aac)!**")

# File uploader per audio separato (opzionale)
uploaded_audio_file = st.file_uploader("🎵 Carica audio separato da glitchare (opzionale — mp3/wav/aac)", type=["mp3", "wav", "aac", "ogg", "flac", "m4a"])

# Controlla se ffmpeg è disponibile (cached per evitare subprocess ad ogni re-run)
@st.cache_data
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
    """Estrae l'audio dal video usando ffmpeg e lo converte in WAV 44100Hz stereo"""
    fd, audio_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
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

def convert_audio_to_wav(audio_path_in):
    """Converte qualsiasi formato audio (mp3, aac, ogg...) in WAV 44100Hz stereo tramite ffmpeg"""
    fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        cmd = ['ffmpeg', '-i', audio_path_in, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', wav_path, '-y']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.warning(f"⚠️ Conversione audio fallita: {result.stderr[-300:]}")
            return None
        return wav_path
    except Exception as e:
        st.warning(f"⚠️ Errore conversione audio: {e}")
        return None

def glitch_audio_vhs(audio, sr, intensity=1.0, wow_flutter=1.0, tape_hiss=1.0):
    """Effetto audio VHS con wow&flutter e tape hiss — vectorizzato NumPy"""
    try:
        audio_out = audio.copy().astype(np.float32)

        # Wow & Flutter — shift via interpolazione vettorizzata
        if wow_flutter > 0:
            flutter_freq  = 0.5 + 2.0 * wow_flutter
            flutter_depth = 0.008 * wow_flutter          # max ~8ms di shift
            n   = len(audio_out)
            t   = np.arange(n, dtype=np.float32) / sr
            mod = np.sin(2 * np.pi * flutter_freq * t) * flutter_depth * sr  # in sample
            idx = np.clip(np.arange(n) - mod.astype(np.int32), 0, n - 1)
            audio_out = audio_out[idx]

        # Tape Hiss — filtro passa-alto vettorizzato
        if tape_hiss > 0:
            hiss = np.random.normal(0, 0.005 * tape_hiss, len(audio_out)).astype(np.float32)
            b, a = signal.butter(4, 2000, 'highpass', fs=sr)
            audio_out += signal.filtfilt(b, a, hiss).astype(np.float32)

        # Saturazione / compressione VHS
        if intensity > 0:
            ratio = 1.0 + 2.0 * intensity
            audio_out = np.tanh(audio_out * ratio) / ratio

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
    """Effetto audio noise con artefatti digitali e bit crushing — vectorizzato NumPy"""
    try:
        audio_out = audio.copy().astype(np.float32)

        # Noise classico
        if noise_intensity > 0:
            audio_out += np.random.normal(0, 0.01 * noise_intensity, len(audio_out)).astype(np.float32)

        # Artefatti digitali (dropout) — vettorizzato con maschera booleana
        if digital_artifacts > 0:
            dropout_prob = 0.001 * digital_artifacts
            # Genera punti di inizio dropout
            starts = np.where(np.random.random(len(audio_out)) < dropout_prob)[0]
            for s in starts:
                length = random.randint(1, max(1, int(sr * 0.01)))
                audio_out[s:s + length] = 0

        # Bit Crushing vettorizzato
        if bit_crush > 0:
            bits  = max(1, int(16 - 12 * bit_crush))
            scale = float(2 ** (bits - 1))
            audio_out = np.round(audio_out * scale) / scale

        return np.clip(audio_out, -1.0, 1.0)
    except Exception as e:
        st.warning(f"Errore effetto noise audio: {e}")
        return audio

def glitch_audio_broken_tv(audio, sr, static_intensity=1.0, channel_separation=1.0, frequency_drift=1.0):
    """Effetto audio broken TV — vectorizzato NumPy"""
    try:
        audio_out = audio.copy().astype(np.float32)

        # Static intermittente
        if static_intensity > 0:
            static_prob  = 0.02 * static_intensity
            static_level = 0.1  * static_intensity
            step = max(1, int(sr * 0.1))
            for i in range(0, len(audio_out), step):
                if random.random() < static_prob:
                    length  = random.randint(int(sr * 0.01), int(sr * 0.1))
                    end_idx = min(i + length, len(audio_out))
                    noise   = np.random.uniform(-static_level, static_level, end_idx - i).astype(np.float32)
                    if audio_out.ndim > 1:
                        audio_out[i:end_idx, :] = noise[:, np.newaxis]
                    else:
                        audio_out[i:end_idx] = noise

        # Separazione canali
        if channel_separation > 0 and audio_out.ndim > 1:
            sep_prob = 0.01 * channel_separation
            step2 = max(1, int(sr * 0.2))
            for i in range(0, len(audio_out), step2):
                if random.random() < sep_prob:
                    length  = random.randint(int(sr * 0.05), int(sr * 0.3))
                    end_idx = min(i + length, len(audio_out))
                    ch      = random.randint(0, audio_out.shape[1] - 1)
                    audio_out[i:end_idx, ch] = 0

        # Frequency Drift — vettorizzato con fancy indexing
        if frequency_drift > 0:
            n           = len(audio_out)
            t           = np.arange(n, dtype=np.float32) / sr
            drift_samps = (np.sin(2 * np.pi * 0.1 * t) * 0.02 * frequency_drift * sr).astype(np.int32)
            idx         = np.clip(np.arange(n) - drift_samps, 0, n - 1)
            if audio_out.ndim > 1:
                audio_out = audio_out[idx, :]
            else:
                audio_out = audio_out[idx]

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
        fd, output_audio_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
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

# --- Funzioni degli effetti video — DISTRUTTIVI VERI ---

def glitch_pixel_sort(frame, intensity=1.0, threshold=0.5, direction=0.5):
    """Pixel sorting per luminosità su righe o colonne."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # soglia: pixel sotto threshold vengono sortati
        thr = np.clip(1.0 - threshold * 0.8, 0.05, 0.95)
        num_rows = max(1, int(h * min(intensity, 1.0)))
        row_indices = np.random.choice(h, num_rows, replace=False)
        for y in row_indices:
            mask = gray[y] < thr
            if not np.any(mask):
                continue
            # trova segmenti contigui sotto soglia
            starts = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == 1)[0]
            ends   = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == -1)[0]
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                seg = arr[y, s:e]
                lum = gray[y, s:e]
                order = np.argsort(lum)
                arr[y, s:e] = seg[order]
        # anche colonne se direction > 0.5
        if direction > 0.5:
            num_cols = max(1, int(w * min((direction - 0.5) * 2 * intensity, 1.0)))
            col_indices = np.random.choice(w, num_cols, replace=False)
            for x in col_indices:
                mask = gray[:, x] < thr
                if not np.any(mask):
                    continue
                starts = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == 1)[0]
                ends   = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == -1)[0]
                for s, e in zip(starts, ends):
                    if e - s < 2:
                        continue
                    seg = arr[s:e, x]
                    lum = gray[s:e, x]
                    order = np.argsort(lum)
                    arr[s:e, x] = seg[order]
        return arr
    except Exception:
        return frame

def glitch_channel_shift(frame, intensity=1.0, spread=1.0, mode=0.5):
    """Separazione canali RGB con offset e bleeding — corruzioni vere."""
    try:
        arr = frame.copy().astype(np.int16)
        h, w = arr.shape[:2]
        max_shift = int(w * 0.05 * intensity * spread)
        if max_shift < 1:
            return frame
        b, g, r = cv2.split(arr.astype(np.uint8))
        # shift orizzontale asimmetrico per canale
        r = np.roll(r,  random.randint(-max_shift, max_shift), axis=1)
        b = np.roll(b, -random.randint(1, max(1, max_shift)), axis=1)
        if mode > 0.5:
            # shift verticale sul canale G
            g = np.roll(g, random.randint(-max_shift // 2, max_shift // 2), axis=0)
        # bleeding: mischia valori tra canali
        bleed = np.clip(intensity * 0.15, 0, 0.4)
        r_out = np.clip(r.astype(np.float32) * (1 - bleed) + g.astype(np.float32) * bleed, 0, 255).astype(np.uint8)
        b_out = np.clip(b.astype(np.float32) * (1 - bleed) + r.astype(np.float32) * bleed, 0, 255).astype(np.uint8)
        return cv2.merge([b_out, g, r_out])
    except Exception:
        return frame

def glitch_datamosh(frame, prev_frame, intensity=1.0, block_size=1.0, chaos=1.0):
    """Datamosh reale: duplica blocchi P-frame dal frame precedente con motion displacement."""
    try:
        if prev_frame is None:
            return frame
        arr = frame.copy()
        h, w = arr.shape[:2]
        bsize = max(8, int(16 * block_size))
        # numero di blocchi proporzionale all'intensità
        n_blocks = int((h // bsize) * (w // bsize) * np.clip(intensity * 0.3, 0.05, 0.95))
        for _ in range(n_blocks):
            # sorgente dal frame precedente
            sx = random.randint(0, w - bsize)
            sy = random.randint(0, h - bsize)
            # destinazione con displacement
            max_disp = int(w * 0.1 * chaos)
            dx = random.randint(-max_disp, max_disp)
            dy = random.randint(-max_disp // 2, max_disp // 2)
            tx = np.clip(sx + dx, 0, w - bsize)
            ty = np.clip(sy + dy, 0, h - bsize)
            # copia blocco da prev_frame a posizione shiftata in frame corrente
            arr[ty:ty+bsize, tx:tx+bsize] = prev_frame[sy:sy+bsize, sx:sx+bsize]
        return arr
    except Exception:
        return frame

def glitch_byte_corrupt(frame, intensity=1.0, chunk_size=1.0, randomize=0.5):
    """Corruzione dati reale: manomette blocchi di byte grezzi come se fossero dati compressi."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        # converti in JPEG e corrompi i byte
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(20, int(85 - 50 * intensity))]
        ret, buf = cv2.imencode('.jpg', arr, encode_param)
        if not ret:
            return frame
        data = bytearray(buf.tobytes())
        # salta header JPEG (primi 500 byte circa)
        start = min(500, len(data) // 4)
        n_corruptions = max(1, int(len(data) * 0.001 * intensity))
        csize = max(1, int(4 * chunk_size))
        for _ in range(n_corruptions):
            pos = random.randint(start, max(start, len(data) - csize - 1))
            if randomize > 0.5:
                # sostituzione con byte casuali
                for i in range(csize):
                    data[pos + i] = random.randint(0, 255)
            else:
                # duplica chunk da altra posizione
                src = random.randint(start, max(start, len(data) - csize - 1))
                data[pos:pos+csize] = data[src:src+csize]
        try:
            corrupted = np.frombuffer(bytes(data), dtype=np.uint8)
            decoded = cv2.imdecode(corrupted, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.shape == frame.shape:
                return decoded
        except Exception:
            pass
        return arr
    except Exception:
        return frame

def glitch_slice_shift(frame, intensity=1.0, num_slices=1.0, drift=0.5):
    """Slice shift: taglia il frame in strisce orizzontali e le sposta in modo asincrono."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        n = max(2, int(5 + 20 * num_slices))
        slice_h = max(1, h // n)
        max_shift = int(w * 0.15 * intensity)
        if max_shift < 1:
            return frame
        for i in range(n):
            y0 = i * slice_h
            y1 = min(y0 + slice_h, h)
            # drift: shift progressivo per strisce adiacenti
            shift = int(max_shift * np.sin(i * drift * np.pi / n + random.uniform(0, np.pi)))
            if shift != 0:
                arr[y0:y1] = np.roll(arr[y0:y1], shift, axis=1)
        return arr
    except Exception:
        return frame

def glitch_vhs_frame(frame, intensity=1.0, scanline_freq=1.0, color_shift=1.0):
    """VHS: scanline + color shift + chroma noise."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        b, g, r = cv2.split(arr)
        sv = int(3 + 12 * color_shift)
        r = np.roll(r,  random.randint(-sv, sv), axis=1)
        b = np.roll(b, -random.randint(1, max(1, sv)), axis=1)
        arr = cv2.merge([b, g, r])
        freq = max(1, int(4 / max(0.1, scanline_freq)))
        for y in range(0, h, freq):
            sv2 = int((intensity * 20) * np.sin(y * 0.15 * scanline_freq + random.uniform(0, 0.5)))
            if sv2 != 0:
                arr[y:y+1] = np.roll(arr[y:y+1], sv2, axis=1)
        luma_noise = np.random.randint(-int(8*intensity), int(8*intensity)+1, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + luma_noise, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        return frame

def glitch_broken_tv_frame(frame, shift_intensity=1.0, line_height=1.0, flicker_prob=1.0):
    """Broken TV: slice shift + static."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        min_lh = max(1, int(2 * (1 - min(line_height, 0.99))))
        max_lh = max(2, int(25 * line_height))
        y = 0
        while y < h:
            lh = random.randint(min_lh, max_lh)
            end = min(y + lh, h)
            if random.random() < shift_intensity:
                ms = int(10 + 30 * shift_intensity)
                s = random.randint(-ms, ms)
                if s != 0:
                    arr[y:end] = np.roll(arr[y:end], s, axis=1)
            if random.random() < 0.1 * flicker_prob:
                noise = np.random.randint(-int(30*flicker_prob), int(30*flicker_prob)+1,
                                          arr[y:end].shape, dtype=np.int16)
                arr[y:end] = np.clip(arr[y:end].astype(np.int16) + noise, 0, 255).astype(np.uint8)
            y += lh
        return arr
    except Exception:
        return frame

def glitch_noise_frame(frame, noise_intensity=1.0, coverage=1.0, chaos=1.0):
    """Noise: bit crush + canale amplificato + bande."""
    try:
        arr = frame.copy().astype(np.int16)
        h, w = arr.shape[:2]
        # bit crush
        bits = max(2, int(8 - 5 * min(chaos, 1.0)))
        scale = 2 ** (8 - bits)
        arr = (arr // scale) * scale
        # bande di noise
        base = int(15 + 60 * noise_intensity)
        n_bands = int(1 + 8 * coverage)
        for _ in range(n_bands):
            y0 = random.randint(0, h-1)
            bh = int(1 + 15 * noise_intensity)
            y1 = min(y0 + bh, h)
            arr[y0:y1] += np.random.randint(-base, base+1, (y1-y0, w, 3), dtype=np.int16)
        ch = random.randint(0, 2)
        arr[:,:,ch] = np.clip(arr[:,:,ch] * random.uniform(0.7, 1.4), 0, 255)
        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_distruttivo_frame(frame, block_size=1.0, num_blocks=1.0, displacement=1.0):
    """Distruttivo: block glitch + inversion + channel swap."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        bw = max(4, int(10 + 60 * block_size))
        bh = max(4, int(6 + 40 * block_size))
        n = max(1, int(3 + 20 * num_blocks))
        for _ in range(n):
            x = random.randint(0, max(0, w - bw - 1))
            y = random.randint(0, max(0, h - bh - 1))
            md = int(max(1, w * 0.12 * displacement))
            dx = random.randint(-md, md)
            dy = random.randint(-md//2, md//2)
            nx = np.clip(x + dx, 0, w - bw)
            ny = np.clip(y + dy, 0, h - bh)
            block = arr[y:y+bh, x:x+bw].copy()
            # operazione distruttiva: inverte o swappa canali
            op = random.random()
            if op < 0.33:
                block = 255 - block
            elif op < 0.66:
                block = block[:, :, ::-1]  # swap BGR
            else:
                block = np.roll(block, random.randint(1, bw-1), axis=1)
            arr[ny:ny+bh, nx:nx+bw] = block
        return arr
    except Exception:
        return frame


def interpolate_keyframes(keyframes_df, fps, total_frames):
    """Interpola i keyframe su tutti i frame del video. Ritorna array di valori."""
    import numpy as np
    if keyframes_df is None or len(keyframes_df) == 0:
        return None
    try:
        times = keyframes_df["Secondo"].astype(float).tolist()
        values = keyframes_df["Intensita'"].astype(float).tolist()
        if len(times) < 2:
            return None
        # Converti secondi in frame index
        frame_times = [t * fps for t in times]
        all_frames = list(range(total_frames))
        interpolated = np.interp(all_frames, frame_times, values)
        return interpolated
    except Exception:
        return None

def process_video(video_path, effect_type, params, max_frames=None, include_audio=True, kf_envelope=None, audio_params_override=None, aspect_ratio="Originale"):
    """Processa il video con l'effetto scelto, includendo l'audio glitch se richiesto."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il video.")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    actual_total_frames = total_frames if (max_frames is None or max_frames == 0) else min(total_frames, max_frames)

    # Calcola dimensioni output in base all'aspect ratio
    def calc_output_size(w, h, ar):
        if ar == "16:9":
            ratio = 16 / 9
        elif ar == "9:16":
            ratio = 9 / 16
        elif ar == "1:1":
            ratio = 1.0
        else:
            return w, h
        if w / h > ratio:
            ow = int(h * ratio); ow -= ow % 2; oh = h
        else:
            oh = int(w / ratio); oh -= oh % 2; ow = w
        return ow, oh

    out_w, out_h = calc_output_size(width, height, aspect_ratio)

    fd1, temp_video_path = tempfile.mkstemp(suffix='_no_audio.mp4')
    os.close(fd1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    try:
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            st.error("❌ Impossibile inizializzare VideoWriter.")
            cap.release()
            return None

        frame_count = 0
        prev_frame = None
        progress_bar = st.progress(0)
        status_text = st.empty()

        while cap.isOpened() and frame_count < actual_total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = frame
            try:
                current_params = params
                if kf_envelope is not None and isinstance(params, tuple) and frame_count < len(kf_envelope):
                    kf_val = float(np.clip(kf_envelope[frame_count], 0.0, 3.0))
                    current_params = (kf_val,) + params[1:]

                if effect_type == 'pixel_sort':
                    processed_frame = glitch_pixel_sort(frame, *current_params)
                elif effect_type == 'channel_shift':
                    processed_frame = glitch_channel_shift(frame, *current_params)
                elif effect_type == 'datamosh':
                    processed_frame = glitch_datamosh(frame, prev_frame, *current_params)
                elif effect_type == 'byte_corrupt':
                    processed_frame = glitch_byte_corrupt(frame, *current_params)
                elif effect_type == 'slice_shift':
                    processed_frame = glitch_slice_shift(frame, *current_params)
                elif effect_type == 'vhs':
                    processed_frame = glitch_vhs_frame(frame, *current_params)
                elif effect_type == 'distruttivo':
                    processed_frame = glitch_distruttivo_frame(frame, *current_params)
                elif effect_type == 'noise':
                    processed_frame = glitch_noise_frame(frame, *current_params)
                elif effect_type == 'broken_tv':
                    processed_frame = glitch_broken_tv_frame(frame, *current_params)
                elif effect_type == 'combined':
                    current_frame = frame
                    if params.get("apply_vhs"):
                        current_frame = glitch_vhs_frame(current_frame, params.get("vhs_intensity",1.0), params.get("vhs_scanline_freq",1.0), params.get("vhs_color_shift",1.0))
                    if params.get("apply_distruttivo"):
                        current_frame = glitch_distruttivo_frame(current_frame, params.get("dest_block_size",1.0), params.get("dest_num_blocks",1.0), params.get("dest_displacement",1.0))
                    if params.get("apply_noise"):
                        current_frame = glitch_noise_frame(current_frame, params.get("noise_intensity",1.0), params.get("noise_coverage",1.0), params.get("noise_chaos",1.0))
                    if params.get("apply_broken_tv"):
                        current_frame = glitch_broken_tv_frame(current_frame, params.get("tv_shift_intensity",1.0), params.get("tv_line_height",1.0), params.get("tv_flicker_prob",1.0))
                    if params.get("apply_pixel_sort"):
                        current_frame = glitch_pixel_sort(current_frame, params.get("ps_intensity",1.0), params.get("ps_threshold",0.5), params.get("ps_direction",0.3))
                    if params.get("apply_channel_shift"):
                        current_frame = glitch_channel_shift(current_frame, params.get("cs_intensity",1.0), params.get("cs_spread",1.0), params.get("cs_mode",0.3))
                    if params.get("apply_slice_shift"):
                        current_frame = glitch_slice_shift(current_frame, params.get("ss_intensity",1.0), params.get("ss_num_slices",1.0), params.get("ss_drift",1.0))
                    processed_frame = current_frame
                elif effect_type == 'random':
                    random_level = params[0] if params else 1.0
                    all_effects = ['pixel_sort','channel_shift','datamosh','byte_corrupt','slice_shift','vhs','broken_tv','noise','distruttivo']
                    chosen = random.choice(all_effects)
                    rp = tuple(random.uniform(0.5, 1.5) * random_level for _ in range(3))
                    effect_fn = {
                        'pixel_sort': glitch_pixel_sort, 'channel_shift': glitch_channel_shift,
                        'byte_corrupt': glitch_byte_corrupt, 'slice_shift': glitch_slice_shift,
                        'vhs': glitch_vhs_frame, 'broken_tv': glitch_broken_tv_frame,
                        'noise': glitch_noise_frame, 'distruttivo': glitch_distruttivo_frame,
                    }
                    if chosen == 'datamosh':
                        processed_frame = glitch_datamosh(frame, prev_frame, *rp)
                    else:
                        processed_frame = effect_fn[chosen](frame, *rp)

            except Exception as e:
                processed_frame = frame

            prev_frame = frame.copy()

            # Crop al formato desiderato
            if aspect_ratio != "Originale":
                ph, pw = processed_frame.shape[:2]
                cx = (pw - out_w) // 2
                cy = (ph - out_h) // 2
                processed_frame = processed_frame[cy:cy+out_h, cx:cx+out_w]

            out.write(processed_frame)
            frame_count += 1
            progress = frame_count / actual_total_frames
            progress_bar.progress(progress)
            status_text.text(f"🎬 Frame {frame_count}/{actual_total_frames} ({progress*100:.1f}%)")

        cap.release()
        out.release()

        fd2, final_output_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd2)
        
        if include_audio and check_ffmpeg():
            status_text.text("🎵 Processando audio...")
            audio_path = extract_audio(video_path)
            if audio_path:
                audio_params = audio_params_override if audio_params_override is not None else params
                audio_effect_type = effect_type if effect_type not in ['pixel_sort','channel_shift','datamosh','byte_corrupt','slice_shift'] else 'noise'
                processed_audio_path = process_audio_glitch(audio_path, audio_effect_type, audio_params if isinstance(audio_params, tuple) else (1.0, 1.0, 1.0))
                if combine_video_audio(temp_video_path, processed_audio_path, final_output_path):
                    status_text.text("✅ Video e audio processati!")
                    for p in [temp_video_path, audio_path, processed_audio_path]:
                        try: os.unlink(p)
                        except: pass
                    return final_output_path
                else:
                    return temp_video_path
            else:
                return temp_video_path
        else:
            return temp_video_path

    except Exception as e:
        st.error(f"❌ Errore: {e}")
        cap.release()
        if 'out' in locals(): out.release()
        return None


def get_crop_filter(w, h, aspect_ratio):
    """Ritorna il filtro ffmpeg per centrare e croppare al rapporto desiderato."""
    if aspect_ratio == "16:9":
        target_w, target_h = 16, 9
    elif aspect_ratio == "9:16":
        target_w, target_h = 9, 16
    elif aspect_ratio == "1:1":
        target_w, target_h = 1, 1
    else:
        return None  # Originale, nessun crop

    # Calcola dimensioni crop mantenendo il massimo possibile
    ratio = target_w / target_h
    if w / h > ratio:
        # Video più largo: crop larghezza
        new_w = int(h * ratio)
        new_w -= new_w % 2
        new_h = h
    else:
        # Video più alto: crop altezza
        new_w = w
        new_h = int(w / ratio)
        new_h -= new_h % 2

    x = (w - new_w) // 2
    y = (h - new_h) // 2
    return f"crop={new_w}:{new_h}:{x}:{y}"

def recompress_h264(input_path, aspect_ratio="Originale"):
    """Ricomprime il video in H.264 con ffmpeg per ridurre le dimensioni."""
    fd, output_path = tempfile.mkstemp(suffix='_h264.mp4')
    os.close(fd)
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k',
            output_path, '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path
        else:
            return input_path  # fallback all'originale
    except Exception:
        return input_path

def get_file_size_mb(path):
    """Ritorna la dimensione del file in MB."""
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    except:
        return 0

def get_video_info(path):
    """Ritorna fps, risoluzione e frame totali del video."""
    try:
        cap = cv2.VideoCapture(path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frames / fps, 1) if fps > 0 else 0
        cap.release()
        return fps, w, h, frames, duration
    except:
        return 0, 0, 0, 0, 0

def build_report(original_name, original_size_mb, output_size_mb,
                 fps, width, height, total_frames, duration,
                 effect_type, params, include_audio, kf_df=None):
    """Genera il report testuale."""

    effect_names = {
        'vhs':        'VHS Glitch',
        'distruttivo':'Distruttivo',
        'noise':      'Noise',
        'combined':   'Combinato',
        'broken_tv':  'Broken TV',
        'random':     'Random'
    }

    hashtag_map = {
        'vhs':        '#vhsglitch #tapeglitch #analogcorruption',
        'distruttivo':'#destructive #blockglitch #datachaos',
        'noise':      '#noiseglitch #bitcrush #digitalartifacts',
        'combined':   '#combinedglitch #multifx #fullcorruption',
        'broken_tv':  '#brokentv #staticnoise #frequencydrift',
        'random':     '#randomglitch #chaosfx #unknownsignal'
    }

    # Parametri leggibili
    if effect_type == 'vhs' and isinstance(params, tuple):
        param_str = f"Intensita' {params[0]} | Scanline {params[1]} | Color Shift {params[2]}"
    elif effect_type == 'distruttivo' and isinstance(params, tuple):
        param_str = f"Block Size {params[0]} | Num Blocks {params[1]} | Displacement {params[2]}"
    elif effect_type == 'noise' and isinstance(params, tuple):
        param_str = f"Intensita' {params[0]} | Coverage {params[1]} | Chaos {params[2]}"
    elif effect_type == 'broken_tv' and isinstance(params, tuple):
        param_str = f"Shift {params[0]} | Line Height {params[1]} | Flicker {params[2]}"
    elif effect_type == 'combined' and isinstance(params, dict):
        active = [k.replace('apply_','').upper() for k,v in params.items() if k.startswith('apply_') and v]
        param_str = "Effetti attivi: " + ", ".join(active)
    elif effect_type == 'random' and isinstance(params, tuple):
        param_str = f"Livello casualita' {params[0]}"
    else:
        param_str = "—"

    effect_hashtags = hashtag_map.get(effect_type, '')

    report = f"""[STUDIO_GLITCH_VIDEO] // VOL_01 // H.264 // DATA_CORRUPTION
:: MOTORE: videodistruktor [v1.1]
:: EFFETTO: {effect_names.get(effect_type, effect_type)}
:: PROCESSO: Frame Destruction / {'Audio Corruption' if include_audio else 'Video Only'}

"Il glitch non e' accaduto. E' stato scelto."

> TECHNICAL LOG SHEET:
* File: {original_name}
* Durata: {duration} sec | Frame: {total_frames} @ {fps}fps
* Risoluzione: {width}x{height}
* Originale: {original_size_mb} MB → Output: {output_size_mb} MB
* Effetto Audio: {'ON' if include_audio else 'OFF'}
* Parametri: {param_str}

{'* Keyframe Intensita\':' + chr(10) + chr(10).join([f'  {row["Secondo"]}s -> {row["Intensita\'"]}'  for _, row in kf_df.iterrows()]) if kf_df is not None and len(kf_df) >= 2 else ''}

> Regia e Algoritmo: Loop507

#loop507 #glitchart #videodistruktor #datacorruption #experimentalvideo
{effect_hashtags} #brutalistart #framecorruption #signalcorruption"""

    return report


# Interfaccia Streamlit principale
if 'report_data' not in st.session_state: st.session_state.report_data = ""
if 'video_ready' not in st.session_state: st.session_state.video_ready = False
if 'h264_path'   not in st.session_state: st.session_state.h264_path   = ""
if 'effect_name_saved' not in st.session_state: st.session_state.effect_name_saved = ""
if 'orig_filename' not in st.session_state: st.session_state.orig_filename = ""
if 'glitched_audio_path' not in st.session_state: st.session_state.glitched_audio_path = ""

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
        ["pixel_sort", "channel_shift", "datamosh", "byte_corrupt", "slice_shift", "vhs", "broken_tv", "noise", "distruttivo", "combined", "random"],
        format_func=lambda x: {
            "pixel_sort":    "🔀 Pixel Sort",
            "channel_shift": "🌈 Channel Shift",
            "datamosh":      "💾 Datamosh",
            "byte_corrupt":  "🦠 Byte Corrupt",
            "slice_shift":   "✂️ Slice Shift",
            "vhs":           "📼 VHS",
            "broken_tv":     "📻 Broken TV",
            "noise":         "📺 Noise",
            "distruttivo":   "💥 Distruttivo",
            "combined":      "🌟 Combinato",
            "random":        "🎲 Random"
        }[x]
    )

    # Parametri specifici per ogni effetto
    params = {}
    audio_params_override = None

    if effect_type == 'pixel_sort':
        st.subheader("🔀 Pixel Sort")
        col1, col2, col3 = st.columns(3)
        with col1: ps_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
        with col2: ps_threshold  = st.slider("Soglia luma", 0.1, 1.0, 0.5, 0.05)
        with col3: ps_direction  = st.slider("Direzione (0=righe, 1=col)", 0.0, 1.0, 0.3, 0.05)
        params = (ps_intensity, ps_threshold, ps_direction)

    elif effect_type == 'channel_shift':
        st.subheader("🌈 Channel Shift")
        col1, col2, col3 = st.columns(3)
        with col1: cs_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
        with col2: cs_spread    = st.slider("Spread", 0.1, 3.0, 1.0, 0.1)
        with col3: cs_mode      = st.slider("Verticale (0=no, 1=sì)", 0.0, 1.0, 0.3, 0.05)
        params = (cs_intensity, cs_spread, cs_mode)

    elif effect_type == 'datamosh':
        st.subheader("💾 Datamosh")
        col1, col2, col3 = st.columns(3)
        with col1: dm_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
        with col2: dm_block_size = st.slider("Block size", 0.1, 3.0, 1.0, 0.1)
        with col3: dm_chaos      = st.slider("Chaos", 0.1, 3.0, 1.0, 0.1)
        params = (dm_intensity, dm_block_size, dm_chaos)

    elif effect_type == 'byte_corrupt':
        st.subheader("🦠 Byte Corrupt")
        col1, col2, col3 = st.columns(3)
        with col1: bc_intensity   = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
        with col2: bc_chunk_size  = st.slider("Chunk size", 0.1, 3.0, 1.0, 0.1)
        with col3: bc_randomize   = st.slider("Random (0=dup, 1=rand)", 0.0, 1.0, 0.7, 0.05)
        params = (bc_intensity, bc_chunk_size, bc_randomize)

    elif effect_type == 'slice_shift':
        st.subheader("✂️ Slice Shift")
        col1, col2, col3 = st.columns(3)
        with col1: ss_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
        with col2: ss_num_slices = st.slider("Numero slice", 0.1, 3.0, 1.0, 0.1)
        with col3: ss_drift      = st.slider("Drift", 0.1, 3.0, 1.0, 0.1)
        params = (ss_intensity, ss_num_slices, ss_drift)

    elif effect_type == 'vhs':
        st.subheader("📼 Parametri VHS")
        col1, col2, col3 = st.columns(3)
        with col1:
            vhs_intensity = st.slider("Intensità generale", 0.1, 3.0, 1.0, 0.1)
        with col2:
            scanline_freq = st.slider("Frequenza scanline", 0.1, 3.0, 1.0, 0.1)
        with col3:
            color_shift = st.slider("Color shift", 0.1, 3.0, 1.0, 0.1)
        
        params = (vhs_intensity, scanline_freq, color_shift)
        audio_params_override = None
        
        if include_audio:
            st.subheader("🎵 Parametri Audio VHS")
            col1, col2, col3 = st.columns(3)
            with col1:
                wow_flutter = st.slider("Wow & Flutter", 0.1, 3.0, 1.0, 0.1)
            with col2:
                tape_hiss = st.slider("Tape Hiss", 0.1, 3.0, 1.0, 0.1)
            audio_params_override = (vhs_intensity, wow_flutter, tape_hiss)

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
        audio_params_override = None
        
        if include_audio:
            st.subheader("🎵 Parametri Audio Distruttivo")
            col1, col2, col3 = st.columns(3)
            with col1:
                chaos_level = st.slider("Livello chaos", 0.1, 3.0, 1.0, 0.1)
            with col2:
                skip_prob = st.slider("Probabilità skip", 0.1, 3.0, 1.0, 0.1)
            with col3:
                reverse_prob = st.slider("Probabilità reverse", 0.1, 3.0, 1.0, 0.1)
            audio_params_override = (chaos_level, skip_prob, reverse_prob)

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
        audio_params_override = None
        
        if include_audio:
            st.subheader("🎵 Parametri Audio Noise")
            col1, col2, col3 = st.columns(3)
            with col1:
                digital_artifacts = st.slider("Artefatti digitali", 0.1, 3.0, 1.0, 0.1)
            with col2:
                bit_crush = st.slider("Bit Crushing", 0.1, 3.0, 1.0, 0.1)
            audio_params_override = (noise_intensity, digital_artifacts, bit_crush)

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
        audio_params_override = None
        
        if include_audio:
            st.subheader("🎵 Parametri Audio Broken TV")
            col1, col2, col3 = st.columns(3)
            with col1:
                static_intensity = st.slider("Intensità static", 0.1, 3.0, 1.0, 0.1)
            with col2:
                channel_separation = st.slider("Separazione canali", 0.1, 3.0, 1.0, 0.1)
            with col3:
                frequency_drift = st.slider("Drift frequenza", 0.1, 3.0, 1.0, 0.1)
            audio_params_override = (static_intensity, channel_separation, frequency_drift)

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
        audio_params_override = None  # random sceglie da solo

    # audio_params_override per combined: usa params dict direttamente (già contiene chiavi audio)
    if effect_type == 'combined':
        audio_params_override = None  # combined usa params dict che già include i valori audio

    # Limita frame per video lunghi
    max_frames = st.number_input("🎬 Limite frame (0 = nessun limite)", min_value=0, max_value=10000, value=0)

    # KEYFRAME — solo per effetti singoli (non combined/random)
    kf_df = None
    use_keyframes = False
    if effect_type not in ['combined', 'random']:
        st.markdown("---")
        use_keyframes = st.toggle("⏱️ Animazione intensità nel tempo",
            help="Interpola l'intensità dell'effetto dall'inizio alla fine del video.")
        if use_keyframes:
            col_kf1, col_kf2 = st.columns(2)
            with col_kf1:
                kf_start = st.slider("Intensità inizio", 0.0, 3.0, 0.5, 0.1, key="kf_start")
            with col_kf2:
                kf_end = st.slider("Intensità fine", 0.0, 3.0, 1.5, 0.1, key="kf_end")
            import pandas as pd
            cap_tmp = cv2.VideoCapture(video_path)
            fps_tmp = cap_tmp.get(cv2.CAP_PROP_FPS) or 24
            frames_tmp = cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap_tmp.release()
            dur_est = round(frames_tmp / fps_tmp, 1) if fps_tmp > 0 else 10.0
            kf_df = pd.DataFrame({"Secondo": [0.0, dur_est], "Intensita'": [kf_start, kf_end]})

    # --- ANTEPRIMA LIVE (si aggiorna ad ogni cambio slider) ---
    st.markdown("---")
    st.caption("🖼️ Anteprima live")
    try:
        cap_live = cv2.VideoCapture(video_path)
        total_f_live = int(cap_live.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_live.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_f_live // 3))
        ret_live, frame_live = cap_live.read()
        cap_live.release()
        if ret_live:
            if effect_type == 'pixel_sort':
                prev_frame = glitch_pixel_sort(frame_live, *params)
            elif effect_type == 'channel_shift':
                prev_frame = glitch_channel_shift(frame_live, *params)
            elif effect_type == 'datamosh':
                # per preview datamosh usiamo frame_live come prev (stesso frame = ghosting)
                prev_frame = glitch_datamosh(frame_live, frame_live, *params)
            elif effect_type == 'byte_corrupt':
                prev_frame = glitch_byte_corrupt(frame_live, *params)
            elif effect_type == 'slice_shift':
                prev_frame = glitch_slice_shift(frame_live, *params)
            elif effect_type == 'vhs':
                prev_frame = glitch_vhs_frame(frame_live, *params)
            elif effect_type == 'distruttivo':
                prev_frame = glitch_distruttivo_frame(frame_live, *params)
            elif effect_type == 'noise':
                prev_frame = glitch_noise_frame(frame_live, *params)
            elif effect_type == 'broken_tv':
                prev_frame = glitch_broken_tv_frame(frame_live, *params)
            elif effect_type == 'combined':
                prev_frame = frame_live.copy()
                if params.get("apply_vhs"):
                    prev_frame = glitch_vhs_frame(prev_frame, params.get("vhs_intensity",1.0), params.get("vhs_scanline_freq",1.0), params.get("vhs_color_shift",1.0))
                if params.get("apply_distruttivo"):
                    prev_frame = glitch_distruttivo_frame(prev_frame, params.get("dest_block_size",1.0), params.get("dest_num_blocks",1.0), params.get("dest_displacement",1.0))
                if params.get("apply_noise"):
                    prev_frame = glitch_noise_frame(prev_frame, params.get("noise_intensity",1.0), params.get("noise_coverage",1.0), params.get("noise_chaos",1.0))
                if params.get("apply_broken_tv"):
                    prev_frame = glitch_broken_tv_frame(prev_frame, params.get("tv_shift_intensity",1.0), params.get("tv_line_height",1.0), params.get("tv_flicker_prob",1.0))
                if params.get("apply_pixel_sort"):
                    prev_frame = glitch_pixel_sort(prev_frame, params.get("ps_intensity",1.0), params.get("ps_threshold",0.5), params.get("ps_direction",0.3))
                if params.get("apply_channel_shift"):
                    prev_frame = glitch_channel_shift(prev_frame, params.get("cs_intensity",1.0), params.get("cs_spread",1.0), params.get("cs_mode",0.3))
                if params.get("apply_slice_shift"):
                    prev_frame = glitch_slice_shift(prev_frame, params.get("ss_intensity",1.0), params.get("ss_num_slices",1.0), params.get("ss_drift",1.0))
            elif effect_type == 'random':
                random_level = params[0] if params else 1.0
                chosen = random.choice(['pixel_sort','channel_shift','byte_corrupt','slice_shift','vhs','broken_tv','noise','distruttivo'])
                rp = tuple(random.uniform(0.5, 1.5) * random_level for _ in range(3))
                fn_map = {'pixel_sort': glitch_pixel_sort, 'channel_shift': glitch_channel_shift, 'byte_corrupt': glitch_byte_corrupt, 'slice_shift': glitch_slice_shift, 'vhs': glitch_vhs_frame, 'broken_tv': glitch_broken_tv_frame, 'noise': glitch_noise_frame, 'distruttivo': glitch_distruttivo_frame}
                prev_frame = fn_map[chosen](frame_live, *rp)
            else:
                prev_frame = frame_live
            prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            st.image(prev_rgb, use_column_width=True)
    except Exception as e:
        st.caption(f"Anteprima non disponibile: {e}")

    # --- PROPORZIONI EXPORT ---
    st.markdown("---")
    st.subheader("📐 Proporzioni export")
    aspect_ratio = st.radio(
        "Formato output:",
        ["Originale", "16:9", "9:16", "1:1"],
        horizontal=True
    )

    # Bottone per processare
    if st.button("🚀 Processa Video"):
        if not any([effect_type != 'combined' or any(params.values()) if isinstance(params, dict) else True]):
            st.warning("⚠️ Seleziona almeno un effetto per la modalità combinata!")
        else:
            with st.spinner("🔥 Processando il video..."):
                # Calcola envelope keyframe se attivo
                kf_envelope = None
                if use_keyframes and kf_df is not None and len(kf_df) >= 2:
                    cap_info = cv2.VideoCapture(video_path)
                    fps_info = int(cap_info.get(cv2.CAP_PROP_FPS)) or 24
                    frames_info = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_info.release()
                    if max_frames > 0:
                        frames_info = min(frames_info, max_frames)
                    kf_envelope = interpolate_keyframes(kf_df, fps_info, frames_info)

                result_path = process_video(video_path, effect_type, params, max_frames, include_audio, kf_envelope, audio_params_override, aspect_ratio)
                
                if result_path:
                    st.success("✅ Video processato!")
                    with st.spinner("🗜️ H.264..."):
                        h264_path = recompress_h264(result_path)

                    orig_size  = get_file_size_mb(video_path)
                    fps_v, w_v, h_v, frames_v, dur_v = get_video_info(video_path)
                    out_size   = get_file_size_mb(h264_path)

                    st.session_state.report_data = build_report(
                        uploaded_file.name, orig_size, out_size,
                        fps_v, w_v, h_v, frames_v, dur_v,
                        effect_type, params, include_audio, kf_df
                    )
                    st.session_state.h264_path = h264_path
                    st.session_state.effect_name_saved = {
                        "pixel_sort":"PixelSort","channel_shift":"ChannelShift",
                        "datamosh":"Datamosh","byte_corrupt":"ByteCorrupt",
                        "slice_shift":"SliceShift","vhs":"VHS","distruttivo":"Distruttivo",
                        "noise":"Noise","combined":"Combinato","broken_tv":"BrokenTV","random":"Random"
                    }[effect_type]
                    st.session_state.orig_filename = uploaded_file.name
                    st.session_state.video_ready = True

                    for p in [result_path, video_path]:
                        try: os.unlink(p)
                        except: pass
                else:
                    st.error("❌ Errore durante il processing.")

else:
    st.info("👆 Carica un video per iniziare!")

# --- AUDIO STANDALONE GLITCH (fuori dall'if video) ---
if uploaded_audio_file is not None and check_ffmpeg():
    st.markdown("---")
    st.subheader("🎵 Glitch Audio Standalone")
    
    ffmpeg_available = check_ffmpeg()
    audio_effect = st.selectbox(
        "Effetto audio:",
        ["vhs", "distruttivo", "noise", "broken_tv"],
        format_func=lambda x: {"vhs":"📼 VHS","distruttivo":"💥 Distruttivo","noise":"📺 Noise","broken_tv":"📻 Broken TV"}[x],
        key="audio_effect_sel"
    )
    col1, col2, col3 = st.columns(3)
    with col1: a_p1 = st.slider("Param 1", 0.1, 3.0, 1.0, 0.1, key="a_p1")
    with col2: a_p2 = st.slider("Param 2", 0.1, 3.0, 1.0, 0.1, key="a_p2")
    with col3: a_p3 = st.slider("Param 3", 0.1, 3.0, 1.0, 0.1, key="a_p3")

    if st.button("🔊 Glitcha Audio"):
        with st.spinner("Glitchando audio..."):
            ext = os.path.splitext(uploaded_audio_file.name)[1].lower()
            fd_ain, audio_in_path = tempfile.mkstemp(suffix=ext)
            os.close(fd_ain)
            with open(audio_in_path, 'wb') as f:
                f.write(uploaded_audio_file.read())
            
            # Converti in WAV per processing
            wav_path = convert_audio_to_wav(audio_in_path)
            if wav_path:
                glitched_wav = process_audio_glitch(wav_path, audio_effect, (a_p1, a_p2, a_p3))
                if glitched_wav:
                    # Converti in mp3 per output leggero
                    fd_out, mp3_out = tempfile.mkstemp(suffix='_glitch.mp3')
                    os.close(fd_out)
                    subprocess.run([
                        'ffmpeg', '-i', glitched_wav,
                        '-codec:a', 'libmp3lame', '-qscale:a', '4',
                        mp3_out, '-y'
                    ], capture_output=True)
                    st.session_state.glitched_audio_path = mp3_out
                    st.session_state.glitched_audio_name = uploaded_audio_file.name
                    # Pulizia temp
                    for p in [audio_in_path, wav_path, glitched_wav]:
                        try: os.unlink(p)
                        except: pass
                    st.success("✅ Audio glitchato!")

    if st.session_state.glitched_audio_path and os.path.exists(st.session_state.glitched_audio_path):
        st.audio(st.session_state.glitched_audio_path)
        with open(st.session_state.glitched_audio_path, 'rb') as af:
            orig_stem = os.path.splitext(st.session_state.get('glitched_audio_name', 'audio'))[0]
            st.download_button("📥 Scarica Audio Glitch (mp3)", af,
                file_name=f"glitch_{orig_stem}.mp3", mime="audio/mpeg", key="down_audio")

# RISULTATI PERSISTENTI
if st.session_state.video_ready:
    st.markdown("---")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        if st.session_state.h264_path and os.path.exists(st.session_state.h264_path):
            with open(st.session_state.h264_path, 'rb') as vf:
                st.download_button(
                    label="📥 Scarica video (H.264)",
                    data=vf,
                    file_name=f"glitched_{st.session_state.effect_name_saved}_{st.session_state.orig_filename}",
                    mime="video/mp4",
                    key="down_v"
                )
    with c_d2:
        st.download_button(
            label="📄 Scarica Report",
            data=st.session_state.report_data,
            file_name="report_glitch.txt",
            key="down_r"
        )
    st.text_area("📄 REPORT", st.session_state.report_data, height=320)

# Footer
st.markdown("---")
st.markdown("🎬 **VideoDistruktor by loop507** - Trasforma i tuoi video in opere d'arte glitched!")
if not check_ffmpeg():
    st.markdown("⚠️ *Per abilitare gli effetti audio, installa FFmpeg sul sistema*")
