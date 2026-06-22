import os
# Disabilita JIT numba — previene segfault su Streamlit Cloud
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import streamlit as st
import numpy as np
import tempfile
from PIL import Image, ImageEnhance, ImageOps
import random
import cv2
import subprocess
import shutil
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import time
import math

# ─────────────────────────────────────────────────────────────────────────────
# 1. AUDIO REACTIVE ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_audio_for_video(audio_path, fps, total_frames):
    """
    Analizza l'audio spettrale e ritorna dizionario di envelope per-frame.
    Risolve i conflitti di memoria con importazione lazy di librosa.
    """
    try:
        import librosa  # Import lazy per evitare segfault numba JIT al bootstrap
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        hop = max(1, int(sr / fps))

        # RMS - Energia globale
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        if rms.max() > 0:
            rms = rms / rms.max()

        # Onset Strength & Beats
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        _, beats_idx = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
        beats = np.zeros_like(rms)
        if len(beats_idx) > 0:
            beats[beats_idx] = 1.0

        # Filtri frequenziali avanzati via STFT
        stft = np.abs(librosa.stft(y, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr)
        
        low_mask = freqs <= 250
        mid_mask = (freqs > 250) & (freqs < 2500)
        high_mask = freqs >= 4000
        
        low_energy = np.sum(stft[low_mask, :], axis=0)
        mid_energy = np.sum(stft[mid_mask, :], axis=0)
        high_energy = np.sum(stft[high_mask, :], axis=0)
        
        if low_energy.max() > 0: low_energy /= low_energy.max()
        if mid_energy.max() > 0: mid_energy /= mid_energy.max()
        if high_energy.max() > 0: high_energy /= high_energy.max()

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        if centroid.max() > 0:
            centroid = centroid / centroid.max()

        out = {}
        for k, arr in [('rms', rms), ('beats', beats), ('low_freq', low_energy), 
                       ('mid_freq', mid_energy), ('high_freq', high_energy), ('spectral', centroid)]:
            if len(arr) < total_frames:
                padded = np.zeros(total_frames)
                padded[:len(arr)] = arr
                padded[len(arr):] = arr[-1] if len(arr) > 0 else 0
                out[k] = padded
            else:
                out[k] = arr[:total_frames]
        return out
    except Exception:
        dummy = np.zeros(total_frames)
        return {'rms': dummy, 'beats': dummy, 'low_freq': dummy, 'mid_freq': dummy, 'high_freq': dummy, 'spectral': dummy}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIGITAL VIDEO GLITCH FUNCTIONS (PARTE A)
# ─────────────────────────────────────────────────────────────────────────────

def glitch_pixel_sort(frame, intensity=1.0, threshold=0.5, direction=0.5):
    """
    CORRETTO: Pixel sorting per luminosità su righe o colonne.
    Risolto il problema del blocco reattivo modificando la maschera di soglia.
    """
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Mappatura soglia dinamica e sicura
        thr = np.clip(threshold * 0.9, 0.05, 0.95)
        num_rows = max(1, int(h * min(intensity, 1.0)))
        row_indices = np.random.choice(h, num_rows, replace=False)
        
        for y in row_indices:
            # Segno invertito (>) per catturare i segmenti luminosi sopra la soglia audio
            mask = gray[y] > thr
            if not np.any(mask):
                continue
            starts = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == 1)[0]
            ends   = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])) == -1)[0]
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                seg = arr[y, s:e]
                lum = gray[y, s:e]
                order = np.argsort(lum)
                arr[y, s:e] = seg[order]
                
        # Scansione verticale se la direzione supera il valore intermedio
        if direction > 0.5:
            num_cols = max(1, int(w * min((direction - 0.5) * 2 * intensity, 1.0)))
            col_indices = np.random.choice(w, num_cols, replace=False)
            for x in col_indices:
                mask = gray[:, x] > thr
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

def glitch_channel_shift(frame, split_x=10, split_y=0):
    """Sfalsa l'allineamento dei canali RGB (Miscele cromatiche 3D Anaglifiche)."""
    try:
        if split_x == 0 and split_y == 0:
            return frame
        arr = frame.copy()
        h, w = arr.shape[:2]
        b, g, r = cv2.split(arr)
        
        M_r = np.float32([[1, 0, split_x], [0, 1, split_y]])
        M_b = np.float32([[1, 0, -split_x], [0, 1, -split_y]])
        
        r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return cv2.merge([b, g, r])
    except Exception:
        return frame

def glitch_jpeg_compression(frame, quality=10):
    """Degrada il flusso generando artefatti a macroblocchi (Lossy Compression)."""
    try:
        q = int(np.clip(quality, 1, 100))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encimg, 1)
    except Exception:
        return frame

def glitch_vhs_lines(frame, intensity=0.5, noise_amount=0.2):
    """Simula lo strappo orizzontale del nastro VHS e linee di disturbo."""
    try:
        arr = frame.copy()
        h, w, c = arr.shape
        num_lines = int(h * intensity * 0.1)
        if num_lines > 0:
            for _ in range(num_lines):
                y = random.randint(0, h-1)
                rh = random.randint(1, max(2, int(h*0.03)))
                shift = random.randint(-int(w*0.06 * intensity), int(w*0.06 * intensity))
                if shift == 0: continue
                y1 = min(y + rh, h)
                M = np.float32([[1, 0, shift], [0, 1, 0]])
                arr[y:y1, :] = cv2.warpAffine(arr[y:y1, :], M, (w, y1-y), borderMode=cv2.BORDER_REFLECT)
                
        if noise_amount > 0:
            noise = np.random.randint(-int(255*noise_amount), int(255*noise_amount), size=arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        return frame

def glitch_scanlines(frame, opacity=0.3, thickness=2):
    """Disegna scanline analogiche scure per emulare monitor CRT e TV a tubo."""
    try:
        arr = frame.copy()
        h, w, c = arr.shape
        mask = np.ones((h, w, 1), dtype=np.float32)
        for y in range(0, h, thickness * 2):
            mask[y:y+thickness, :] = 1.0 - opacity
        return (arr.astype(np.float32) * mask).astype(np.uint8)
    except Exception:
        return frame

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIGITAL VIDEO GLITCH FUNCTIONS (PARTE B — EFFETTI AVANZATI)
# ─────────────────────────────────────────────────────────────────────────────

def glitch_noise_interference(frame, intensity=0.3):
    """Aggiunge disturbo statico analogico e barre di rumore neve."""
    try:
        arr = frame.copy()
        h, w, c = arr.shape
        if intensity <= 0:
            return frame
        noise = np.random.choice([0, 128, 255], size=(h, w, 1), p=[1 - intensity*0.5, intensity*0.3, intensity*0.2]).astype(np.uint8)
        noise = cv2.merge([noise, noise, noise])
        return cv2.addWeighted(arr, 1.0 - intensity*0.4, noise, intensity*0.4, 0)
    except Exception:
        return frame

def glitch_datamosh_mblock(frame, prev_frame, block_size=16, intensity=0.5):
    """Mantiene i macroblocchi del frame precedente simulando la perdita di I-Frame."""
    try:
        if prev_frame is None or intensity <= 0:
            return frame
        arr = frame.copy()
        h, w, c = arr.shape
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if random.random() < intensity * 0.4:
                    yb = min(y + block_size, h)
                    xb = min(x + block_size, w)
                    arr[y:yb, x:xb] = prev_frame[y:yb, x:xb]
        return arr
    except Exception:
        return frame

def glitch_byte_corrupt(frame, intensity=1.0, frequency=1.0, chaos=1.0):
    """Simula la corruzione dei byte grezzi della matrice comprimendo ed alterando gli array."""
    try:
        h, w, c = frame.shape
        flat = frame.flatten()
        num_corruptions = max(1, int(10 * intensity * chaos))
        for _ in range(num_corruptions):
            pos = random.randint(0, len(flat) - 1)
            length = max(1, int(5 * frequency))
            flat[pos:pos+length] = random.randint(0, 255)
        return flat.reshape((h, w, c))
    except Exception:
        return frame

def glitch_slice_shift(frame, intensity=1.0, slices=10, drift=0.5):
    """Taglia l'immagine in sezioni orizzontali e le trasla in modo asincrono."""
    try:
        arr = frame.copy()
        h, w, c = arr.shape
        num_slices = max(2, int(slices * intensity))
        slice_h = h // num_slices
        for i in range(num_slices):
            if random.random() < drift:
                y_start = i * slice_h
                y_end = (i + 1) * slice_h if i < num_slices - 1 else h
                shift = int(random.randint(-int(w*0.1), int(w*0.1)) * intensity)
                arr[y_start:y_end, :] = np.roll(arr[y_start:y_end, :], shift, axis=1)
        return arr
    except Exception:
        return frame

def glitch_echo_smear(frame, prev_frame, intensity=1.0, weight=0.5, blur=0.0):
    """Crea una scia temporale (ghosting) miscelando il frame attuale con il precedente."""
    try:
        if prev_frame is None:
            return frame
        alpha = np.clip(weight * intensity, 0.0, 0.95)
        blended = cv2.addWeighted(frame, 1.0 - alpha, prev_frame, alpha, 0)
        if blur > 0.1:
            k = int(blur * 5) * 2 + 1
            blended = cv2.GaussianBlur(blended, (k, k), 0)
        return blended
    except Exception:
        return frame

def glitch_rgb_wave(frame, intensity=1.0, amp=10, freq=0.1):
    """Applica una distorsione sinusoidale indipendente sui tre canali di colore."""
    try:
        arr = frame.copy()
        h, w, c = arr.shape
        b, g, r = cv2.split(arr)
        amplitude = amp * intensity
        
        # Mappa di deformazione ondulatoria
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Onde sfasate per R e B
        wave_r = map_x + amplitude * np.sin(map_y * freq)
        wave_b = map_x - amplitude * np.cos(map_y * freq)
        
        r = cv2.remap(r, wave_r, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b = cv2.remap(b, wave_b, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return cv2.merge([b, g, r])
    except Exception:
        return frame

def glitch_mirror_blocks(frame, intensity=1.0, block_count=4, mode=0.0):
    """Seleziona regioni casuali dell'immagine e le inverte specularmente."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        n = max(1, int(block_count * intensity))
        for _ in range(n):
            bh = random.randint(30, h // 2)
            bw = random.randint(30, w // 2)
            y = random.randint(0, h - bh)
            x = random.randint(0, w - bw)
            
            block = arr[y:y+bh, x:x+bw]
            if mode > 0.5:
                arr[y:y+bh, x:x+bw] = cv2.flip(block, 0) # Inversione verticale
            else:
                arr[y:y+bh, x:x+bw] = cv2.flip(block, 1) # Inversione orizzontale
        return arr
    except Exception:
        return frame

def glitch_color_quantize(frame, intensity=1.0, bits=4, dither=0.0):
    """Riduce la profondità di colore creando un effetto posterizzazione a bit ridotti."""
    try:
        b = max(1, min(8, int(bits / (intensity + 1e-3) if intensity > 0 else bits)))
        shift = 8 - b
        arr = (frame >> shift) << shift
        return arr
    except Exception:
        return frame

def glitch_moire(frame, intensity=1.0, scale=1.0, angle=0.0):
    """Sovrappone griglie geometriche rotanti generando pattern d'interferenza Moiré."""
    try:
        h, w, c = frame.shape
        mask = np.zeros((h, w), dtype=np.float32)
        freq = int(20 * scale)
        for y in range(0, h, freq):
            mask[y:y+max(1, freq//2), :] = 1.0
            
        # Rotazione della griglia d'interferenza
        M = cv2.getRotationMatrix2D((w//2, h//2), angle * 360, 1.0)
        rotated_mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        rotated_mask = cv2.merge([rotated_mask, rotated_mask, rotated_mask])
        
        moire_layer = (frame.astype(np.float32) * (1.0 - (rotated_mask * intensity * 0.5))).astype(np.uint8)
        return moire_layer
    except Exception:
        return frame

def glitch_feedback_loop(frame, prev_frame, intensity=1.0, zoom=1.01, rotation=2.0):
    """Simula il feedback ottico (puntamento telecamera sul monitor) che ruota e zooma."""
    try:
        if prev_frame is None or intensity <= 0:
            return frame
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotation * intensity, zoom)
        warped = cv2.warpAffine(prev_frame, M, (w, h), borderMode=cv2.BORDER_WRAP)
        alpha = np.clip(0.25 + 0.45 * intensity, 0.1, 0.8)
        blended = cv2.addWeighted(frame.astype(np.float32), 1.0 - alpha, warped.astype(np.float32), alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_pixel_drift(frame, intensity=1.0, steps=1.0, direction=0.0):
    """Sposta i pixel orizzontalmente o verticalmente creando scie trascinate."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        num_drifts = max(1, int(10 + 40 * intensity))
        drift_len = max(2, int(15 * steps))
        for _ in range(num_drifts):
            if direction < 0.5:
                x = random.randint(0, max(0, w - drift_len - 1))
                y = random.randint(0, h - 1)
                arr[y, x:w] = arr[y, x + drift_len if x + drift_len < w else x]
            else:
                x = random.randint(0, w - 1)
                y = random.randint(0, max(0, h - drift_len - 1))
                arr[y:h, x] = arr[y + drift_len if y + drift_len < h else y, x]
        return arr
    except Exception:
        return frame

def glitch_slit_scan(frame, intensity=1.0, speed=1.0, mode=0.0):
    """Simula l'effetto Slit-Scan ritardando la scansione temporale delle linee."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        shift = int(speed * 10 * intensity)
        if shift < 1:
            return frame
        for y in range(h):
            s = int(shift * np.sin(y * np.pi / h))
            if s != 0:
                arr[y] = np.roll(arr[y], s, axis=0 if mode > 0.5 else 1)
        return arr
    except Exception:
        return frame

def glitch_thermal(frame, intensity=1.0, hue_shift=0.5, contrast=1.0):
    """Mappa i colori in una palette termica falsa basata sulla luminanza."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if contrast != 1.0:
            gray = cv2.LUT(gray, np.clip(np.array([((i/255.0)**contrast)*255 for i in range(256)]), 0, 255).astype(np.uint8))
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        if hue_shift != 0.5:
            hsv = cv2.cvtColor(thermal, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:,:,0] = (hsv[:,:,0] + int((hue_shift - 0.5) * 180)) % 180
            thermal = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(frame, 1.0 - intensity, thermal, intensity, 0)
    except Exception:
        return frame

def glitch_ascii_glitch(frame, intensity=1.0, scale=1.0, font_size=0.5):
    """Applica un subset di matrici di testo ASCII sovrapposte distruggendo i contorni."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        f_size = max(0.2, font_size * 0.8)
        chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
        step = max(8, int(16 * scale))
        for y in range(0, h, step):
            for x in range(0, w, step):
                if random.random() < intensity:
                    b, g, r = arr[y, x]
                    c = random.choice(chars)
                    cv2.putText(arr, c, (x, y), cv2.FONT_HERSHEY_SIMPLEX, f_size, (int(b), int(g), int(r)), 1)
        return arr
    except Exception:
        return frame

def glitch_halftone(frame, intensity=1.0, dot_size=1.0, angle=0.5):
    """Simula una retinatura di stampa CMYK/Monocromatica sfalsata."""
    try:
        arr = frame.copy().astype(np.float32)
        h, w = arr.shape[:2]
        r_max = max(2, int(6 * dot_size))
        grid = max(4, int(8 * dot_size))
        for y in range(0, h, grid):
            for x in range(0, w, grid):
                b, g, r = arr[y, x]
                lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                radius = int(r_max * (1.0 - lum) * intensity)
                if radius > 0:
                    cv2.circle(arr, (x, y), radius, (0, 0, 0), -1)
        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_chroma_pulse(frame, intensity=1.0, frequency=1.0, saturation=1.0):
    """Modula in modo pulsante i canali di crominanza (Cr/Cb o U/V)."""
    try:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        pulse = 1.0 + intensity * 0.5 * np.sin(frequency * time.time() * np.pi * 2)
        ycrcb[:,:,1] = np.clip(ycrcb[:,:,1] * pulse * saturation, 0, 255)
        ycrcb[:,:,2] = np.clip(ycrcb[:,:,2] * (2.0 - pulse) * saturation, 0, 255)
        return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    except Exception:
        return frame

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUDIO GLITCH LABS ENGINE (MANIPOLAZIONE AUDIO GREZZA)
# ─────────────────────────────────────────────────────────────────────────────

def process_audio_glitch(input_audio_path, effect_type, p1, p2):
    """Applica manipolazioni digitali distruttive direttamente sul campionamento audio PCM."""
    out_audio_path = tempfile.mktemp(suffix=".wav")
    try:
        sr, data = wavfile.read(input_audio_path)
        is_stereo = len(data.shape) > 1
        float_data = data.astype(np.float32) / 32768.0

        if effect_type == "vhs":
            # Wow and flutter (vibrato ciclico analogico) + bitcrush leggero
            t = np.arange(len(float_data))
            flutter = 1.0 + (p1 * 0.02) * np.sin(2 * np.pi * 4.0 * t / sr)
            new_indices = np.clip(t * flutter, 0, len(float_data) - 1).astype(np.int32)
            float_data = float_data[new_indices]
            
        elif effect_type == "destructive":
            # Bitcrush brutale + Overdrive clipping estremo
            bits = max(2, int(16 - (p1 * 12)))
            quant_steps = 2 ** bits
            float_data = np.round(float_data * quant_steps) / quant_steps
            float_data = float_data * (1.0 + p2 * 10.0)
            float_data = np.clip(float_data, -1.0, 1.0)
            
        elif effect_type == "noise":
            # Sfondo bianco continuo ed impulsi di crackle statico digitali
            noise = np.random.normal(0, p1 * 0.15, float_data.shape)
            float_data += noise
            num_cracks = int(p2 * 100)
            for _ in range(num_cracks):
                pos = random.randint(0, len(float_data)-1)
                length = random.randint(10, 300)
                float_data[pos:pos+length] = random.choice([-1.0, 1.0, 0.0])
                
        elif effect_type == "broken_tv":
            # Sincronia audio a scatti (Stuttering ciclico di buffer audio)
            stutter_len = max(512, int(sr * 0.05 * p1))
            i = 0
            while i < len(float_data):
                if random.random() < p2 * 0.3:
                    repeats = random.randint(2, 6)
                    block = float_data[i:i+stutter_len]
                    if len(block) == 0: break
                    for _ in range(repeats):
                        end_pos = min(i + len(block), len(float_data))
                        float_data[i:end_pos] = block[:end_pos-i]
                        i += len(block)
                else:
                    i += stutter_len

        # Limitatore finale per prevenire danni di clipping all'hardware
        float_data = np.clip(float_data, -0.99, 0.99)
        out_data = (float_data * 32767.0).astype(np.int16)
        wavfile.write(out_audio_path, sr, out_data)
        return out_audio_path
    except Exception:
        return input_audio_path

# ─────────────────────────────────────────────────────────────────────────────
# 4. FFMPEG BACKEND TOOLS (ESTRAZIONE, COPIATURA E RICODIFICA UNIVERSALE)
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg():
    """Verifica la presenza di FFmpeg nel PATH di sistema."""
    return shutil.which("ffmpeg") is not None

def extract_audio(video_path):
    """Estrae in modo nativo la traccia audio dal file video in formato WAV PCM."""
    audio_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2', audio_path, '-y'
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 100:
        return audio_path
    return None

def convert_audio_to_wav(audio_path):
    """Converte qualsiasi traccia audio esterna caricata in WAV standard per l'analisi spettrale."""
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        'ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2', out_path, '-y'
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and os.path.exists(out_path):
        return out_path
    return audio_path

def combine_video_audio(video_path, audio_path, output_path):
    """Combina la traccia video muta e l'audio distorto forzando la sincronizzazione."""
    try:
        # Rimosso '-shortest' per evitare tagli bruschi indesiderati alla fine del video.
        # Utilizzato '-async 1' per vincolare l'allineamento dei flussi dall'inizio.
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-async', '1',
            output_path, '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# 5. CORE PIPELINE ORCHESTRATOR & AUDIO-REACTIVE MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def apply_audio_reactive_modulation(base_params, effect_type, audio_env, frame_idx, ar_intensity):
    """Modula dinamicamente i parametri di controllo in base alle bande di frequenza audio."""
    if not audio_env or frame_idx >= len(audio_env['rms']):
        return base_params

    p1, p2, p3 = base_params
    rms_val = audio_env['rms'][frame_idx] * ar_intensity
    low_val = audio_env['low_freq'][frame_idx] * ar_intensity
    high_val = audio_env['high_freq'][frame_idx] * ar_intensity
    beat_val = audio_env['beats'][frame_idx] * ar_intensity

    # Strategie di mappatura differenziata per tipologia di effetto
    if effect_type in ['vhs', 'vhs_lines', 'slice_shift']:
        p1 = np.clip(p1 * (1.0 + rms_val), 0.0, 3.0)
        p2 = np.clip(p2 * (1.0 + low_val), 0.0, 3.0)
    elif effect_type in ['distruttivo', 'jpeg_comp', 'byte_corrupt']:
        p1 = np.clip(p1 * (1.0 + beat_val * 1.5), 0.0, 3.0)
        p3 = np.clip(p3 * (1.0 + high_val), 0.0, 3.0)
    elif effect_type in ['pixel_sort', 'pixel_drift', 'slit_scan']:
        p1 = np.clip(p1 * (1.0 + rms_val * 1.2), 0.0, 3.0)
        p2 = np.clip(p2 * (1.0 - high_val * 0.4), 0.01, 0.99) # La soglia si abbassa sui picchi alti
    elif effect_type in ['channel_shift', 'rgb_wave', 'chroma_pulse']:
        p1 = np.clip(p1 * (1.0 + low_val * 1.5), 0.0, 3.0)
        p2 = np.clip(p2 * (1.0 + beat_val), 0.0, 3.0)
    elif effect_type in ['feedback_loop', 'echo_smear']:
        p1 = np.clip(p1 * (1.0 + rms_val), 0.0, 3.0)
    
    return (p1, p2, p3)

def apply_selected_effects(frame, prev_frame, effect_type, params, frame_idx, audio_env=None, ar_intensity=1.0):
    """Esegue l'instr
