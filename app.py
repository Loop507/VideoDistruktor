import os
# Disabilita JIT numba — previene segfault su Streamlit Cloud
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import streamlit as st
import numpy as np
import tempfile
from PIL import Image
import random
import cv2
import subprocess
import shutil
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import time
# librosa importato lazy in analyze_audio_for_video() per evitare segfault da numba JIT

# ─────────────────────────────────────────────
# AUDIO REACTIVE ANALYSIS
# ─────────────────────────────────────────────

def analyze_audio_for_video(audio_path, fps, total_frames):
    """
    Analizza l'audio e ritorna dizionario di envelope per-frame:
      - rms       : energia globale (0-1)
      - beats     : 1.0 sui beat, 0 altrove
      - low_freq  : energia basse frequenze (0-1)
      - high_freq : energia alte frequenze (0-1)
      - spectral  : centroide spettrale normalizzato (0-1)
    """
    try:
        import librosa  # import lazy: evita segfault numba al bootstrap
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        hop = max(1, int(sr / fps))

        # RMS energy per frame
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms = rms / (rms.max() + 1e-8)

        # Beat detection
        tempo, beat_frames_raw = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beats = np.zeros(len(rms), dtype=np.float32)
        for b in beat_frames_raw:
            if b < len(beats):
                # gaussiana attorno al beat per 3 frame
                for d in range(-2, 3):
                    idx = b + d
                    if 0 <= idx < len(beats):
                        beats[idx] = max(beats[idx], np.exp(-0.5 * d**2))

        # Spettrogramma per freq basse/alte
        D = np.abs(librosa.stft(y, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask  = freqs < 250
        high_mask = freqs > 4000

        low_energy  = D[low_mask,  :].mean(axis=0)
        high_energy = D[high_mask, :].mean(axis=0)
        low_energy  = low_energy  / (low_energy.max()  + 1e-8)
        high_energy = high_energy / (high_energy.max() + 1e-8)

        # Centroide spettrale
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        centroid  = centroid / (centroid.max() + 1e-8)

        def _pad(arr):
            """Porta l'array esattamente a total_frames."""
            if len(arr) >= total_frames:
                return arr[:total_frames].astype(np.float32)
            return np.pad(arr, (0, total_frames - len(arr)), mode='edge').astype(np.float32)

        return {
            'rms':      _pad(rms),
            'beats':    _pad(beats),
            'low_freq': _pad(low_energy),
            'high_freq':_pad(high_energy),
            'spectral': _pad(centroid),
        }
    except Exception as e:
        # Fallback: tutti a 0.5
        flat = np.full(total_frames, 0.5, dtype=np.float32)
        return {'rms': flat, 'beats': np.zeros(total_frames, dtype=np.float32),
                'low_freq': flat.copy(), 'high_freq': flat.copy(), 'spectral': flat.copy()}


def apply_audio_reactive(params, effect_type, audio_env, frame_idx, ar_intensity=1.0):
    """
    Modifica params in base all'audio envelope del frame corrente.
    Ritorna una nuova tupla params con valori modulati.
    """
    if audio_env is None or not isinstance(params, tuple):
        return params

    rms   = float(audio_env['rms'][frame_idx])
    beat  = float(audio_env['beats'][frame_idx])
    low   = float(audio_env['low_freq'][frame_idx])
    high  = float(audio_env['high_freq'][frame_idx])
    spec  = float(audio_env['spectral'][frame_idx])

    p = list(params)
    ar = ar_intensity

    # Strategia per effetto: quale parametro è pilotato da cosa
    STRATEGIES = {
        'pixel_sort':    [('rms', 0), ('high_freq', 1)],   # intensità↑ con rms, soglia↑ con high
        'channel_shift': [('rms', 0), ('beats', 1)],
        'datamosh':      [('low_freq', 0), ('rms', 2)],     # blocchi grandi sui bassi, chaos su rms
        'byte_corrupt':  [('rms', 0), ('beats', 1)],
        'slice_shift':   [('rms', 0), ('spectral', 1)],
        'echo_smear':    [('rms', 0), ('low_freq', 2)],
        'rgb_wave':      [('spectral', 1), ('high_freq', 2)],
        'mirror_blocks': [('beats', 0), ('rms', 1)],
        'color_quantize':[('rms', 0), ('low_freq', 1)],
        'moire':         [('rms', 0), ('spectral', 1)],
        'feedback_loop': [('rms', 0), ('beats', 2)],
        'pixel_drift':   [('low_freq', 0), ('rms', 1)],
        'slit_scan':     [('rms', 0), ('spectral', 1)],
        'thermal':       [('rms', 0), ('high_freq', 1)],
        'ascii_glitch':  [('rms', 0), ('beats', 1)],
        'halftone':      [('rms', 0), ('low_freq', 1)],
        'chroma_pulse':  [('beats', 0), ('rms', 1)],
        'vhs':           [('rms', 0), ('high_freq', 2)],
        'distruttivo':   [('rms', 1), ('beats', 0)],
        'noise':         [('rms', 0), ('high_freq', 1)],
        'broken_tv':     [('rms', 0), ('beats', 2)],
    }

    sources = {'rms': rms, 'beats': beat, 'low_freq': low, 'high_freq': high, 'spectral': spec}
    strategy = STRATEGIES.get(effect_type, [('rms', 0)])

    for src_name, param_idx in strategy:
        if param_idx < len(p):
            base = p[param_idx]
            mod  = sources[src_name] * ar * base
            p[param_idx] = float(np.clip(base + mod, 0.01, 4.0))

    # Beat flash: spike su parametro 0 sui beat
    if beat > 0.5 and len(p) > 0:
        p[0] = float(np.clip(p[0] * (1.0 + beat * ar), 0.01, 4.0))

    return tuple(p)

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
        import librosa  # lazy import — evita segfault numba
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
    """Corruzione dati reale: manomette blocchi di byte grezzi como se fossero dati compressi."""
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

def glitch_echo_smear(frame, prev_frame, intensity=1.0, decay=0.5, smear=1.0):
    """Frame echo con decay: mischia frame corrente e precedente creando ghosting/smearing."""
    try:
        if prev_frame is None:
            return frame
        alpha = np.clip(0.3 + 0.5 * intensity, 0.1, 0.95)
        # smear: shift del prev_frame prima di mixare
        shift = int(smear * 15)
        pf = np.roll(prev_frame, shift, axis=1) if shift else prev_frame
        # decay sul prev
        pf = (pf.astype(np.float32) * (1.0 - decay * 0.3)).astype(np.uint8)
        blended = cv2.addWeighted(frame.astype(np.float32), 1.0 - alpha, pf.astype(np.float32), alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_rgb_wave(frame, intensity=1.0, freq=1.0, phase_chaos=0.5):
    """Onde sinusoidali indipendenti per canale R, G, B — effetto psychedelic warp."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        xs = np.arange(w, dtype=np.float32)
        amplitude = int(max(1, 8 * intensity))
        for ch in range(3):
            phase = ch * np.pi * 2 / 3 + random.uniform(0, phase_chaos * np.pi)
            f = freq * (0.5 + ch * 0.3)
            offsets = (amplitude * np.sin(2 * np.pi * f * xs / w + phase)).astype(np.int32)
            for y in range(h):
                arr[y, :, ch] = np.roll(arr[y, :, ch], int(offsets[y % len(offsets)]))
        return arr
    except Exception:
        return frame

def glitch_mirror_blocks(frame, intensity=1.0, block_size=1.0, flip_prob=0.5):
    """Specchia blocchi casuali orizzontalmente o verticalmente — glitch geometrico."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        bw = max(16, int(w * 0.1 * block_size))
        bh = max(16, int(h * 0.1 * block_size))
        n = max(1, int(4 + 12 * intensity))
        for _ in range(n):
            x = random.randint(0, max(0, w - bw - 1))
            y = random.randint(0, max(0, h - bh - 1))
            block = arr[y:y+bh, x:x+bw].copy()
            if random.random() < flip_prob:
                block = block[:, ::-1]  # flip orizzontale
            else:
                block = block[::-1, :]  # flip verticale
            arr[y:y+bh, x:x+bw] = block
        return arr
    except Exception:
        return frame

def glitch_color_quantize(frame, intensity=1.0, levels=1.0, dither=0.5):
    """Quantizzazione colore estrema + dithering — palette ridotta, posterizzazione glitch."""
    try:
        arr = frame.copy().astype(np.float32)
        # livelli di quantizzazione: da 64 (basso intensity) a 2
        n_levels = max(2, int(64 - 60 * np.clip(intensity * levels, 0, 1)))
        step = 256.0 / n_levels
        quantized = (np.floor(arr / step) * step).astype(np.float32)
        if dither > 0:
            # Floyd-Steinberg semplificato vettorizzato: aggiungi rumore prima di quantizzare
            noise = np.random.uniform(-step * dither * 0.5, step * dither * 0.5, arr.shape).astype(np.float32)
            quantized = np.clip(np.floor((arr + noise) / step) * step, 0, 255)
        return quantized.astype(np.uint8)
    except Exception:
        return frame

def glitch_moire(frame, intensity=1.0, freq=1.0, angle=0.5):
    """Moiré pattern: sovrappone griglie sinusoidali sfasate per canale — effetto interferenza ottica."""
    try:
        arr = frame.copy().astype(np.float32)
        h, w = arr.shape[:2]
        xs = np.linspace(0, freq * 2 * np.pi * w / 100, w, dtype=np.float32)
        ys = np.linspace(0, freq * 2 * np.pi * h / 100, h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        # griglia rotata per canale
        theta = angle * np.pi / 4
        grid1 = np.sin(xx * np.cos(theta) + yy * np.sin(theta))
        grid2 = np.sin(xx * np.cos(theta + 0.3) + yy * np.sin(theta + 0.3))
        grid3 = np.sin(xx * np.cos(theta + 0.6) + yy * np.sin(theta + 0.6))
        amp = intensity * 80
        arr[:,:,0] = np.clip(arr[:,:,0] + grid1 * amp, 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] + grid2 * amp, 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] + grid3 * amp, 0, 255)
        return arr.astype(np.uint8)
    except Exception:
        return frame

def glitch_feedback_loop(frame, prev_frame, intensity=1.0, zoom=0.5, rotate=0.5):
    """Video feedback loop: zoom + leggera rotazione del frame precedente sovrapposta al corrente."""
    try:
        if prev_frame is None:
            return frame
        h, w = frame.shape[:2]
        scale = 1.0 + 0.05 * zoom * intensity
        angle_deg = (rotate - 0.5) * 4 * intensity
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
        warped = cv2.warpAffine(prev_frame, M, (w, h), borderMode=cv2.BORDER_WRAP)
        alpha = np.clip(0.25 + 0.45 * intensity, 0.1, 0.85)
        blended = cv2.addWeighted(frame.astype(np.float32), 1.0 - alpha, warped.astype(np.float32), alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_pixel_drift(frame, intensity=1.0, drift_len=1.0, vertical=0.5):
    """Trascina pixel consecutivi lungo righe o colonne creando sbavature digitali lineari."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        is_vertical = vertical > 0.5
        n_lines = max(1, int((h if not is_vertical else w) * 0.1 * intensity))
        length = max(4, int(20 + 80 * drift_len))

        if not is_vertical:
            rows = np.random.choice(h, n_lines, replace=False)
            for y in rows:
                if w - length <= 1: continue
                x = random.randint(0, w - length - 1)
                arr[y, x:x+length] = arr[y, x]
        else:
            cols = np.random.choice(w, n_lines, replace=False)
            for x in cols:
                if h - length <= 1: continue
                y = random.randint(0, h - length - 1)
                arr[y:y+length, x] = arr[y, x]
        return arr
    except Exception:
        return frame

def glitch_slit_scan(frame, intensity=1.0, time_offset=1.0, mode=0.5):
    """Simula lo slit-scan bloccando e shiftando sezioni orizzontali o verticali del fotogramma."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        h_slice = max(4, int(h * 0.05 * intensity))
        n_slices = max(1, int(3 + 10 * time_offset))
        
        for _ in range(n_slices):
            y = random.randint(0, max(0, h - h_slice - 1))
            shift = int(w * 0.08 * intensity)
            if mode > 0.5:
                arr[y:y+h_slice] = np.roll(arr[y:y+h_slice], random.randint(-shift, shift), axis=1)
            else:
                arr[y:y+h_slice] = np.roll(arr[y:y+h_slice], random.randint(-shift//2, shift//2), axis=0)
        return arr
    except Exception:
        return frame

def glitch_thermal(frame, intensity=1.0, colormap_idx=0.5, contrast=1.0):
    """Converte selettivamente aree del frame in una mappa termica basata sulla luminanza."""
    try:
        arr = frame.copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if contrast != 1.0:
            gray = np.clip(gray.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
        
        cm_type = cv2.COLORMAP_JET if colormap_idx < 0.5 else cv2.COLORMAP_RAINBOW
        thermal = cv2.applyColorMap(gray, cm_type)
        
        alpha = np.clip(intensity * 0.6, 0.1, 0.9)
        blended = cv2.addWeighted(arr, 1.0 - alpha, thermal, alpha, 0)
        return blended
    except Exception:
        return frame

def glitch_ascii_glitch(frame, intensity=1.0, cell_size=1.0, glyph_chaos=0.5):
    """Scompone il frame in blocchi di luminanza emulando un terminale a matrice di testo corrotta."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        csize = max(4, int(4 + 12 * cell_size))
        
        for y in range(0, h, csize):
            for x in range(0, w, csize):
                if random.random() < intensity * 0.4:
                    y1 = min(y + csize, h)
                    x1 = min(x + csize, w)
                    block = arr[y:y1, x:x1]
                    mean_color = cv2.mean(block)[:3]
                    if glyph_chaos > 0.5:
                        arr[y:y1, x:x1] = (np.array(mean_color) * random.uniform(0.6, 1.4)).astype(np.uint8)
                    else:
                        arr[y:y1, x:x1] = np.array(mean_color, dtype=np.uint8)
        return arr
    except Exception:
        return frame

def glitch_halftone(frame, intensity=1.0, dot_max=1.0, pattern_type=0.5):
    """Applica una retinatura a reticolo distorta che si allarga o stringe sui canali BGR."""
    try:
        arr = frame.copy().astype(np.float32)
        h, w = arr.shape[:2]
        max_r = max(2, int(3 + 12 * dot_max))
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        mask = (grid_x % max_r == 0) & (grid_y % max_r == 0)
        
        dots = np.zeros((h, w), dtype=np.float32)
        dots[mask] = 1.0
        dots = cv2.GaussianBlur(dots, (0, 0), sigmaX=max_r * 0.4)
        dots = dots / (dots.max() + 1e-8)
        
        alpha = np.clip(intensity * 0.7, 0.0, 1.0)
        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * ((1.0 - alpha) + alpha * dots)
        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_chroma_pulse(frame, intensity=1.0, sat_mult=1.0, hue_shift=0.5):
    """Altera e satura asincronamente i canali HSV creando flash cromatici distruttivi."""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + (hue_shift * 180 * intensity)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + intensity * sat_mult), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    except Exception:
        return frame

# --- ORCHESTRATORE DEGLI EFFETTI VIDEO ---
def apply_video_effect(frame, prev_frame, effect_type, params, frame_idx):
    """Indirizza il frame all'algoritmo video corretto passando i parametri modulati."""
    if effect_type == 'pixel_sort':
        return glitch_pixel_sort(frame, intensity=params[0], threshold=params[1], direction=params[2])
    elif effect_type == 'channel_shift':
        return glitch_channel_shift(frame, intensity=params[0], spread=params[1], mode=params[2])
    elif effect_type == 'datamosh':
        return glitch_datamosh(frame, prev_frame, intensity=params[0], block_size=params[1], chaos=params[2])
    elif effect_type == 'byte_corrupt':
        return glitch_byte_corrupt(frame, intensity=params[0], chunk_size=params[1], randomize=params[2])
    elif effect_type == 'slice_shift':
        return glitch_slice_shift(frame, intensity=params[0], num_slices=params[1], drift=params[2])
    elif effect_type == 'vhs':
        return glitch_vhs_frame(frame, intensity=params[0], scanline_freq=params[1], color_shift=params[2])
    elif effect_type == 'broken_tv':
        return glitch_broken_tv_frame(frame, shift_intensity=params[0], line_height=params[1], flicker_prob=params[2])
    elif effect_type == 'noise':
        return glitch_noise_frame(frame, noise_intensity=params[0], coverage=params[1], chaos=params[2])
    elif effect_type == 'distruttivo':
        return glitch_distruttivo_frame(frame, block_size=params[0], num_blocks=params[1], displacement=params[2])
    elif effect_type == 'echo_smear':
        return glitch_echo_smear(frame, prev_frame, intensity=params[0], decay=params[1], smear=params[2])
    elif effect_type == 'rgb_wave':
        return glitch_rgb_wave(frame, intensity=params[0], freq=params[1], phase_chaos=params[2])
    elif effect_type == 'mirror_blocks':
        return glitch_mirror_blocks(frame, intensity=params[0], block_size=params[1], flip_prob=params[2])
    elif effect_type == 'color_quantize':
        return glitch_color_quantize(frame, intensity=params[0], levels=params[1], dither=params[2])
    elif effect_type == 'moire':
        return glitch_moire(frame, intensity=params[0], freq=params[1], angle=params[2])
    elif effect_type == 'feedback_loop':
        return glitch_feedback_loop(frame, prev_frame, intensity=params[0], zoom=params[1], rotate=params[2])
    elif effect_type == 'pixel_drift':
        return glitch_pixel_drift(frame, intensity=params[0], drift_len=params[1], vertical=params[2])
    elif effect_type == 'slit_scan':
        return glitch_slit_scan(frame, intensity=params[0], time_offset=params[1], mode=params[2])
    elif effect_type == 'thermal':
        return glitch_thermal(frame, intensity=params[0], colormap_idx=params[1], contrast=params[2])
    elif effect_type == 'ascii_glitch':
        return glitch_ascii_glitch(frame, intensity=params[0], cell_size=params[1], glyph_chaos=params[2])
    elif effect_type == 'halftone':
        return glitch_halftone(frame, intensity=params[0], dot_max=params[1], pattern_type=params[2])
    elif effect_type == 'chroma_pulse':
        return glitch_chroma_pulse(frame, intensity=params[0], sat_mult=params[1], hue_shift=params[2])
    return frame

# ─────────────────────────────────────────────
# INITIALIZE INTERFACE & UI SELECTION
# ─────────────────────────────────────────────

# Inizializzazione Session State per prevenire ricaricamenti molesti di Streamlit
if 'video_ready' not in st.session_state: st.session_state.video_ready = False
if 'h264_path' not in st.session_state: st.session_state.h264_path = None
if 'glitched_audio_path' not in st.session_state: st.session_state.glitched_audio_path = None
if 'output_video_name' not in st.session_state: st.session_state.output_video_name = ""
if 'report_data' not in st.session_state: st.session_state.report_data = ""
if 'report_filename' not in st.session_state: st.session_state.report_filename = ""

# Selettore Algoritmo Principale
effect_choice = st.selectbox("🎛️ Seleziona l'Algoritmo di Glitch:", [
    'vhs', 'distruttivo', 'noise', 'broken_tv', 'pixel_sort', 'channel_shift', 
    'datamosh', 'byte_corrupt', 'slice_shift', 'echo_smear', 'rgb_wave', 
    'mirror_blocks', 'color_quantize', 'moire', 'feedback_loop', 'pixel_drift', 
    'slit_scan', 'thermal', 'ascii_glitch', 'halftone', 'chroma_pulse', 'combined', 'random'
])

# Controlli Reattività Audio
st.markdown("### 🎙️ Impostazioni Audio-Reactive")
audio_reactive = st.checkbox("Abilita Modulazione Audio-Reactive sui parametri video", value=True)
ar_intensity = st.slider("Intensità Guadagno Audio (AR)", 0.1, 3.0, 1.0) if audio_reactive else 1.0

# Gestione Dinamica dei Pannelli dei Parametri Base
st.markdown("### 🎚️ Regolazioni Parametri Base Manuali")
p1, p2, p3 = 1.0, 1.0, 1.0
combined_audio_params = {}

if effect_choice == 'combined':
    st.info("Configura quali moduli audio iniettare nella catena sommatrice:")
    combined_audio_params["apply_vhs"] = st.checkbox("Audio VHS", value=True)
    if combined_audio_params["apply_vhs"]:
        combined_audio_params["vhs_intensity"] = st.slider("VHS Intensity", 0.0, 3.0, 1.0)
        combined_audio_params["vhs_wow_flutter"] = st.slider("Wow & Flutter", 0.0, 3.0, 1.0)
        combined_audio_params["vhs_tape_hiss"] = st.slider("Tape Hiss", 0.0, 3.0, 0.5)
        
    combined_audio_params["apply_distruttivo"] = st.checkbox("Audio Distruttivo", value=False)
    if combined_audio_params["apply_distruttivo"]:
        combined_audio_params["dest_chaos_level"] = st.slider("Chaos Level", 0.0, 3.0, 1.0)
        combined_audio_params["dest_skip_prob"] = st.slider("Skip Prob", 0.0, 1.0, 0.3)
        combined_audio_params["dest_reverse_prob"] = st.slider("Reverse Prob", 0.0, 1.0, 0.2)
        
    combined_audio_params["apply_noise"] = st.checkbox("Audio Noise", value=False)
    if combined_audio_params["apply_noise"]:
        combined_audio_params["noise_intensity"] = st.slider("Noise Intensity", 0.0, 3.0, 1.0)
        combined_audio_params["noise_digital_artifacts"] = st.slider("Digital Artifacts", 0.0, 3.0, 0.5)
        combined_audio_params["noise_bit_crush"] = st.slider("Bit Crush", 0.0, 1.0, 0.2)
        
    combined_audio_params["apply_broken_tv"] = st.checkbox("Audio Broken TV", value=False)
    if combined_audio_params["apply_broken_tv"]:
        combined_audio_params["tv_static_intensity"] = st.slider("Static Intensity", 0.0, 3.0, 1.0)
        combined_audio_params["tv_channel_separation"] = st.slider("Channel Separation", 0.0, 3.0, 1.0)
        combined_audio_params["tv_frequency_drift"] = st.slider("Frequency Drift", 0.0, 3.0, 0.5)
else:
    p1 = st.slider("Parametro 1 (Intensità / Alfa)", 0.0, 3.0, 1.0)
    p2 = st.slider("Parametro 2 (Frequenza / Scala)", 0.0, 3.0, 1.0)
    p3 = st.slider("Parametro 3 (Caos / Smear / Direzione)", 0.0, 3.0, 1.0)

# ─────────────────────────────────────────────
# CORE PIPELINE RUNNER
# ─────────────────────────────────────────────

if uploaded_file is not None:
    if st.button("🚀 GENERA GLITCH VIDEO & AUDIO", use_container_width=True):
        t_start = time.time()
        
        # Salvataggio video originale in tempfile
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t_in.write(uploaded_file.read())
        t_in.close()
        
        # Estrazione o conversione traccia audio sorgente
        with st.spinner("Ancoraggio ed estrazione traccia audio nativa/esterna..."):
            if uploaded_audio_file is not None:
                t_aud_ext = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio_file.name)[1])
                t_aud_ext.write(uploaded_audio_file.read())
                t_aud_ext.close()
                audio_wav = convert_audio_to_wav(t_aud_ext.name)
            else:
                audio_wav = extract_audio(t_in.name)

        # Inizializzazione proprietà video OpenCV
        cap = cv2.VideoCapture(t_in.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Estrazione envelope spettrali per-frame
        audio_env = None
        if audio_reactive and audio_wav and os.path.exists(audio_wav):
            with st.spinner("Analisi spettrale delle frequenze audio in corso..."):
                audio_env = analyze_audio_for_video(audio_wav, fps, total_frames)
                
        # Processing ed elaborazione distruttiva della traccia audio
        glitched_audio_path = None
        if audio_wav and os.path.exists(audio_wav):
            with st.spinner("Applicazione filtri distruttivi all'audio..."):
                a_params = combined_audio_params if effect_choice == 'combined' else (p1, p2, p3)
                glitched_audio_path = process_audio_glitch(audio_wav, effect_choice, a_params)

        # Configurazione video muto intermediario
        t_out_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t_out_raw.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(t_out_raw.name, fourcc, fps, (width, height))
        
        # Loop sequenziale frame per frame
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        prev_frame = None
        frame_idx = 0
        
        # Lista interna di backup per l'effetto random
        all_effects = ['vhs', 'distruttivo', 'noise', 'broken_tv', 'pixel_sort', 'channel_shift', 
                       'datamosh', 'byte_corrupt', 'slice_shift', 'echo_smear', 'rgb_wave', 
                       'mirror_blocks', 'color_quantize', 'moire', 'feedback_loop', 'pixel_drift', 
                       'slit_scan', 'thermal', 'ascii_glitch', 'halftone', 'chroma_pulse']
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Scelta dell'effetto (gestione random ed effetti singoli)
            current_effect = effect_choice
            if effect_choice == 'random':
                current_effect = all_effects[(frame_idx // 12) % len(all_effects)] # cambia effetto ogni 12 frame
            elif effect_choice == 'combined':
                current_effect = 'vhs' # fallback grafico se combinato
                
            # Modulazione reattiva dei parametri via audio envelope
            base_params = (p1, p2, p3)
            if audio_reactive and audio_env:
                active_params = apply_audio_reactive(base_params, current_effect, audio_env, frame_idx, ar_intensity)
            else:
                active_params = base_params
                
            # Applicazione matematica dei glitch grafici
            glitched_frame = apply_video_effect(frame, prev_frame, current_effect, active_params, frame_idx)
            
            out_writer.write(glitched_frame)
            prev_frame = frame.copy()
            frame_idx += 1
            
            if frame_idx % max(1, total_frames // 20) == 0:
                pct = min(1.0, frame_idx / total_frames)
                progress_bar.progress(int(pct * 100))
                status_text.text(f"🎬 Elaborazione frame: {frame_idx}/{total_frames} ({int(pct*100)}%)")
                
        cap.release()
        out_writer.release()
        progress_bar.progress(100)
        
        # Muxing ed unione finale flussi H.264 tramite FFmpeg web-friendly
        status_text.text("📦 Muxing finale traccia video e traccia audio corrotte...")
        t_final = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t_final.close()
        
        if glitched_audio_path and os.path.exists(glitched_audio_path):
            cmd = [
                'ffmpeg', '-i', t_out_raw.name, '-i', glitched_audio_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
                '-c:a', 'aac', t_final.name, '-y'
            ]
        else:
            cmd = [
                'ffmpeg', '-i', t_out_raw.name,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
                '-an', t_final.name, '-y'
            ]
            
        subprocess.run(cmd, capture_output=True)
        elapsed_time = time.time() - t_start
        orig_name = os.path.splitext(uploaded_file.name)[0]
        
        # Salvataggio nel Session State per persistenza dei dati sul browser
        st.session_state.h264_path = t_final.name
        st.session_state.glitched_audio_path = glitched_audio_path
        st.session_state.glitched_audio_name = uploaded_audio_file.name if uploaded_audio_file else uploaded_file.name
        st.session_state.output_video_name = f"glitch_{orig_name}.mp4"
        st.session_state.video_ready = True
        
        # Scrittura report di sistema
        report_lines = [
            "📟 VIDEODISTRUKTOR SYSTEM REPORT",
            "===============================",
            f"File Video: {uploaded_file.name}",
            f"Algoritmo: {effect_choice}",
            f"Risoluzione: {width}x{height}",
            f"Frame totali: {total_frames} @ {fps:.2f} FPS",
            f"Audio-Reactive: {audio_reactive} (Gain: {ar_intensity})",
            f"Parametri Base: P1={p1:.2f}, P2={p2:.2f}, P3={p3:.2f}",
            f"Tempo calcolo: {elapsed_time:.2f} secondi",
            "Stato: COMPLETED SUCCESS"
        ]
        st.session_state.report_data = "\n".join(report_lines)
        st.session_state.report_filename = f"report_glitch_{orig_name}.txt"
        st.success(f"✨ Rendering completato in {elapsed_time:.2f} secondi!")

# ─────────────────────────────────────────────
# DISPLAY OUTPUT & DOWNLOADS PERSISTENTI
# ─────────────────────────────────────────────

if st.session_state.video_ready and st.session_state.h264_path:
    st.markdown("---")
    st.subheader("📺 Anteprima Video Generato")
    st.video(st.session_state.h264_path)
    
    if st.session_state.glitched_audio_path and os.path.exists(st.session_state.glitched_audio_path):
        st.audio(st.session_state.glitched_audio_path)
        with open(st.session_state.glitched_audio_path, 'rb') as af:
            orig_stem = os.path.splitext(st.session_state.get('glitched_audio_name', 'audio'))[0]
            st.download_button("📥 Scarica Audio Glitch (mp3)", af,
                file_name=f"glitch_{orig_stem}.mp3", mime="audio/mpeg", key="down_audio")

if st.session_state.video_ready:
    st.markdown("---")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        if st.session_state.h264_path and os.path.exists(st.session_state.h264_path):
            with open(st.session_state.h264_path, 'rb') as vf:
                st.download_button(
                    label="📥 Scarica video (H.264)",
                    data=vf,
                    file_name=st.session_state.output_video_name,
                    mime="video/mp4",
                    key="down_v"
                )
    with c_d2:
        st.download_button(
            label="📄 Scarica Report",
            data=st.session_state.report_data,
            file_name=st.session_state.report_filename,
            key="down_rep"
        )
