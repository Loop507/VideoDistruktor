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

        # Beat detection — compatibile librosa >= 0.10 (ritorna BeatTrackResult, non tupla)
        _beat_result = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beat_frames_raw = _beat_result[1] if isinstance(_beat_result, tuple) else _beat_result.beat_frames
        # beat_frames_exact: indici frame video esatti (non hop-frame, ma video-frame)
        beat_frames_exact = np.unique(np.clip(
            np.round(beat_frames_raw.astype(np.float32) * hop / (sr / fps)).astype(np.int32),
            0, total_frames - 1
        ))
        beats = np.zeros(len(rms), dtype=np.float32)
        for b in beat_frames_raw:
            if b < len(beats):
                # gaussiana attorno al beat per 3 frame (usata solo in modalità non-sync)
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
            'rms':               _pad(rms),
            'beats':             _pad(beats),
            'low_freq':          _pad(low_energy),
            'high_freq':         _pad(high_energy),
            'spectral':          _pad(centroid),
            'beat_frames_exact': beat_frames_exact,  # array di frame-index video esatti
        }
    except Exception as e:
        # Fallback silenzioso: tutti a 0.5 — l'errore non è critico
        flat = np.full(total_frames, 0.5, dtype=np.float32)
        return {'rms': flat, 'beats': np.zeros(total_frames, dtype=np.float32),
                'low_freq': flat.copy(), 'high_freq': flat.copy(), 'spectral': flat.copy(),
                'beat_frames_exact': np.array([], dtype=np.int32)}


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

def build_beat_envelope(beat_frames_exact, total_frames, decay_frames=6, peak=1.0):
    """
    Costruisce un envelope float32 (total_frames,) con:
    - picco = peak sui beat esatti
    - decadimento esponenziale: val = peak * exp(-distanza / decay_frames)
    - 0 lontano dai beat
    decay_frames: numero di frame per scendere a ~37% del picco (costante di tempo)
    """
    env = np.zeros(total_frames, dtype=np.float32)
    if len(beat_frames_exact) == 0:
        return env
    for bf in beat_frames_exact:
        bf = int(bf)
        # decadimento verso destra (post-beat)
        for d in range(min(decay_frames * 4, total_frames)):
            idx = bf + d
            if idx >= total_frames:
                break
            val = peak * np.exp(-d / max(1, decay_frames))
            if val > env[idx]:
                env[idx] = val
    return env


# Configurazione della pagina
st.set_page_config(page_title="VideoDistruktor by loop507", layout="wide")

st.markdown("<h1>🎬🔥 VideoDistruktor <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica un video e distruggi: VHS, Distruttivo, Noise, Combinato, Broken TV e molto altro. **Audio reactive, keyframe, glitch audio su 4 modalità.**")

# File uploader per video — unico punto di ingresso
uploaded_file = st.file_uploader("📁 Carica un video", type=["mp4", "avi", "mov", "mkv"])

# Controlla se ffmpeg è disponibile (cached per evitare subprocess ad ogni re-run)
@st.cache_data
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def frame_to_pil(frame):
    """Converte frame OpenCV (BGR) in PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_frame(pil_img):
    """Converte PIL Image (RGB) in frame OpenCV (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Funzioni degli effetti audio ---
def extract_audio(video_path, silent=False):
    """Estrae l'audio dal video usando ffmpeg e lo converte in WAV 44100Hz stereo"""
    fd, audio_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path, '-y']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if not silent:
                st.warning("⚠️ Impossibile estrarre l'audio. Il video potrebbe non avere traccia audio.")
            try: os.unlink(audio_path)
            except: pass
            return None
        return audio_path
    except Exception as e:
        if not silent:
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
        thr = np.clip(threshold, 0.01, 0.99)
        num_rows = max(1, int(h * np.clip(intensity, 0.05, 3.0) / 3.0))
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
        blended = cv2.addWeighted(frame.astype(np.float32), 1.0 - alpha,
                                   pf.astype(np.float32), alpha, 0)
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
                block = block[:, ::-1]   # flip orizzontale
            else:
                block = block[::-1, :]   # flip verticale
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
    """Video feedback loop: zoom + leggera rotazione del frame precedente sovrapposta al corrente — simula il loop di monitor su monitor."""
    try:
        if prev_frame is None:
            return frame
        h, w = frame.shape[:2]
        scale = 1.0 + 0.05 * zoom * intensity
        angle_deg = (rotate - 0.5) * 4 * intensity
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
        warped = cv2.warpAffine(prev_frame, M, (w, h), borderMode=cv2.BORDER_WRAP)
        alpha = np.clip(0.25 + 0.45 * intensity, 0.1, 0.85)
        blended = cv2.addWeighted(frame.astype(np.float32), 1.0 - alpha,
                                   warped.astype(np.float32), alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return frame

def glitch_pixel_drift(frame, intensity=1.0, drift_speed=0.5, turbulence=0.5):
    """Pixel drift: displacement map rumoroso — liquid glitch. Vettorizzato, safe remap."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        mag = max(1, int(20 * intensity))
        noise_x = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        noise_y = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        ksize = max(3, int(51 - 40 * float(np.clip(turbulence, 0, 0.99))))
        if ksize % 2 == 0:
            ksize += 1
        smooth_x = cv2.GaussianBlur(noise_x, (ksize, ksize), 0) * float(mag)
        smooth_y = cv2.GaussianBlur(noise_y, (ksize, ksize), 0) * float(mag) * float(drift_speed)
        base_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        base_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
        map_x = np.clip(base_x + smooth_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(base_y + smooth_y, 0, h - 1).astype(np.float32)
        return cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    except Exception:
        return frame


def glitch_slit_scan(frame, slit_buffer, intensity=1.0, speed=0.5, tilt=0.5):
    """Slit Scan: ogni colonna presa da un momento temporale diverso del buffer."""
    try:
        h, w = frame.shape[:2]
        buf_len = len(slit_buffer)
        if buf_len < 2:
            return frame
        out = frame.copy()
        for x in range(w):
            t_offset = int(
                (float(x) / w) * buf_len * float(speed) * float(intensity) +
                np.sin(float(x) / w * np.pi * float(tilt) * 4) * buf_len * 0.1
            )
            src_idx = int(buf_len - 1 - (abs(t_offset) % buf_len))
            src_idx = max(0, min(buf_len - 1, src_idx))
            src = slit_buffer[src_idx]
            sh, sw = src.shape[:2]
            sx = min(x, sw - 1)
            col_h = min(h, sh)
            out[:col_h, x] = src[:col_h, sx]
        return out
    except Exception:
        return frame

def glitch_thermal(frame, intensity=1.0, noise_level=0.5, aberration=0.5):
    """Thermal Vision Glitch: falsi colori termici (COLORMAP_JET) + aberrazione + noise sorveglianza."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # boost contrasto termico
        gray = np.clip(gray * (1.0 + 0.5 * intensity), 0, 255).astype(np.uint8)
        # noise tipo sensore termico
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 15, gray.shape).astype(np.float32)
            gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        # aberrazione cromatica sulle tre bande
        if aberration > 0:
            sh = int(aberration * 8 * intensity)
            b, g, r = cv2.split(thermal)
            r = np.roll(r,  sh, axis=1)
            b = np.roll(b, -sh, axis=1)
            thermal = cv2.merge([b, g, r])
        # blend con originale in base a intensity
        alpha = np.clip(0.4 + 0.5 * intensity, 0.4, 1.0)
        return cv2.addWeighted(thermal.astype(np.float32), alpha,
                               frame.astype(np.float32), 1.0 - alpha, 0).astype(np.uint8)
    except Exception:
        return frame

def glitch_ascii_glitch(frame, intensity=1.0, block_size=1.0, chaos=0.5):
    """ASCII Glitch: divide il frame in blocchi, sostituisce con la luminosità media come valore piatto,
    poi corrompe blocchi casuali — estetica low-res distruttiva."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        bsize = max(4, int(8 + 24 * (1.0 - min(block_size, 0.99))))
        for y in range(0, h, bsize):
            for x in range(0, w, bsize):
                block = arr[y:y+bsize, x:x+bsize]
                if block.size == 0:
                    continue
                mean_val = block.mean(axis=(0, 1)).astype(np.uint8)
                # quantizza il blocco alla media (ASCII-like flat)
                arr[y:y+bsize, x:x+bsize] = mean_val
                # chaos: corrompi blocchi casuali
                if random.random() < chaos * intensity * 0.3:
                    ch = random.randint(0, 2)
                    arr[y:y+bsize, x:x+bsize, ch] = random.randint(0, 255)
        # scanline nere (ogni N righe) per effetto CRT a bassa risoluzione
        step = max(2, bsize)
        if intensity > 0.5:
            arr[::step] = (arr[::step].astype(np.float32) * (1.0 - 0.4 * intensity)).astype(np.uint8)
        return arr
    except Exception:
        return frame

def glitch_halftone(frame, intensity=1.0, dot_size=0.5, angle=0.3):
    """Halftone Destroy: retino tipografico per canale con angoli sfasati — vettorizzato."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        dsize = max(4, int(4 + 20 * dot_size))
        out = np.zeros((h, w, 3), dtype=np.uint8)

        for ch in range(3):
            ch_img = arr[:, :, ch].astype(np.float32) / 255.0
            ch_out = np.zeros((h, w), dtype=np.float32)
            # griglia di centri blocchi
            ys = np.arange(dsize // 2, h, dsize)
            xs = np.arange(dsize // 2, w, dsize)
            for cy in ys:
                for cx in xs:
                    lum = float(ch_img[min(cy, h-1), min(cx, w-1)])
                    radius = int(lum * dsize / 2 * (1.0 + intensity * 0.5))
                    if radius > 0:
                        cv2.circle(ch_out, (int(cx), int(cy)), min(radius, dsize), 1.0, -1)
            out[:, :, ch] = np.clip(ch_out * 255, 0, 255).astype(np.uint8)

        alpha = np.clip(0.5 + 0.4 * intensity, 0.5, 1.0)
        return cv2.addWeighted(out, alpha, arr, 1.0 - alpha, 0)
    except Exception:
        return frame

def glitch_chroma_pulse(frame, intensity=1.0, radial=0.5, pulse_speed=0.5, _frame_idx=0):
    """Chromatic Aberration Pulse: aberrazione cromatica radiale pulsante."""
    try:
        arr = frame.copy()
        h, w = arr.shape[:2]
        cx, cy = float(w) / 2.0, float(h) / 2.0
        phase = float(_frame_idx) * float(pulse_speed) * 0.1
        amp_r = max(0, int(intensity * 12 * (1.0 + 0.5 * np.sin(phase))))
        amp_b = max(0, int(intensity * 12 * (1.0 + 0.5 * np.cos(phase + 1.0))))

        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)  # shape (h,w)
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx * dx + dy * dy) + 1e-8
        base_x = xx  # float32 (h,w)
        base_y = yy

        def rshift(ch_img, amp):
            if amp == 0:
                return ch_img
            sh_x = (dx / dist * amp * float(radial)).astype(np.float32)
            sh_y = (dy / dist * amp * float(radial) * 0.5).astype(np.float32)
            mx = np.clip(base_x + sh_x, 0, w - 1).astype(np.float32)
            my = np.clip(base_y + sh_y, 0, h - 1).astype(np.float32)
            return cv2.remap(ch_img, mx, my, cv2.INTER_LINEAR)

        b, g, r = cv2.split(arr)
        r = rshift(r, amp_r)
        b = rshift(b, amp_b)
        return cv2.merge([b, g, r])
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

def build_mask(h, w, mask_type, mask_x, mask_y, mask_w, mask_h, feather=0, reverse=False):
    """
    Genera una maschera float32 (0.0–1.0) dove 1.0 = effetto attivo.
    mask_type : 'striscia_h' | 'striscia_v' | 'cerchio' | 'rettangolo' | 'nessuna'
    mask_x/y  : posizione centro (0.0–1.0 relativa al frame)
    mask_w/h  : dimensione (0.0–1.0 relativa al frame)
    feather   : pixel di sfumatura ai bordi
    reverse   : se True, inverte la maschera (effetto fuori, originale dentro)
    """
    mask = np.zeros((h, w), dtype=np.float32)

    cx = int(np.clip(mask_x, 0.0, 1.0) * w)
    cy = int(np.clip(mask_y, 0.0, 1.0) * h)
    mw = max(2, int(np.clip(mask_w, 0.01, 1.0) * w))
    mh = max(2, int(np.clip(mask_h, 0.01, 1.0) * h))

    if mask_type == 'striscia_h':
        y0 = max(0, cy - mh // 2)
        y1 = min(h, cy + mh // 2)
        mask[y0:y1, :] = 1.0

    elif mask_type == 'striscia_v':
        x0 = max(0, cx - mw // 2)
        x1 = min(w, cx + mw // 2)
        mask[:, x0:x1] = 1.0

    elif mask_type == 'rettangolo':
        x0 = max(0, cx - mw // 2)
        x1 = min(w, cx + mw // 2)
        y0 = max(0, cy - mh // 2)
        y1 = min(h, cy + mh // 2)
        mask[y0:y1, x0:x1] = 1.0

    elif mask_type == 'cerchio':
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        # ellisse: normalizza per avere cerchio vero in pixel space
        rx = max(1, mw / 2)
        ry = max(1, mh / 2)
        dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        mask[dist <= 1.0] = 1.0

    elif mask_type == 'nessuna':
        mask[:] = 1.0
        if reverse:
            return np.zeros((h, w), dtype=np.float32)
        return mask

    # Feathering con GaussianBlur
    if feather > 1:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather * 0.5)

    if reverse:
        mask = 1.0 - mask

    return mask


def build_combined_mask(h, w, masks_list):
    """
    Combina una lista di maschere con OR logico (np.maximum).
    masks_list: lista di dict con chiavi:
        mask_type, mask_x, mask_y, mask_w, mask_h, mask_feather, mask_reverse
    Ritorna maschera float32 (h, w) combinata.
    Se masks_list è vuota, ritorna maschera di zeri (nessun effetto).
    """
    if not masks_list:
        return np.zeros((h, w), dtype=np.float32)
    combined = np.zeros((h, w), dtype=np.float32)
    for m in masks_list:
        single = build_mask(
            h, w,
            m.get('mask_type', 'nessuna'),
            m.get('mask_x', 0.5),
            m.get('mask_y', 0.5),
            m.get('mask_w', 1.0),
            m.get('mask_h', 0.3),
            m.get('mask_feather', 0),
            m.get('mask_reverse', False),
        )
        combined = np.maximum(combined, single)
    return np.clip(combined, 0.0, 1.0)


def interpolate_mask_at_frame(msk, frame_idx, total_frames):
    """
    Ritorna una copia del dict maschera con i valori geometrici interpolati
    al frame corrente, se 'animate_pos' è True.
    Interpola: mask_y, mask_h (striscia_h), mask_x, mask_w (striscia_v),
               mask_x, mask_y, mask_w, mask_h (rettangolo/cerchio).
    """
    if not msk.get('animate_pos', False):
        return msk
    t = float(frame_idx) / max(1, total_frames - 1)  # 0.0 → 1.0

    out = dict(msk)
    mt = msk.get('mask_type', 'nessuna')

    def lerp(key_s, key_e, fallback):
        s = msk.get(key_s, fallback)
        e = msk.get(key_e, fallback)
        return s + (e - s) * t

    if mt == 'striscia_h':
        out['mask_y'] = lerp('mask_y_start', 'mask_y_end', msk.get('mask_y', 0.5))
        out['mask_h'] = lerp('mask_h_start', 'mask_h_end', msk.get('mask_h', 0.25))
    elif mt == 'striscia_v':
        out['mask_x'] = lerp('mask_x_start', 'mask_x_end', msk.get('mask_x', 0.5))
        out['mask_w'] = lerp('mask_w_start', 'mask_w_end', msk.get('mask_w', 0.25))
    elif mt in ('rettangolo', 'cerchio'):
        out['mask_x'] = lerp('mask_x_start', 'mask_x_end', msk.get('mask_x', 0.5))
        out['mask_y'] = lerp('mask_y_start', 'mask_y_end', msk.get('mask_y', 0.5))
        out['mask_w'] = lerp('mask_w_start', 'mask_w_end', msk.get('mask_w', 0.5))
        out['mask_h'] = lerp('mask_h_start', 'mask_h_end', msk.get('mask_h', 0.5))
    return out


def build_combined_mask_at_frame(h, w, masks_list, frame_idx=0, total_frames=1):
    """
    Come build_combined_mask ma interpola i parametri geometrici per ogni
    maschera animata prima di costruire la maschera.
    """
    if not masks_list:
        return np.zeros((h, w), dtype=np.float32)
    combined = np.zeros((h, w), dtype=np.float32)
    for m in masks_list:
        mi = interpolate_mask_at_frame(m, frame_idx, total_frames)
        single = build_mask(
            h, w,
            mi.get('mask_type', 'nessuna'),
            mi.get('mask_x', 0.5),
            mi.get('mask_y', 0.5),
            mi.get('mask_w', 1.0),
            mi.get('mask_h', 0.3),
            mi.get('mask_feather', 0),
            mi.get('mask_reverse', False),
        )
        combined = np.maximum(combined, single)
    return np.clip(combined, 0.0, 1.0)


def apply_mask_blend(original, processed, mask):
    """
    Blend originale e processato usando la maschera.
    mask float32 (h,w): 1.0 = prendi processed, 0.0 = prendi original
    """
    m = mask[:, :, np.newaxis]  # (h, w, 1) per broadcasting su 3 canali
    blended = (processed.astype(np.float32) * m + original.astype(np.float32) * (1.0 - m))
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_video(video_path, effect_type, params, max_frames=None, audio_mode="0_originale",
                  kf_envelope=None, audio_params_override=None, aspect_ratio="Originale",
                  audio_source_path=None, audio_env=None, ar_intensity=0.0,
                  mask_type="nessuna", mask_x=0.5, mask_y=0.5, mask_w=1.0, mask_h=0.3,
                  mask_feather=0, mask_reverse=False,
                  beat_sync=False, beat_decay=6, beat_intensity=2.0,
                  masks_list=None,
                  temporal_crossfade=False, tc_source="rms",
                  tc_alpha_min=0.0, tc_alpha_max=1.0, tc_smooth=0.7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il video.")
        return None

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_total_frames = total_frames if (max_frames is None or max_frames == 0) else min(total_frames, max_frames)

    # Calcola dimensioni output esatte
    TARGET_SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}
    if aspect_ratio in TARGET_SIZES:
        out_w, out_h = TARGET_SIZES[aspect_ratio]
    else:
        out_w, out_h = width, height

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
        prev_frame  = None
        slit_buffer = []
        SLIT_BUF_LEN = 30
        progress_bar = st.progress(0)
        status_text  = st.empty()

        # Crossfade audio-reattivo: stato smoothing (attack/release) tra frame
        _tc_active = temporal_crossfade and audio_env is not None and tc_source in audio_env
        _tc_smoothed = 0.5

        # Determina se usare il sistema multi-maschera (con eventuale animazione)
        _use_multi_mask = masks_list is not None and len(masks_list) > 0
        _use_mask_legacy = not _use_multi_mask and mask_type != 'nessuna'

        if _use_multi_mask:
            _use_mask = any(m.get('mask_type', 'nessuna') != 'nessuna' for m in masks_list)
            # Determina se almeno una maschera è animata
            _has_animated_mask = any(m.get('animate_pos', False) for m in masks_list)
            # Maschera statica pre-calcolata (usata quando nessuna è animata)
            if not _has_animated_mask:
                glitch_mask = build_combined_mask(out_h, out_w, masks_list)
        else:
            _use_mask = _use_mask_legacy
            _has_animated_mask = False
            if _use_mask_legacy:
                glitch_mask = build_mask(out_h, out_w, mask_type, mask_x, mask_y,
                                         mask_w, mask_h, mask_feather, mask_reverse)

        # Beat-sync envelope (solo se attivo e audio_env disponibile)
        beat_env = None
        if beat_sync and audio_env is not None:
            bfe = audio_env.get('beat_frames_exact', np.array([], dtype=np.int32))
            beat_env = build_beat_envelope(bfe, actual_total_frames,
                                           decay_frames=beat_decay, peak=beat_intensity)

        def apply_effect(frame, prev_frame, frame_count):
            cp = params
            # keyframe
            if kf_envelope is not None and isinstance(params, tuple) and frame_count < len(kf_envelope):
                kf_val = float(np.clip(kf_envelope[frame_count], 0.0, 3.0))
                cp = (kf_val,) + params[1:]
            # beat-sync: sovrascrive il param 0 con l'envelope beat esatto
            if beat_env is not None and frame_count < len(beat_env):
                bval = float(beat_env[frame_count])
                if isinstance(cp, tuple) and len(cp) > 0:
                    cp = (float(np.clip(bval, 0.01, 4.0)),) + cp[1:]
                elif isinstance(cp, tuple):
                    cp = (float(np.clip(bval, 0.01, 4.0)),)
            # audio reactive (solo se beat_sync non attivo)
            elif audio_env is not None and ar_intensity > 0 and isinstance(cp, tuple):
                cp = apply_audio_reactive(cp, effect_type, audio_env, frame_count, ar_intensity)

            fn_map = {
                'pixel_sort':    lambda f: glitch_pixel_sort(f, *cp),
                'channel_shift': lambda f: glitch_channel_shift(f, *cp),
                'datamosh':      lambda f: glitch_datamosh(f, prev_frame, *cp),
                'byte_corrupt':  lambda f: glitch_byte_corrupt(f, *cp),
                'slice_shift':   lambda f: glitch_slice_shift(f, *cp),
                'echo_smear':    lambda f: glitch_echo_smear(f, prev_frame, *cp),
                'rgb_wave':      lambda f: glitch_rgb_wave(f, *cp),
                'mirror_blocks': lambda f: glitch_mirror_blocks(f, *cp),
                'color_quantize':lambda f: glitch_color_quantize(f, *cp),
                'moire':         lambda f: glitch_moire(f, *cp),
                'feedback_loop': lambda f: glitch_feedback_loop(f, prev_frame, *cp),
                'pixel_drift':   lambda f: glitch_pixel_drift(f, *cp),
                'slit_scan':     lambda f: glitch_slit_scan(f, slit_buffer, *cp),
                'thermal':       lambda f: glitch_thermal(f, *cp),
                'ascii_glitch':  lambda f: glitch_ascii_glitch(f, *cp),
                'halftone':      lambda f: glitch_halftone(f, *cp),
                'chroma_pulse':  lambda f: glitch_chroma_pulse(f, *cp, _frame_idx=frame_count),
                'vhs':           lambda f: glitch_vhs_frame(f, *cp),
                'distruttivo':   lambda f: glitch_distruttivo_frame(f, *cp),
                'noise':         lambda f: glitch_noise_frame(f, *cp),
                'broken_tv':     lambda f: glitch_broken_tv_frame(f, *cp),
            }
            if effect_type in fn_map:
                return fn_map[effect_type](frame)
            elif effect_type == 'combined':
                cf = frame.copy()
                if params.get("apply_vhs"):         cf = glitch_vhs_frame(cf, params.get("vhs_intensity",1.0), params.get("vhs_scanline_freq",1.0), params.get("vhs_color_shift",1.0))
                if params.get("apply_distruttivo"): cf = glitch_distruttivo_frame(cf, params.get("dest_block_size",1.0), params.get("dest_num_blocks",1.0), params.get("dest_displacement",1.0))
                if params.get("apply_noise"):       cf = glitch_noise_frame(cf, params.get("noise_intensity",1.0), params.get("noise_coverage",1.0), params.get("noise_chaos",1.0))
                if params.get("apply_broken_tv"):   cf = glitch_broken_tv_frame(cf, params.get("tv_shift_intensity",1.0), params.get("tv_line_height",1.0), params.get("tv_flicker_prob",1.0))
                if params.get("apply_pixel_sort"):  cf = glitch_pixel_sort(cf, params.get("ps_intensity",1.0), params.get("ps_threshold",0.5), params.get("ps_direction",0.3))
                if params.get("apply_channel_shift"):cf = glitch_channel_shift(cf, params.get("cs_intensity",1.0), params.get("cs_spread",1.0), params.get("cs_mode",0.3))
                if params.get("apply_slice_shift"): cf = glitch_slice_shift(cf, params.get("ss_intensity",1.0), params.get("ss_num_slices",1.0), params.get("ss_drift",1.0))
                return cf
            elif effect_type == 'random':
                rl = cp[0] if cp else 1.0
                all_fx = ['pixel_sort','channel_shift','datamosh','byte_corrupt','slice_shift',
                          'echo_smear','rgb_wave','mirror_blocks','color_quantize','moire',
                          'feedback_loop','pixel_drift','thermal','ascii_glitch','chroma_pulse',
                          'vhs','broken_tv','noise','distruttivo']
                ch = random.choice(all_fx)
                rp = tuple(random.uniform(0.5, 1.5) * rl for _ in range(3))
                if ch in fn_map:
                    return fn_map[ch](frame)
                return glitch_noise_frame(frame, *rp)
            return frame

        while cap.isOpened() and frame_count < actual_total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                processed = apply_effect(frame, prev_frame, frame_count)
            except Exception:
                processed = frame

            # aggiorna slit buffer
            slit_buffer.append(frame.copy())
            if len(slit_buffer) > SLIT_BUF_LEN:
                slit_buffer.pop(0)

            prev_frame = frame.copy()

            # Crop/resize al formato target
            if aspect_ratio in TARGET_SIZES:
                ph, pw = processed.shape[:2]
                scale = max(out_w / pw, out_h / ph)
                rw, rh = int(pw * scale), int(ph * scale)
                resized = cv2.resize(processed, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
                cx = (rw - out_w) // 2
                cy = (rh - out_h) // 2
                processed = resized[cy:cy+out_h, cx:cx+out_w]
                # Resize anche frame originale per il blend maschera
                orig_resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
                orig_cropped = orig_resized[cy:cy+out_h, cx:cx+out_w]
            else:
                orig_cropped = frame if frame.shape[:2] == (out_h, out_w) else cv2.resize(frame, (out_w, out_h))

            # Crossfade audio-reattivo: dissolvenza morbida originale <-> effetto
            if _tc_active:
                _env = audio_env[tc_source]
                _raw = float(_env[frame_count]) if frame_count < len(_env) else float(_env[-1])
                _tc_smoothed = _tc_smoothed * tc_smooth + _raw * (1.0 - tc_smooth)
                _alpha = tc_alpha_min + (tc_alpha_max - tc_alpha_min) * float(np.clip(_tc_smoothed, 0.0, 1.0))
                processed = cv2.addWeighted(orig_cropped, 1.0 - _alpha, processed, _alpha, 0)

            # Applica maschera (nessuna = passa tutto, altrimenti blend)
            if _use_mask:
                if _has_animated_mask:
                    # Ricalcola la maschera per questo frame con interpolazione geometrica
                    glitch_mask = build_combined_mask_at_frame(
                        out_h, out_w, masks_list, frame_count, actual_total_frames)
                processed = apply_mask_blend(orig_cropped, processed, glitch_mask)

            out.write(processed)
            frame_count += 1
            progress_bar.progress(frame_count / actual_total_frames)
            status_text.text(f"🎬 Frame {frame_count}/{actual_total_frames} ({frame_count/actual_total_frames*100:.1f}%)")

        cap.release()
        out.release()

        fd2, final_output_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd2)

        if audio_mode == "0_originale":
            # Metti audio originale senza modifiche
            if check_ffmpeg():
                result = subprocess.run(['ffmpeg','-i',temp_video_path,'-i',video_path,
                                '-map','0:v:0','-map','1:a:0?',
                                '-c:v','copy','-c:a','aac',
                                '-shortest',final_output_path,'-y'],
                                capture_output=True,text=True)
                if result.returncode != 0:
                    return temp_video_path
                try: os.unlink(temp_video_path)
                except: pass
                return final_output_path
            return temp_video_path

        elif audio_mode in ("2_distruggi", "1_carica", "1_carica_distruggi") and check_ffmpeg():
            # Sorgente audio: file esterno caricato se presente, altrimenti quello del video
            if audio_mode in ("1_carica", "1_carica_distruggi") and audio_source_path:
                raw_audio = convert_audio_to_wav(audio_source_path)
            else:
                raw_audio = extract_audio(video_path)

            if raw_audio:
                if audio_mode == "1_carica":
                    # Audio esterno usato COSÌ COM'È, nessun glitch applicato
                    status_text.text("🎵 Uso audio caricato (non modificato)...")
                    glitched_audio = None
                    audio_to_use = raw_audio
                else:
                    # "2_distruggi" (audio del video) o "1_carica_distruggi" (audio esterno): applica il glitch
                    status_text.text("🎵 Glitch audio in corso...")
                    # effetti video-only usano 'noise' per l'audio (nessun corrispondente audio nativo)
                    VIDEO_ONLY_FX = {'pixel_sort','channel_shift','datamosh','byte_corrupt',
                                     'slice_shift','echo_smear','rgb_wave','mirror_blocks',
                                     'color_quantize','moire','feedback_loop','pixel_drift',
                                     'slit_scan','thermal','ascii_glitch','halftone',
                                     'chroma_pulse'}
                    a_eff = 'noise' if effect_type in VIDEO_ONLY_FX else effect_type
                    a_params = audio_params_override if audio_params_override is not None else (1.0, 1.0, 1.0)
                    glitched_audio = process_audio_glitch(raw_audio, a_eff, a_params)
                    audio_to_use = glitched_audio if glitched_audio else raw_audio

                cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_to_use,
                       '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                       '-shortest', final_output_path, '-y']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"FFmpeg audio merge error: {result.stderr[-500:]}")
                    return temp_video_path
                for p in [temp_video_path, raw_audio] + ([glitched_audio] if glitched_audio else []):
                    try: os.unlink(p)
                    except: pass
                return final_output_path
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
    """Ricomprime in H.264 CRF23 rispettando la risoluzione target esatta."""
    fd, output_path = tempfile.mkstemp(suffix='_h264.mp4')
    os.close(fd)
    TARGET_SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}
    try:
        if aspect_ratio in TARGET_SIZES:
            tw, th = TARGET_SIZES[aspect_ratio]
            # scale con padding nero se necessario per avere esattamente le dimensioni target
            vf = f"scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
        else:
            # risoluzione originale, garantisce multipli di 2
            vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
            '-vf', vf,
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            output_path, '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path
        else:
            return input_path
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
                 effect_type, params, include_audio, kf_df=None,
                 tc_info=None):
    """Genera il report testuale bilingue IT/EN."""

    effect_names_it = {
        'vhs':        'VHS Glitch',
        'distruttivo':'Distruttivo',
        'noise':      'Noise',
        'combined':   'Combinato',
        'broken_tv':  'Broken TV',
        'random':     'Random'
    }
    effect_names_en = {
        'vhs':        'VHS Glitch',
        'distruttivo':'Destructive',
        'noise':      'Noise',
        'combined':   'Combined',
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

    # Parametri leggibili (IT / EN)
    if effect_type == 'vhs' and isinstance(params, tuple):
        param_str_it = f"Intensita' {params[0]} | Scanline {params[1]} | Color Shift {params[2]}"
        param_str_en = f"Intensity {params[0]} | Scanline {params[1]} | Color Shift {params[2]}"
    elif effect_type == 'distruttivo' and isinstance(params, tuple):
        param_str_it = f"Block Size {params[0]} | Num Blocks {params[1]} | Displacement {params[2]}"
        param_str_en = f"Block Size {params[0]} | Block Count {params[1]} | Displacement {params[2]}"
    elif effect_type == 'noise' and isinstance(params, tuple):
        param_str_it = f"Intensita' {params[0]} | Coverage {params[1]} | Chaos {params[2]}"
        param_str_en = f"Intensity {params[0]} | Coverage {params[1]} | Chaos {params[2]}"
    elif effect_type == 'broken_tv' and isinstance(params, tuple):
        param_str_it = f"Shift {params[0]} | Line Height {params[1]} | Flicker {params[2]}"
        param_str_en = f"Shift {params[0]} | Line Height {params[1]} | Flicker {params[2]}"
    elif effect_type == 'combined' and isinstance(params, dict):
        active = [k.replace('apply_','').upper() for k,v in params.items() if k.startswith('apply_') and v]
        param_str_it = "Effetti attivi: " + ", ".join(active)
        param_str_en = "Active effects: " + ", ".join(active)
    elif effect_type == 'random' and isinstance(params, tuple):
        param_str_it = f"Livello casualita' {params[0]}"
        param_str_en = f"Randomness level {params[0]}"
    else:
        param_str_it = "—"
        param_str_en = "—"

    effect_hashtags = hashtag_map.get(effect_type, '')

    kf_block_it = ('* Keyframe Intensita\':' + chr(10) +
                   chr(10).join([f'  {row["Secondo"]}s -> {row["Intensita\'"]}' for _, row in kf_df.iterrows()])
                   ) if kf_df is not None and len(kf_df) >= 2 else ''
    kf_block_en = ('* Intensity Keyframes:' + chr(10) +
                   chr(10).join([f'  {row["Secondo"]}s -> {row["Intensita\'"]}' for _, row in kf_df.iterrows()])
                   ) if kf_df is not None and len(kf_df) >= 2 else ''

    tc_block_it = f'* Crossfade Audio-Reattivo: {tc_info}' if tc_info else ''
    tc_block_en = f'* Audio-Reactive Crossfade: {tc_info}' if tc_info else ''

    report = f"""[STUDIO_GLITCH_VIDEO] // VOL_01 // H.264 // DATA_CORRUPTION
:: MOTORE / ENGINE: videodistruktor [v1.1]
:: EFFETTO / EFFECT: {effect_names_it.get(effect_type, effect_type)} / {effect_names_en.get(effect_type, effect_type)}
:: PROCESSO / PROCESS: Frame Destruction / {'Audio Corruption' if include_audio else 'Video Only'}

"Il glitch non e' accaduto. E' stato scelto."
"The glitch didn't happen. It was chosen."

──────────────────────────────────
:: IT — SCHEDA TECNICA
──────────────────────────────────
* File: {original_name}
* Durata: {duration} sec | Frame: {total_frames} @ {fps}fps
* Risoluzione: {width}x{height}
* Originale: {original_size_mb} MB → Output: {output_size_mb} MB
* Effetto Audio: {'ON' if include_audio else 'OFF'}
* Parametri: {param_str_it}
{kf_block_it}
{tc_block_it}

──────────────────────────────────
:: EN — TECHNICAL LOG SHEET
──────────────────────────────────
* File: {original_name}
* Duration: {duration} sec | Frames: {total_frames} @ {fps}fps
* Resolution: {width}x{height}
* Original: {original_size_mb} MB → Output: {output_size_mb} MB
* Audio Effect: {'ON' if include_audio else 'OFF'}
* Parameters: {param_str_en}
{kf_block_en}
{tc_block_en}

> Regia e Algoritmo / Direction & Algorithm: Loop507

#loop507 #glitchart #videodistruktor #datacorruption #experimentalvideo
{effect_hashtags} #brutalistart #framecorruption #signalcorruption"""

    return report


# ─────────────────────────────────────────────────────────────────
# 🎞️ SESSIONE MULTI-CLIP — montaggio con crossfade tra più video
# ─────────────────────────────────────────────────────────────────
SESSION_EFFECT_PARAM_SPEC = {
    # effetto: [(label, min, max, default, step), ...] — stessi valori della UI video-singolo
    'pixel_sort':    [("Intensità",0.1,3.0,1.0,0.1), ("Soglia luma",0.1,1.0,0.5,0.05), ("Direzione",0.0,1.0,0.3,0.05)],
    'channel_shift': [("Intensità",0.1,3.0,1.0,0.1), ("Spread",0.1,3.0,1.0,0.1), ("Verticale",0.0,1.0,0.3,0.05)],
    'datamosh':      [("Intensità",0.1,3.0,1.0,0.1), ("Block size",0.1,3.0,1.0,0.1), ("Chaos",0.1,3.0,1.0,0.1)],
    'byte_corrupt':  [("Intensità",0.1,3.0,1.0,0.1), ("Chunk size",0.1,3.0,1.0,0.1), ("Random",0.0,1.0,0.7,0.05)],
    'slice_shift':   [("Intensità",0.1,3.0,1.0,0.1), ("Num slices",0.1,3.0,1.0,0.1), ("Drift",0.1,3.0,1.0,0.1)],
    'echo_smear':    [("Intensità",0.1,3.0,1.0,0.1), ("Decay",0.0,1.0,0.5,0.05), ("Smear",0.1,3.0,1.0,0.1)],
    'rgb_wave':      [("Intensità",0.1,3.0,1.0,0.1), ("Frequenza",0.1,5.0,1.0,0.1), ("Phase chaos",0.0,1.0,0.5,0.05)],
    'mirror_blocks': [("Intensità",0.1,3.0,1.0,0.1), ("Block size",0.1,3.0,1.0,0.1), ("Flip prob",0.0,1.0,0.5,0.05)],
    'color_quantize':[("Intensità",0.1,3.0,1.0,0.1), ("Livelli",0.1,3.0,1.0,0.1), ("Dither",0.0,1.0,0.5,0.05)],
    'moire':         [("Intensità",0.1,3.0,1.0,0.1), ("Frequenza",0.1,5.0,1.0,0.1), ("Angolo",0.0,1.0,0.5,0.05)],
    'feedback_loop': [("Intensità",0.1,3.0,1.0,0.1), ("Zoom",0.0,1.0,0.5,0.05), ("Rotazione",0.0,1.0,0.5,0.05)],
    'pixel_drift':   [("Intensità",0.1,3.0,1.0,0.1), ("Drift speed",0.1,3.0,1.0,0.1), ("Turbolenza",0.0,1.0,0.5,0.05)],
    'slit_scan':     [("Intensità",0.1,3.0,1.0,0.1), ("Speed",0.1,3.0,1.0,0.1), ("Tilt",0.0,1.0,0.5,0.05)],
    'thermal':       [("Intensità",0.1,3.0,1.0,0.1), ("Noise sensore",0.0,1.0,0.5,0.05), ("Aberrazione",0.0,1.0,0.5,0.05)],
    'ascii_glitch':  [("Intensità",0.1,3.0,1.0,0.1), ("Block size",0.1,1.0,0.5,0.05), ("Chaos",0.0,1.0,0.5,0.05)],
    'halftone':      [("Intensità",0.1,3.0,1.0,0.1), ("Dot size",0.1,1.0,0.5,0.05), ("Angolo",0.0,1.0,0.3,0.05)],
    'chroma_pulse':  [("Intensità",0.1,3.0,1.0,0.1), ("Radiale",0.0,1.0,0.5,0.05), ("Pulse speed",0.1,3.0,1.0,0.1)],
    'vhs':           [("Intensità",0.1,3.0,1.0,0.1), ("Scanline",0.1,3.0,1.0,0.1), ("Color shift",0.1,3.0,1.0,0.1)],
    'distruttivo':   [("Block size",0.1,3.0,1.0,0.1), ("Num blocks",0.1,3.0,1.0,0.1), ("Displacement",0.1,3.0,1.0,0.1)],
    'noise':         [("Intensità",0.1,3.0,1.0,0.1), ("Coverage",0.1,3.0,1.0,0.1), ("Chaos",0.1,3.0,1.0,0.1)],
    'broken_tv':     [("Shift",0.1,3.0,1.0,0.1), ("Line height",0.1,3.0,1.0,0.1), ("Flicker",0.1,3.0,1.0,0.1)],
}
SESSION_EFFECTS = list(SESSION_EFFECT_PARAM_SPEC.keys()) + ['combined', 'random']
SESSION_EFFECT_LABELS = {
    "pixel_sort":"🔀 Pixel Sort", "channel_shift":"🌈 Channel Shift", "datamosh":"💾 Datamosh",
    "byte_corrupt":"🦠 Byte Corrupt", "slice_shift":"✂️ Slice Shift", "echo_smear":"👻 Echo Smear",
    "rgb_wave":"🌊 RGB Wave", "mirror_blocks":"🪞 Mirror Blocks", "color_quantize":"🎨 Color Quantize",
    "moire":"🕸️ Moiré Pattern", "feedback_loop":"🔁 Feedback Loop", "pixel_drift":"💧 Pixel Drift",
    "slit_scan":"📷 Slit Scan", "thermal":"🌡️ Thermal Vision", "ascii_glitch":"⌨️ ASCII Glitch",
    "halftone":"🔵 Halftone Destroy", "chroma_pulse":"💥 Chroma Pulse", "vhs":"📼 VHS",
    "broken_tv":"📻 Broken TV", "noise":"📺 Noise", "distruttivo":"💥 Distruttivo",
    "combined":"🌟 Combinato", "random":"🎲 Random"
}
SESSION_TRANSITIONS = ["fade", "fadeblack", "wipeleft", "wiperight", "circleopen", "pixelize"]
SESSION_TARGET_SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}


def render_mini_effect_picker(key_prefix, default_effect="vhs"):
    """Selettore effetto (usato per l'effetto globale e per gli effetti individuali per-clip).
    Copre TUTTI gli effetti disponibili nel motore, non solo i 5 base."""
    effect = st.selectbox("Effetto", SESSION_EFFECTS, format_func=lambda x: SESSION_EFFECT_LABELS[x],
                          index=SESSION_EFFECTS.index(default_effect), key=f"{key_prefix}_efftype")

    if effect in SESSION_EFFECT_PARAM_SPEC:
        spec = SESSION_EFFECT_PARAM_SPEC[effect]
        cols = st.columns(len(spec))
        vals = []
        for i, (label, mn, mx, dft, step) in enumerate(spec):
            with cols[i]:
                vals.append(st.slider(label, mn, mx, dft, step, key=f"{key_prefix}_p{i}"))
        params = tuple(vals)

    elif effect == 'combined':
        st.caption("Seleziona gli effetti da combinare:")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: apply_vhs = st.checkbox("📼 VHS", value=True, key=f"{key_prefix}_cvhs")
        with cc2: apply_dst = st.checkbox("💥 Distr.", value=True, key=f"{key_prefix}_cdst")
        with cc3: apply_nse = st.checkbox("📺 Noise", value=True, key=f"{key_prefix}_cnse")
        with cc4: apply_tv  = st.checkbox("📻 TV", value=True, key=f"{key_prefix}_ctv")
        params = {"apply_vhs": apply_vhs, "apply_distruttivo": apply_dst,
                  "apply_noise": apply_nse, "apply_broken_tv": apply_tv}
        if apply_vhs:
            c1,c2,c3 = st.columns(3)
            with c1: params["vhs_intensity"]    = st.slider("VHS Intensità",0.1,3.0,1.0,0.1,key=f"{key_prefix}_vi")
            with c2: params["vhs_scanline_freq"]= st.slider("VHS Scanline", 0.1,3.0,1.0,0.1,key=f"{key_prefix}_vs")
            with c3: params["vhs_color_shift"]  = st.slider("VHS Color",    0.1,3.0,1.0,0.1,key=f"{key_prefix}_vc")
        if apply_dst:
            c1,c2,c3 = st.columns(3)
            with c1: params["dest_block_size"]  = st.slider("Dest Block",  0.1,3.0,1.0,0.1,key=f"{key_prefix}_db")
            with c2: params["dest_num_blocks"]  = st.slider("Dest Num",    0.1,3.0,1.0,0.1,key=f"{key_prefix}_dn")
            with c3: params["dest_displacement"]= st.slider("Dest Disp",   0.1,3.0,1.0,0.1,key=f"{key_prefix}_dd")
        if apply_nse:
            c1,c2,c3 = st.columns(3)
            with c1: params["noise_intensity"]  = st.slider("Noise Int",  0.1,3.0,1.0,0.1,key=f"{key_prefix}_ni")
            with c2: params["noise_coverage"]   = st.slider("Noise Cov",  0.1,3.0,1.0,0.1,key=f"{key_prefix}_nc")
            with c3: params["noise_chaos"]      = st.slider("Noise Chaos",0.1,3.0,1.0,0.1,key=f"{key_prefix}_nh")
        if apply_tv:
            c1,c2,c3 = st.columns(3)
            with c1: params["tv_shift_intensity"]= st.slider("TV Shift",  0.1,3.0,1.0,0.1,key=f"{key_prefix}_ts")
            with c2: params["tv_line_height"]    = st.slider("TV Line",   0.1,3.0,1.0,0.1,key=f"{key_prefix}_tl")
            with c3: params["tv_flicker_prob"]   = st.slider("TV Flicker",0.1,3.0,1.0,0.1,key=f"{key_prefix}_tf")

    else:  # random
        p1 = st.slider("Livello casualità", 0.1, 3.0, 1.0, 0.1, key=f"{key_prefix}_p1")
        params = (p1,)

    return effect, params


def build_xfade_chain(clips, crossfades, transitions, audio_xfade):
    """Costruisce ed esegue la catena ffmpeg xfade (video) + acrossfade (audio) tra N clip normalizzate."""
    n = len(clips)
    inputs = []
    for c in clips:
        inputs += ["-i", c["video"]]
    if audio_xfade:
        for c in clips:
            inputs += ["-i", c["audio"]]

    filter_parts = []
    running = clips[0]["duration"]
    last_v = "[0:v]"
    for i in range(1, n):
        off = max(0.0, running - crossfades[i - 1])
        out_lbl = f"[v{i}]"
        filter_parts.append(
            f"{last_v}[{i}:v]xfade=transition={transitions[i-1]}:duration={crossfades[i-1]:.2f}:offset={off:.3f}{out_lbl}"
        )
        running = running + clips[i]["duration"] - crossfades[i - 1]
        last_v = out_lbl

    maps = ["-map", last_v]
    if audio_xfade:
        last_a = f"[{n}:a]"
        for i in range(1, n):
            out_lbl = f"[a{i}]"
            filter_parts.append(f"{last_a}[{n+i}:a]acrossfade=d={crossfades[i-1]:.2f}{out_lbl}")
            last_a = out_lbl
        maps += ["-map", last_a]

    filter_complex = ";".join(filter_parts)
    fd, out_path = tempfile.mkstemp(suffix='_session.mp4')
    os.close(fd)
    cmd = ["ffmpeg"] + inputs + ["-filter_complex", filter_complex] + maps + [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        out_path, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"❌ Errore FFmpeg xfade: {result.stderr[-800:]}")
        return None
    return out_path


def run_multiclip_session(clips_cfg, global_effect, global_params, aspect_ratio, target_fps, audio_xfade):
    """Applica l'effetto (globale o individuale) a ogni clip, normalizza fps/risoluzione, poi le monta con crossfade."""
    if len(clips_cfg) < 2:
        st.error("Servono almeno 2 clip per montare una sessione.")
        return None

    tmp_dir = tempfile.mkdtemp()
    processed = []
    prog = st.progress(0)
    stat = st.empty()
    tw, th = SESSION_TARGET_SIZES.get(aspect_ratio, (0, 0))

    for i, entry in enumerate(clips_cfg):
        stat.text(f"🎬 Clip {i+1}/{len(clips_cfg)}: {entry['name']} — applico effetto...")
        in_path = os.path.join(tmp_dir, f"in_{i}.mp4")
        with open(in_path, "wb") as f:
            f.write(entry["file"].getbuffer())

        eff = entry.get("effect_type", global_effect) if entry.get("mode") == "individuale" else global_effect
        prm = entry.get("params", global_params) if entry.get("mode") == "individuale" else global_params

        proc_path = process_video(in_path, eff, prm, max_frames=None,
                                  audio_mode="0_originale", aspect_ratio=aspect_ratio)
        if proc_path is None:
            st.error(f"❌ Errore processando {entry['name']}")
            return None

        norm_path = os.path.join(tmp_dir, f"norm_{i}.mp4")
        if tw and th:
            vf = f"scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        else:
            vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
        r = subprocess.run(["ffmpeg", "-i", proc_path, "-r", str(target_fps), "-vf", vf,
                           "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-an", norm_path, "-y"], capture_output=True, text=True)
        if r.returncode != 0:
            st.error(f"❌ Errore normalizzazione {entry['name']}: {r.stderr[-400:]}")
            return None

        _, _, _, _, dur = get_video_info(norm_path)

        aud_path = None
        if audio_xfade:
            aud_path = os.path.join(tmp_dir, f"aud_{i}.wav")
            ar = subprocess.run(["ffmpeg", "-i", in_path, "-vn", "-acodec", "pcm_s16le",
                               "-ar", "44100", "-ac", "2", aud_path, "-y"], capture_output=True, text=True)
            if ar.returncode != 0 or not os.path.exists(aud_path) or get_file_size_mb(aud_path) == 0:
                # nessuna traccia audio nella clip: genera silenzio della stessa durata
                subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                               "-t", str(dur), aud_path, "-y"], capture_output=True, text=True)

        processed.append({"video": norm_path, "audio": aud_path, "duration": dur})
        prog.progress((i + 1) / (len(clips_cfg) + 1))

    stat.text("🔀 Applico crossfade tra le clip...")
    crossfades  = [c.get("crossfade_dur", 1.0) for c in clips_cfg[:-1]]
    transitions = [c.get("transition_type", "fade") for c in clips_cfg[:-1]]
    for i in range(len(crossfades)):
        max_allowed = min(processed[i]["duration"], processed[i + 1]["duration"]) - 0.2
        crossfades[i] = max(0.1, min(crossfades[i], max_allowed if max_allowed > 0.1 else 0.1))

    final_path = build_xfade_chain(processed, crossfades, transitions, audio_xfade)
    prog.progress(1.0)
    if final_path:
        stat.text("✅ Sessione completata!")
    return final_path


def render_session_mode():
    if "session_clip_settings" not in st.session_state:
        st.session_state.session_clip_settings = []

    uploaded_clips = st.file_uploader("Carica le clip, in ordine di caricamento", type=["mp4", "mov", "avi", "mkv"],
                                      accept_multiple_files=True, key="session_uploader")

    if uploaded_clips:
        existing = {c["key"]: c for c in st.session_state.session_clip_settings}
        new_list = []
        for f in uploaded_clips:
            k = f"{f.name}_{f.size}"
            entry = existing.get(k, {"key": k, "name": f.name, "mode": "globale"})
            entry["file"] = f
            new_list.append(entry)
        st.session_state.session_clip_settings = new_list

    clips_cfg = st.session_state.session_clip_settings

    if not clips_cfg:
        st.info("Carica almeno 2 video per iniziare a montare la sessione.")
        return

    st.markdown(f"**{len(clips_cfg)} clip in coda**")
    for i, entry in enumerate(clips_cfg):
        with st.container(border=True):
            top = st.columns([6, 1, 1])
            top[0].markdown(f"**{i+1}. {entry['name']}**")
            if top[1].button("⬆️", key=f"up_{entry['key']}", disabled=(i == 0)):
                clips_cfg[i-1], clips_cfg[i] = clips_cfg[i], clips_cfg[i-1]
                st.rerun()
            if top[2].button("⬇️", key=f"down_{entry['key']}", disabled=(i == len(clips_cfg)-1)):
                clips_cfg[i+1], clips_cfg[i] = clips_cfg[i], clips_cfg[i+1]
                st.rerun()

            mode_choice = st.radio("Effetto per questa clip", ["🌐 Globale", "🎛️ Individuale"],
                                   index=0 if entry.get("mode", "globale") == "globale" else 1,
                                   horizontal=True, key=f"mode_{entry['key']}")
            entry["mode"] = "globale" if mode_choice == "🌐 Globale" else "individuale"
            if entry["mode"] == "individuale":
                eff, prm = render_mini_effect_picker(f"ind_{entry['key']}")
                entry["effect_type"], entry["params"] = eff, prm

            if i < len(clips_cfg) - 1:
                cx1, cx2 = st.columns(2)
                with cx1:
                    entry["crossfade_dur"] = st.slider(f"↔️ Crossfade verso clip {i+2} (sec)", 0.2, 3.0,
                                                       entry.get("crossfade_dur", 1.0), 0.1,
                                                       key=f"xf_{entry['key']}")
                with cx2:
                    entry["transition_type"] = st.selectbox("Tipo transizione", SESSION_TRANSITIONS,
                                                            key=f"tt_{entry['key']}")

    st.markdown("#### 🌐 Effetto globale (per le clip in modalità Globale)")
    global_effect, global_params = render_mini_effect_picker("session_global")

    st.markdown("#### ⚙️ Impostazioni sessione")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        session_aspect = st.selectbox("Formato output", ["16:9", "9:16", "1:1", "Originale"], key="session_aspect")
    with sc2:
        session_fps = st.selectbox("FPS sessione", [24, 25, 30], key="session_fps")
    with sc3:
        session_audio_xfade = st.checkbox("🔊 Crossfade audio", value=True, key="session_audio_xfade")

    if len(clips_cfg) < 2:
        st.warning("⚠️ Carica almeno 2 clip per poter montare una sessione con crossfade.")
        return

    if st.button("🎬 Genera Sessione", type="primary", key="session_generate"):
        if not check_ffmpeg():
            st.error("❌ FFmpeg non disponibile: impossibile montare la sessione.")
            return
        final_path = run_multiclip_session(clips_cfg, global_effect, global_params,
                                           session_aspect, session_fps, session_audio_xfade)
        if final_path:
            st.video(final_path)
            with open(final_path, "rb") as f:
                st.download_button("⬇️ Scarica sessione montata", f, file_name="videodistruktor_session.mp4",
                                  mime="video/mp4", key="session_download")


# Interfaccia Streamlit principale
if 'report_data' not in st.session_state: st.session_state.report_data = ""
if 'video_ready' not in st.session_state: st.session_state.video_ready = False
if 'h264_path'   not in st.session_state: st.session_state.h264_path   = ""
if 'effect_name_saved' not in st.session_state: st.session_state.effect_name_saved = ""
if 'orig_filename' not in st.session_state: st.session_state.orig_filename = ""
if 'output_video_name' not in st.session_state: st.session_state.output_video_name = "glitch_output.mp4"
if 'report_filename'   not in st.session_state: st.session_state.report_filename   = "report_glitch.txt"
if 'use_audio_reactive' not in st.session_state: st.session_state.use_audio_reactive = False
if 'glitched_audio_path' not in st.session_state: st.session_state.glitched_audio_path = ""
if 'glitched_audio_name' not in st.session_state: st.session_state.glitched_audio_name = ""

with st.expander("🎞️ Sessione Multi-Clip — carica più video, montali con crossfade (Beta)", expanded=False):
    st.caption("Carica più clip, ordinale con ⬆️⬇️, scegli l'effetto (globale o per singola clip) "
              "e la durata/tipo di crossfade tra ogni coppia di clip.")
    render_session_mode()

st.divider()

if uploaded_file is not None:
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        st.warning("⚠️ FFmpeg non disponibile — effetti audio disabilitati.")

    # Salva il file video in temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # ─── LAYOUT 2 COLONNE: sinistra=controlli, destra=anteprima ───
    col_ctrl, col_prev = st.columns([1, 1], gap="large")

    with col_ctrl:
        # ── EFFETTO ──────────────────────────────────────────────
        effect_type = st.selectbox(
            "🎭 Effetto glitch:",
            ["pixel_sort", "channel_shift", "datamosh", "byte_corrupt", "slice_shift",
             "echo_smear", "rgb_wave", "mirror_blocks", "color_quantize",
             "moire", "feedback_loop", "pixel_drift",
             "slit_scan", "thermal", "ascii_glitch", "halftone", "chroma_pulse",
             "vhs", "broken_tv", "noise", "distruttivo", "combined", "random"],
            format_func=lambda x: {
                "pixel_sort":    "🔀 Pixel Sort",
                "channel_shift": "🌈 Channel Shift",
                "datamosh":      "💾 Datamosh",
                "byte_corrupt":  "🦠 Byte Corrupt",
                "slice_shift":   "✂️ Slice Shift",
                "echo_smear":    "👻 Echo Smear",
                "rgb_wave":      "🌊 RGB Wave",
                "mirror_blocks": "🪞 Mirror Blocks",
                "color_quantize":"🎨 Color Quantize",
                "moire":         "🕸️ Moiré Pattern",
                "feedback_loop": "🔁 Feedback Loop",
                "pixel_drift":   "💧 Pixel Drift",
                "slit_scan":     "📷 Slit Scan",
                "thermal":       "🌡️ Thermal Vision",
                "ascii_glitch":  "⌨️ ASCII Glitch",
                "halftone":      "🔵 Halftone Destroy",
                "chroma_pulse":  "💥 Chroma Pulse",
                "vhs":           "📼 VHS",
                "broken_tv":     "📻 Broken TV",
                "noise":         "📺 Noise",
                "distruttivo":   "💥 Distruttivo",
                "combined":      "🌟 Combinato",
                "random":        "🎲 Random"
            }[x]
        )

        # ── PARAMETRI PER EFFETTO ────────────────────────────────
        params = {}
        audio_params_override = None

        if effect_type == 'pixel_sort':
            c1,c2,c3 = st.columns(3)
            with c1: ps_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: ps_threshold = st.slider("Soglia luma", 0.1, 1.0, 0.5, 0.05)
            with c3: ps_direction = st.slider("Direzione", 0.0, 1.0, 0.3, 0.05)
            params = (ps_intensity, ps_threshold, ps_direction)

        elif effect_type == 'channel_shift':
            c1,c2,c3 = st.columns(3)
            with c1: cs_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: cs_spread    = st.slider("Spread", 0.1, 3.0, 1.0, 0.1)
            with c3: cs_mode      = st.slider("Verticale", 0.0, 1.0, 0.3, 0.05)
            params = (cs_intensity, cs_spread, cs_mode)

        elif effect_type == 'datamosh':
            c1,c2,c3 = st.columns(3)
            with c1: dm_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: dm_block_size = st.slider("Block size", 0.1, 3.0, 1.0, 0.1)
            with c3: dm_chaos      = st.slider("Chaos", 0.1, 3.0, 1.0, 0.1)
            params = (dm_intensity, dm_block_size, dm_chaos)

        elif effect_type == 'byte_corrupt':
            c1,c2,c3 = st.columns(3)
            with c1: bc_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: bc_chunk_size = st.slider("Chunk size", 0.1, 3.0, 1.0, 0.1)
            with c3: bc_randomize  = st.slider("Random", 0.0, 1.0, 0.7, 0.05)
            params = (bc_intensity, bc_chunk_size, bc_randomize)

        elif effect_type == 'slice_shift':
            c1,c2,c3 = st.columns(3)
            with c1: ss_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: ss_num_slices = st.slider("Num slices", 0.1, 3.0, 1.0, 0.1)
            with c3: ss_drift      = st.slider("Drift", 0.1, 3.0, 1.0, 0.1)
            params = (ss_intensity, ss_num_slices, ss_drift)

        elif effect_type == 'slit_scan':
            c1,c2,c3 = st.columns(3)
            with c1: sl_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: sl_speed     = st.slider("Speed", 0.1, 3.0, 1.0, 0.1)
            with c3: sl_tilt      = st.slider("Tilt", 0.0, 1.0, 0.5, 0.05)
            params = (sl_intensity, sl_speed, sl_tilt)

        elif effect_type == 'thermal':
            c1,c2,c3 = st.columns(3)
            with c1: th_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: th_noise      = st.slider("Noise sensore", 0.0, 1.0, 0.5, 0.05)
            with c3: th_aberration = st.slider("Aberrazione", 0.0, 1.0, 0.5, 0.05)
            params = (th_intensity, th_noise, th_aberration)

        elif effect_type == 'ascii_glitch':
            c1,c2,c3 = st.columns(3)
            with c1: ag_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: ag_block_size = st.slider("Block size", 0.1, 1.0, 0.5, 0.05)
            with c3: ag_chaos      = st.slider("Chaos", 0.0, 1.0, 0.5, 0.05)
            params = (ag_intensity, ag_block_size, ag_chaos)

        elif effect_type == 'halftone':
            c1,c2,c3 = st.columns(3)
            with c1: ht_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: ht_dot_size  = st.slider("Dot size", 0.1, 1.0, 0.5, 0.05)
            with c3: ht_angle     = st.slider("Angolo", 0.0, 1.0, 0.3, 0.05)
            params = (ht_intensity, ht_dot_size, ht_angle)

        elif effect_type == 'chroma_pulse':
            c1,c2,c3 = st.columns(3)
            with c1: cp_intensity   = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: cp_radial      = st.slider("Radiale", 0.0, 1.0, 0.5, 0.05)
            with c3: cp_pulse_speed = st.slider("Pulse speed", 0.1, 3.0, 1.0, 0.1)
            params = (cp_intensity, cp_radial, cp_pulse_speed)

        elif effect_type == 'moire':
            c1,c2,c3 = st.columns(3)
            with c1: mo_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: mo_freq      = st.slider("Frequenza", 0.1, 5.0, 1.0, 0.1)
            with c3: mo_angle     = st.slider("Angolo", 0.0, 1.0, 0.5, 0.05)
            params = (mo_intensity, mo_freq, mo_angle)

        elif effect_type == 'feedback_loop':
            c1,c2,c3 = st.columns(3)
            with c1: fl_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: fl_zoom      = st.slider("Zoom", 0.0, 1.0, 0.5, 0.05)
            with c3: fl_rotate    = st.slider("Rotazione", 0.0, 1.0, 0.5, 0.05)
            params = (fl_intensity, fl_zoom, fl_rotate)

        elif effect_type == 'pixel_drift':
            c1,c2,c3 = st.columns(3)
            with c1: pd_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: pd_speed      = st.slider("Drift speed", 0.1, 3.0, 1.0, 0.1)
            with c3: pd_turbulence = st.slider("Turbolenza", 0.0, 1.0, 0.5, 0.05)
            params = (pd_intensity, pd_speed, pd_turbulence)

        elif effect_type == 'echo_smear':
            c1,c2,c3 = st.columns(3)
            with c1: es_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: es_decay     = st.slider("Decay", 0.0, 1.0, 0.5, 0.05)
            with c3: es_smear     = st.slider("Smear", 0.1, 3.0, 1.0, 0.1)
            params = (es_intensity, es_decay, es_smear)

        elif effect_type == 'rgb_wave':
            c1,c2,c3 = st.columns(3)
            with c1: rw_intensity   = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: rw_freq        = st.slider("Frequenza", 0.1, 5.0, 1.0, 0.1)
            with c3: rw_phase_chaos = st.slider("Phase chaos", 0.0, 1.0, 0.5, 0.05)
            params = (rw_intensity, rw_freq, rw_phase_chaos)

        elif effect_type == 'mirror_blocks':
            c1,c2,c3 = st.columns(3)
            with c1: mb_intensity  = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: mb_block_size = st.slider("Block size", 0.1, 3.0, 1.0, 0.1)
            with c3: mb_flip_prob  = st.slider("Flip prob", 0.0, 1.0, 0.5, 0.05)
            params = (mb_intensity, mb_block_size, mb_flip_prob)

        elif effect_type == 'color_quantize':
            c1,c2,c3 = st.columns(3)
            with c1: cq_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: cq_levels    = st.slider("Livelli", 0.1, 3.0, 1.0, 0.1)
            with c3: cq_dither    = st.slider("Dither", 0.0, 1.0, 0.5, 0.05)
            params = (cq_intensity, cq_levels, cq_dither)

        elif effect_type == 'vhs':
            c1,c2,c3 = st.columns(3)
            with c1: vhs_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: scanline_freq = st.slider("Scanline", 0.1, 3.0, 1.0, 0.1)
            with c3: color_shift   = st.slider("Color shift", 0.1, 3.0, 1.0, 0.1)
            params = (vhs_intensity, scanline_freq, color_shift)

        elif effect_type == 'distruttivo':
            c1,c2,c3 = st.columns(3)
            with c1: block_size   = st.slider("Block size", 0.1, 3.0, 1.0, 0.1)
            with c2: num_blocks   = st.slider("Num blocks", 0.1, 3.0, 1.0, 0.1)
            with c3: displacement = st.slider("Displacement", 0.1, 3.0, 1.0, 0.1)
            params = (block_size, num_blocks, displacement)

        elif effect_type == 'noise':
            c1,c2,c3 = st.columns(3)
            with c1: noise_intensity = st.slider("Intensità", 0.1, 3.0, 1.0, 0.1)
            with c2: coverage        = st.slider("Coverage", 0.1, 3.0, 1.0, 0.1)
            with c3: chaos           = st.slider("Chaos", 0.1, 3.0, 1.0, 0.1)
            params = (noise_intensity, coverage, chaos)

        elif effect_type == 'broken_tv':
            c1,c2,c3 = st.columns(3)
            with c1: shift_intensity = st.slider("Shift", 0.1, 3.0, 1.0, 0.1)
            with c2: line_height     = st.slider("Line height", 0.1, 3.0, 1.0, 0.1)
            with c3: flicker_prob    = st.slider("Flicker", 0.1, 3.0, 1.0, 0.1)
            params = (shift_intensity, line_height, flicker_prob)

        elif effect_type == 'combined':
            st.write("Seleziona gli effetti da combinare:")
            apply_vhs        = st.checkbox("📼 VHS", value=True)
            apply_distruttivo= st.checkbox("💥 Distruttivo", value=True)
            apply_noise      = st.checkbox("📺 Noise", value=True)
            apply_broken_tv  = st.checkbox("📻 Broken TV", value=True)
            params = {"apply_vhs": apply_vhs, "apply_distruttivo": apply_distruttivo,
                      "apply_noise": apply_noise, "apply_broken_tv": apply_broken_tv}
            if apply_vhs:
                c1,c2,c3 = st.columns(3)
                with c1: params["vhs_intensity"]   = st.slider("VHS Intensità", 0.1, 3.0, 1.0, 0.1, key="vhs_int")
                with c2: params["vhs_scanline_freq"]= st.slider("VHS Scanline",  0.1, 3.0, 1.0, 0.1, key="vhs_scan")
                with c3: params["vhs_color_shift"]  = st.slider("VHS Color",     0.1, 3.0, 1.0, 0.1, key="vhs_color")
            if apply_distruttivo:
                c1,c2,c3 = st.columns(3)
                with c1: params["dest_block_size"]  = st.slider("Dest Block",   0.1, 3.0, 1.0, 0.1, key="dest_block")
                with c2: params["dest_num_blocks"]  = st.slider("Dest Num",     0.1, 3.0, 1.0, 0.1, key="dest_num")
                with c3: params["dest_displacement"]= st.slider("Dest Disp",    0.1, 3.0, 1.0, 0.1, key="dest_disp")
            if apply_noise:
                c1,c2,c3 = st.columns(3)
                with c1: params["noise_intensity"]  = st.slider("Noise Int",    0.1, 3.0, 1.0, 0.1, key="noise_int")
                with c2: params["noise_coverage"]   = st.slider("Noise Cov",    0.1, 3.0, 1.0, 0.1, key="noise_cov")
                with c3: params["noise_chaos"]      = st.slider("Noise Chaos",  0.1, 3.0, 1.0, 0.1, key="noise_chaos")
            if apply_broken_tv:
                c1,c2,c3 = st.columns(3)
                with c1: params["tv_shift_intensity"]    = st.slider("TV Shift",  0.1, 3.0, 1.0, 0.1, key="tv_shift")
                with c2: params["tv_line_height"]        = st.slider("TV Line",   0.1, 3.0, 1.0, 0.1, key="tv_line")
                with c3: params["tv_flicker_prob"]       = st.slider("TV Flicker",0.1, 3.0, 1.0, 0.1, key="tv_flick")
            audio_params_override = None  # combined usa il dict params

        elif effect_type == 'random':
            random_level = st.slider("Livello casualità", 0.1, 3.0, 1.0, 0.1)
            params = (random_level,)

        st.markdown("---")

        # ── MODALITÀ AUDIO ───────────────────────────────────────
        # Un solo file uploader, 4 modalità chiare
        st.markdown("**🎵 Audio**")
        if not ffmpeg_available:
            audio_mode = "0_originale"
            st.caption("⚠️ FFmpeg non disponibile — audio originale invariato.")
        else:
            audio_mode = st.radio(
                "Modalità:",
                ["0_originale", "2_distruggi", "1_mix", "3_esterno", "4_esterno_distruggi"],
                format_func=lambda x: {
                    "0_originale":        "🔇 Originale (nessuna modifica)",
                    "2_distruggi":        "💥 Distruggi audio originale",
                    "1_mix":              "🎛️ Mixa originale + glitch",
                    "3_esterno":          "🎵 Usa audio caricato",
                    "4_esterno_distruggi":"🔥 Usa audio caricato + distruggi",
                }[x],
                horizontal=False,
                key="audio_mode_radio"
            )

        uploaded_audio_inline = None
        if audio_mode in ("3_esterno", "4_esterno_distruggi"):
            uploaded_audio_inline = st.file_uploader(
                "📂 Carica audio (mp3/wav/aac/ogg/flac/m4a)",
                type=["mp3","wav","aac","ogg","flac","m4a"],
                key="audio_inline_uploader"
            )

        # Parametri audio override solo per effetti con audio dedicato
        AUDIO_FX = {'vhs','distruttivo','noise','broken_tv','combined'}
        include_audio = audio_mode != "0_originale" and ffmpeg_available
        if include_audio and effect_type in AUDIO_FX and effect_type != 'combined':
            with st.expander("⚙️ Parametri audio effetto", expanded=False):
                ca1,ca2,ca3 = st.columns(3)
                with ca1: a_p1 = st.slider("Audio P1", 0.1, 3.0, 1.0, 0.1, key="aud_p1")
                with ca2: a_p2 = st.slider("Audio P2", 0.1, 3.0, 1.0, 0.1, key="aud_p2")
                with ca3: a_p3 = st.slider("Audio P3", 0.1, 3.0, 1.0, 0.1, key="aud_p3")
                audio_params_override = (a_p1, a_p2, a_p3)

        st.markdown("---")

        # ── AUDIO REACTIVE ───────────────────────────────────────
        use_audio_reactive = st.toggle("🎚️ Audio Reactive",
            help="Analizza RMS/beat/freq e modula i parametri frame per frame.")
        ar_intensity = 0.0
        if use_audio_reactive:
            ar_intensity = st.slider("Intensità reattività", 0.1, 3.0, 1.0, 0.1, key="ar_intensity")

        # ── BEAT SYNC ────────────────────────────────────────────
        beat_sync = st.toggle("🥁 Beat Sync",
            help="L'effetto scatta esattamente sui beat rilevati, con decadimento esponenziale.")
        beat_decay   = 6
        beat_intensity = 2.0
        if beat_sync:
            use_audio_reactive = False  # i due modi si escludono
            bs1, bs2 = st.columns(2)
            with bs1:
                beat_intensity = st.slider("Intensità picco", 0.5, 4.0, 2.0, 0.1, key="bs_intensity")
            with bs2:
                beat_decay = st.slider("Decay (frame)", 1, 30, 6, 1, key="bs_decay",
                    help="Frame per tornare al 37% del picco. 6@24fps = ~250ms")

        # ── CROSSFADE AUDIO-REATTIVO (originale <-> effetto) ─────
        use_temporal_crossfade = st.toggle("🌊 Crossfade Audio-Reattivo",
            help="Dissolvenza morbida tra frame originale ed effetto, invece del taglio secco. "
                 "L'intensità della dissolvenza segue l'audio (RMS/beat/frequenze).")
        tc_source     = "rms"
        tc_alpha_min  = 0.0
        tc_alpha_max  = 1.0
        tc_smooth     = 0.7
        if use_temporal_crossfade:
            tc1, tc2 = st.columns(2)
            with tc1:
                tc_source = st.selectbox("Sorgente audio", ["rms", "beats", "low_freq", "high_freq", "spectral"],
                    format_func=lambda x: {"rms":"🔊 RMS (energia)", "beats":"🥁 Beat",
                                            "low_freq":"🔈 Bassi", "high_freq":"🔉 Alti",
                                            "spectral":"📊 Centroide spettrale"}[x],
                    key="tc_source")
                tc_smooth = st.slider("Morbidezza dissolvenza", 0.0, 0.95, 0.7, 0.05, key="tc_smooth",
                    help="Più alto = transizioni più lente e fluide, meno alto = più reattivo/scattante.")
            with tc2:
                tc_alpha_min = st.slider("Effetto visibile (minimo)", 0.0, 1.0, 0.0, 0.05, key="tc_alpha_min",
                    help="Quanto effetto resta visibile nei momenti 'silenziosi'.")
                tc_alpha_max = st.slider("Effetto visibile (massimo)", 0.0, 1.0, 1.0, 0.05, key="tc_alpha_max",
                    help="Quanto effetto è visibile nei picchi audio.")

        # ── KEYFRAME ─────────────────────────────────────────────
        kf_df = None
        use_keyframes = False
        if effect_type not in ['combined','random']:
            use_keyframes = st.toggle("⏱️ Animazione intensità",
                help="Interpola l'intensità dall'inizio alla fine del video.")
            if use_keyframes:
                ck1, ck2 = st.columns(2)
                with ck1: kf_start = st.slider("Intensità inizio", 0.0, 3.0, 0.5, 0.1, key="kf_start")
                with ck2: kf_end   = st.slider("Intensità fine",   0.0, 3.0, 1.5, 0.1, key="kf_end")
                import pandas as pd
                cap_tmp = cv2.VideoCapture(video_path)
                fps_tmp  = cap_tmp.get(cv2.CAP_PROP_FPS) or 24
                frames_tmp = cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap_tmp.release()
                dur_est = round(frames_tmp / fps_tmp, 1) if fps_tmp > 0 else 10.0
                kf_df = pd.DataFrame({"Secondo": [0.0, dur_est], "Intensita'": [kf_start, kf_end]})

        # ── PROPORZIONI EXPORT ───────────────────────────────────
        aspect_ratio = st.radio("📐 Formato output:",
            ["Originale","16:9","9:16","1:1"], horizontal=True)

        # ── LIMITE FRAME ─────────────────────────────────────────
        max_frames = st.number_input("🎬 Limite frame (0=nessun limite)",
            min_value=0, max_value=10000, value=0)

        st.markdown("---")

        # ── MASCHERA EFFETTO (multi) ──────────────────────────────
        st.markdown("**🎭 Maschere effetto**")

        # Inizializza lista maschere in session_state
        if 'masks_list' not in st.session_state:
            st.session_state.masks_list = []

        # Bottone aggiungi maschera
        col_ma, col_mb = st.columns([1, 1])
        with col_ma:
            if st.button("➕ Aggiungi maschera", use_container_width=True):
                st.session_state.masks_list.append({
                    'mask_type': 'striscia_h',
                    'mask_x': 0.5, 'mask_y': 0.5,
                    'mask_w': 1.0, 'mask_h': 0.25,
                    'mask_feather': 0, 'mask_reverse': False,
                    'animate_pos': False,
                })
        with col_mb:
            if st.button("🗑️ Rimuovi tutte", use_container_width=True):
                st.session_state.masks_list = []

        # Render di ogni maschera
        masks_to_remove = []
        for mi, msk in enumerate(st.session_state.masks_list):
            with st.expander(f"Maschera {mi+1} — {msk['mask_type']}", expanded=True):
                msk['mask_type'] = st.selectbox(
                    "Forma", ["striscia_h","striscia_v","rettangolo","cerchio"],
                    format_func=lambda x: {
                        "striscia_h": "➖ Striscia H",
                        "striscia_v": "➕ Striscia V",
                        "rettangolo": "▭ Rettangolo",
                        "cerchio":    "⭕ Cerchio",
                    }[x],
                    index=["striscia_h","striscia_v","rettangolo","cerchio"].index(msk['mask_type']),
                    key=f"mtype_{mi}"
                )
                if msk['mask_type'] == 'striscia_h':
                    msk['mask_y'] = st.slider("Pos. verticale", 0.0, 1.0, msk['mask_y'], 0.01, key=f"my_{mi}")
                    msk['mask_h'] = st.slider("Altezza", 0.01, 1.0, msk['mask_h'], 0.01, key=f"mh_{mi}")
                    msk['mask_w'] = 1.0
                elif msk['mask_type'] == 'striscia_v':
                    msk['mask_x'] = st.slider("Pos. orizzontale", 0.0, 1.0, msk['mask_x'], 0.01, key=f"mx_{mi}")
                    msk['mask_w'] = st.slider("Larghezza", 0.01, 1.0, msk['mask_w'], 0.01, key=f"mw_{mi}")
                    msk['mask_h'] = 1.0
                elif msk['mask_type'] in ('rettangolo','cerchio'):
                    cm1, cm2 = st.columns(2)
                    with cm1: msk['mask_x'] = st.slider("Centro X", 0.0, 1.0, msk['mask_x'], 0.01, key=f"mx_{mi}")
                    with cm2: msk['mask_y'] = st.slider("Centro Y", 0.0, 1.0, msk['mask_y'], 0.01, key=f"my_{mi}")
                    cm3, cm4 = st.columns(2)
                    with cm3: msk['mask_w'] = st.slider("Larghezza", 0.01, 1.0, msk['mask_w'], 0.01, key=f"mw_{mi}")
                    with cm4: msk['mask_h'] = st.slider("Altezza", 0.01, 1.0, msk['mask_h'], 0.01, key=f"mhh_{mi}")
                msk['mask_feather'] = st.slider("Sfumatura bordi (px)", 0, 60, msk['mask_feather'], 2, key=f"mf_{mi}")
                msk['mask_reverse'] = st.checkbox("🔄 Inverti", msk['mask_reverse'], key=f"mr_{mi}")

                # ── ANIMAZIONE KEYFRAME MASCHERA ─────────────────────────
                msk['animate_pos'] = st.toggle(
                    "🎞️ Anima posizione/dimensione",
                    value=msk.get('animate_pos', False),
                    key=f"manim_{mi}",
                    help="Interpola linearmente posizione e dimensione dall'inizio alla fine del video."
                )
                if msk['animate_pos']:
                    st.caption("**Inizio video → Fine video**")
                    mt = msk['mask_type']
                    if mt == 'striscia_h':
                        ka1, ka2 = st.columns(2)
                        with ka1:
                            msk['mask_y_start'] = st.slider(
                                "Pos. Y — inizio", 0.0, 1.0,
                                float(msk.get('mask_y_start', msk.get('mask_y', 0.5))),
                                0.01, key=f"mys_{mi}")
                            msk['mask_h_start'] = st.slider(
                                "Altezza — inizio", 0.01, 1.0,
                                float(msk.get('mask_h_start', msk.get('mask_h', 0.25))),
                                0.01, key=f"mhs_{mi}")
                        with ka2:
                            msk['mask_y_end'] = st.slider(
                                "Pos. Y — fine", 0.0, 1.0,
                                float(msk.get('mask_y_end', msk.get('mask_y', 0.5))),
                                0.01, key=f"mye_{mi}")
                            msk['mask_h_end'] = st.slider(
                                "Altezza — fine", 0.01, 1.0,
                                float(msk.get('mask_h_end', msk.get('mask_h', 0.25))),
                                0.01, key=f"mhe_{mi}")
                    elif mt == 'striscia_v':
                        ka1, ka2 = st.columns(2)
                        with ka1:
                            msk['mask_x_start'] = st.slider(
                                "Pos. X — inizio", 0.0, 1.0,
                                float(msk.get('mask_x_start', msk.get('mask_x', 0.5))),
                                0.01, key=f"mxs_{mi}")
                            msk['mask_w_start'] = st.slider(
                                "Largh. — inizio", 0.01, 1.0,
                                float(msk.get('mask_w_start', msk.get('mask_w', 0.25))),
                                0.01, key=f"mws_{mi}")
                        with ka2:
                            msk['mask_x_end'] = st.slider(
                                "Pos. X — fine", 0.0, 1.0,
                                float(msk.get('mask_x_end', msk.get('mask_x', 0.5))),
                                0.01, key=f"mxe_{mi}")
                            msk['mask_w_end'] = st.slider(
                                "Largh. — fine", 0.01, 1.0,
                                float(msk.get('mask_w_end', msk.get('mask_w', 0.25))),
                                0.01, key=f"mwe_{mi}")
                    elif mt in ('rettangolo', 'cerchio'):
                        ka1, ka2 = st.columns(2)
                        with ka1:
                            st.caption("Inizio")
                            msk['mask_x_start'] = st.slider(
                                "Centro X — inizio", 0.0, 1.0,
                                float(msk.get('mask_x_start', msk.get('mask_x', 0.5))),
                                0.01, key=f"mxs_{mi}")
                            msk['mask_y_start'] = st.slider(
                                "Centro Y — inizio", 0.0, 1.0,
                                float(msk.get('mask_y_start', msk.get('mask_y', 0.5))),
                                0.01, key=f"mys_{mi}")
                            msk['mask_w_start'] = st.slider(
                                "Largh. — inizio", 0.01, 1.0,
                                float(msk.get('mask_w_start', msk.get('mask_w', 0.5))),
                                0.01, key=f"mws_{mi}")
                            msk['mask_h_start'] = st.slider(
                                "Altezza — inizio", 0.01, 1.0,
                                float(msk.get('mask_h_start', msk.get('mask_h', 0.5))),
                                0.01, key=f"mhs_{mi}")
                        with ka2:
                            st.caption("Fine")
                            msk['mask_x_end'] = st.slider(
                                "Centro X — fine", 0.0, 1.0,
                                float(msk.get('mask_x_end', msk.get('mask_x', 0.5))),
                                0.01, key=f"mxe_{mi}")
                            msk['mask_y_end'] = st.slider(
                                "Centro Y — fine", 0.0, 1.0,
                                float(msk.get('mask_y_end', msk.get('mask_y', 0.5))),
                                0.01, key=f"mye_{mi}")
                            msk['mask_w_end'] = st.slider(
                                "Largh. — fine", 0.01, 1.0,
                                float(msk.get('mask_w_end', msk.get('mask_w', 0.5))),
                                0.01, key=f"mwe_{mi}")
                            msk['mask_h_end'] = st.slider(
                                "Altezza — fine", 0.01, 1.0,
                                float(msk.get('mask_h_end', msk.get('mask_h', 0.5))),
                                0.01, key=f"mhe_{mi}")
                if st.button(f"🗑️ Rimuovi maschera {mi+1}", key=f"mdel_{mi}"):
                    masks_to_remove.append(mi)

        for idx in reversed(masks_to_remove):
            st.session_state.masks_list.pop(idx)

        # Variabili legacy per retro-compatibilità (anteprima live)
        masks_list = st.session_state.masks_list
        # fallback singola maschera per anteprima
        if masks_list:
            _m0 = masks_list[0]
            mask_type    = _m0['mask_type']
            mask_x       = _m0['mask_x']
            mask_y       = _m0['mask_y']
            mask_w       = _m0['mask_w']
            mask_h       = _m0['mask_h']
            mask_feather = _m0['mask_feather']
            mask_reverse = _m0['mask_reverse']
        else:
            mask_type    = 'nessuna'
            mask_x, mask_y = 0.5, 0.5
            mask_w, mask_h = 1.0, 0.3
            mask_feather = 0
            mask_reverse = False

        # ── BOTTONE PROCESSA ─────────────────────────────────────
        do_process = st.button("🚀 Processa Video", use_container_width=True)

    # ── COLONNA DESTRA: ANTEPRIMA LIVE ──────────────────────────
    with col_prev:
        st.caption("🖼️ Anteprima live")
        try:
            cap_live = cv2.VideoCapture(video_path)
            total_f_live = int(cap_live.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_live.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_f_live // 3))
            ret_live, frame_live = cap_live.read()
            cap_live.release()
            if ret_live:
                if effect_type == 'pixel_sort':
                    pf = glitch_pixel_sort(frame_live, *params)
                elif effect_type == 'channel_shift':
                    pf = glitch_channel_shift(frame_live, *params)
                elif effect_type == 'datamosh':
                    pf = glitch_datamosh(frame_live, frame_live, *params)
                elif effect_type == 'byte_corrupt':
                    pf = glitch_byte_corrupt(frame_live, *params)
                elif effect_type == 'slice_shift':
                    pf = glitch_slice_shift(frame_live, *params)
                elif effect_type == 'slit_scan':
                    pf = glitch_slit_scan(frame_live, [frame_live], *params)
                elif effect_type == 'thermal':
                    pf = glitch_thermal(frame_live, *params)
                elif effect_type == 'ascii_glitch':
                    pf = glitch_ascii_glitch(frame_live, *params)
                elif effect_type == 'halftone':
                    pf = glitch_halftone(frame_live, *params)
                elif effect_type == 'chroma_pulse':
                    pf = glitch_chroma_pulse(frame_live, *params, _frame_idx=0)
                elif effect_type == 'moire':
                    pf = glitch_moire(frame_live, *params)
                elif effect_type == 'feedback_loop':
                    pf = glitch_feedback_loop(frame_live, frame_live, *params)
                elif effect_type == 'pixel_drift':
                    pf = glitch_pixel_drift(frame_live, *params)
                elif effect_type == 'echo_smear':
                    pf = glitch_echo_smear(frame_live, frame_live, *params)
                elif effect_type == 'rgb_wave':
                    pf = glitch_rgb_wave(frame_live, *params)
                elif effect_type == 'mirror_blocks':
                    pf = glitch_mirror_blocks(frame_live, *params)
                elif effect_type == 'color_quantize':
                    pf = glitch_color_quantize(frame_live, *params)
                elif effect_type == 'vhs':
                    pf = glitch_vhs_frame(frame_live, *params)
                elif effect_type == 'distruttivo':
                    pf = glitch_distruttivo_frame(frame_live, *params)
                elif effect_type == 'noise':
                    pf = glitch_noise_frame(frame_live, *params)
                elif effect_type == 'broken_tv':
                    pf = glitch_broken_tv_frame(frame_live, *params)
                elif effect_type == 'combined':
                    pf = frame_live.copy()
                    if params.get("apply_vhs"):
                        pf = glitch_vhs_frame(pf, params.get("vhs_intensity",1.0), params.get("vhs_scanline_freq",1.0), params.get("vhs_color_shift",1.0))
                    if params.get("apply_distruttivo"):
                        pf = glitch_distruttivo_frame(pf, params.get("dest_block_size",1.0), params.get("dest_num_blocks",1.0), params.get("dest_displacement",1.0))
                    if params.get("apply_noise"):
                        pf = glitch_noise_frame(pf, params.get("noise_intensity",1.0), params.get("noise_coverage",1.0), params.get("noise_chaos",1.0))
                    if params.get("apply_broken_tv"):
                        pf = glitch_broken_tv_frame(pf, params.get("tv_shift_intensity",1.0), params.get("tv_line_height",1.0), params.get("tv_flicker_prob",1.0))
                elif effect_type == 'random':
                    rl = params[0] if params else 1.0
                    chosen = random.choice(['pixel_sort','channel_shift','byte_corrupt','slice_shift','vhs','broken_tv','noise','distruttivo'])
                    rp = tuple(random.uniform(0.5,1.5)*rl for _ in range(3))
                    _fn = {'pixel_sort':glitch_pixel_sort,'channel_shift':glitch_channel_shift,'byte_corrupt':glitch_byte_corrupt,'slice_shift':glitch_slice_shift,'vhs':glitch_vhs_frame,'broken_tv':glitch_broken_tv_frame,'noise':glitch_noise_frame,'distruttivo':glitch_distruttivo_frame}
                    pf = _fn[chosen](frame_live, *rp)
                else:
                    pf = frame_live
                # Applica maschera anche in anteprima
                if masks_list:
                    h_lv, w_lv = frame_live.shape[:2]
                    prev_mask = build_combined_mask(h_lv, w_lv, masks_list)
                    pf = apply_mask_blend(frame_live, pf, prev_mask)
                elif mask_type != 'nessuna':
                    h_lv, w_lv = frame_live.shape[:2]
                    prev_mask = build_mask(h_lv, w_lv, mask_type, mask_x, mask_y,
                                          mask_w, mask_h, mask_feather, mask_reverse)
                    pf = apply_mask_blend(frame_live, pf, prev_mask)
                st.image(cv2.cvtColor(pf, cv2.COLOR_BGR2RGB), use_container_width=True)
        except Exception as e:
            st.caption(f"Anteprima non disponibile: {e}")

    # ── PROCESSING (fuori dalle colonne) ────────────────────────
    if do_process:
        with st.spinner("🔥 Processando il video..."):
            # Calcola envelope keyframe
            kf_envelope = None
            if use_keyframes and kf_df is not None and len(kf_df) >= 2:
                cap_info = cv2.VideoCapture(video_path)
                fps_info   = int(cap_info.get(cv2.CAP_PROP_FPS)) or 24
                frames_info= int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_info.release()
                if max_frames > 0:
                    frames_info = min(frames_info, max_frames)
                kf_envelope = interpolate_keyframes(kf_df, fps_info, frames_info)

            # Analisi audio (serve sia per audio_reactive che per beat_sync)
            audio_env = None
            if use_audio_reactive or beat_sync or use_temporal_crossfade:
                with st.spinner("🎵 Analisi audio..."):
                    _tmp_wav = extract_audio(video_path, silent=True)
                    if _tmp_wav:
                        cap_ar = cv2.VideoCapture(video_path)
                        _fps_ar = cap_ar.get(cv2.CAP_PROP_FPS) or 24
                        _tot_ar = int(cap_ar.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap_ar.release()
                        audio_env = analyze_audio_for_video(_tmp_wav, _fps_ar, _tot_ar)
                        try: os.unlink(_tmp_wav)
                        except: pass
                    else:
                        st.warning("⚠️ Nessuna traccia audio trovata nel video — Audio Reactive/Beat Sync/Crossfade disabilitati.")

            # Gestione sorgente audio esterna
            audio_source_path = None
            if audio_mode in ("3_esterno","4_esterno_distruggi") and uploaded_audio_inline is not None:
                ext = os.path.splitext(uploaded_audio_inline.name)[1].lower()
                fd_as, audio_source_path = tempfile.mkstemp(suffix=ext)
                os.close(fd_as)
                with open(audio_source_path,'wb') as f:
                    f.write(uploaded_audio_inline.read())

            # Mappa le nuove modalità audio alle chiavi attese da process_video
            _mode_map = {
                "0_originale":        "0_originale",
                "2_distruggi":        "2_distruggi",
                "1_mix":              "2_distruggi",   # mix: distruggi poi mixiamo sotto
                "3_esterno":          "1_carica",            # audio esterno pulito, NON glitchato
                "4_esterno_distruggi":"1_carica_distruggi",  # audio esterno + glitch
            }
            _process_audio_mode = _mode_map.get(audio_mode, "0_originale")

            result_path = process_video(video_path, effect_type, params, max_frames,
                                        _process_audio_mode, kf_envelope, audio_params_override,
                                        aspect_ratio, audio_source_path,
                                        audio_env, ar_intensity,
                                        mask_type, mask_x, mask_y, mask_w, mask_h,
                                        mask_feather, mask_reverse,
                                        beat_sync, beat_decay, beat_intensity,
                                        masks_list=masks_list,
                                        temporal_crossfade=use_temporal_crossfade, tc_source=tc_source,
                                        tc_alpha_min=tc_alpha_min, tc_alpha_max=tc_alpha_max, tc_smooth=tc_smooth)

            if result_path:
                st.success("✅ Video processato!")

                # Mix post-processing: se modalità 1_mix, combina video glitch + audio originale a 50%
                if audio_mode == "1_mix" and ffmpeg_available:
                    fd_mix, mix_path = tempfile.mkstemp(suffix='_mix.mp4')
                    os.close(fd_mix)
                    # Combina audio originale (volume 0.5) + audio glitchato (volume 0.5)
                    r_mix = subprocess.run([
                        'ffmpeg', '-i', result_path, '-i', video_path,
                        '-filter_complex',
                        '[0:a]volume=0.5[a0];[1:a]volume=0.5[a1];[a0][a1]amix=inputs=2:duration=shortest[aout]',
                        '-map','0:v:0','-map','[aout]',
                        '-c:v','copy','-c:a','aac',
                        mix_path,'-y'
                    ], capture_output=True, text=True)
                    if r_mix.returncode == 0:
                        try: os.unlink(result_path)
                        except: pass
                        result_path = mix_path

                with st.spinner("🗜️ H.264..."):
                    h264_path = recompress_h264(result_path, aspect_ratio)

                orig_size  = get_file_size_mb(video_path)
                fps_v, w_v, h_v, frames_v, dur_v = get_video_info(video_path)
                out_size   = get_file_size_mb(h264_path)

                video_stem   = os.path.splitext(uploaded_file.name)[0]
                effect_label = {
                    "pixel_sort":"PixelSort","channel_shift":"ChannelShift",
                    "datamosh":"Datamosh","byte_corrupt":"ByteCorrupt",
                    "slice_shift":"SliceShift","echo_smear":"EchoSmear",
                    "rgb_wave":"RGBWave","mirror_blocks":"MirrorBlocks",
                    "color_quantize":"ColorQuantize","moire":"Moire",
                    "feedback_loop":"FeedbackLoop","pixel_drift":"PixelDrift",
                    "slit_scan":"SlitScan","thermal":"Thermal",
                    "ascii_glitch":"AsciiGlitch","halftone":"Halftone",
                    "chroma_pulse":"ChromaPulse",
                    "vhs":"VHS","distruttivo":"Distruttivo","noise":"Noise",
                    "combined":"Combinato","broken_tv":"BrokenTV","random":"Random"
                }.get(effect_type, effect_type)
                output_video_name = f"{video_stem}_{effect_label}.mp4"
                report_name       = f"{video_stem}_{effect_label}.txt"

                _tc_info = None
                if use_temporal_crossfade:
                    _tc_info = (f"{tc_source} | range {tc_alpha_min}-{tc_alpha_max} | "
                                f"smooth {tc_smooth}")

                st.session_state.report_data = build_report(
                    uploaded_file.name, orig_size, out_size,
                    fps_v, w_v, h_v, frames_v, dur_v,
                    effect_type, params, audio_mode != "0_originale", kf_df,
                    tc_info=_tc_info
                )
                st.session_state.h264_path         = h264_path
                st.session_state.output_video_name = output_video_name
                st.session_state.orig_filename     = uploaded_file.name
                st.session_state.report_filename   = report_name
                st.session_state.video_ready       = True

                for p in [result_path, video_path]:
                    try: os.unlink(p)
                    except: pass
            else:
                st.error("❌ Errore durante il processing.")

else:
    st.info("👆 Carica un video per iniziare!")

# RISULTATI PERSISTENTI
if st.session_state.video_ready:
    st.markdown("---")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        if st.session_state.h264_path and os.path.exists(st.session_state.h264_path):
            with open(st.session_state.h264_path,'rb') as vf:
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
            key="down_r"
        )
    st.text_area("📄 REPORT", st.session_state.report_data, height=320)

# Footer
st.markdown("---")
st.markdown("🎬 **VideoDistruktor by loop507** - Trasforma i tuoi video in opere d'arte glitched!")
if not check_ffmpeg():
    st.markdown("⚠️ *Per abilitare gli effetti audio, installa FFmpeg sul sistema*")
