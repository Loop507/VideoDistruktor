import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
import random
import io
import math # Importa math per i calcoli degli effetti audio

# Importa OpenCV con gestione errori
try:
    import cv2
    OPENCV_AVAILABLE = True
    st.success("✅ OpenCV caricato correttamente!")
except ImportError as e:
    st.error(f"❌ Errore: OpenCV non trovato. Assicurati di averlo installato con 'pip install opencv-python'. Dettagli: {e}")
    st.stop() # Ferma l'esecuzione se OpenCV non è disponibile

# Importa MoviePy per la gestione audio/video
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips, vfx
    MOVIEPY_AVAILABLE = True
    st.success("✅ MoviePy caricato correttamente!")
except ImportError as e:
    st.error(f"❌ Errore: MoviePy non trovato. Assicurati di averlo installato con 'pip install moviepy'. Dettagli: {e}")
    st.warning("Gli effetti audio non saranno disponibili.")
    MOVIEPY_AVAILABLE = False


# Configurazione della pagina
st.set_page_config(page_title="VideoDistruktor - Video Edition", layout="centered")

st.title("🎬🔥 VideoDistruktor")
st.write("Carica un video e genera versioni glitchate: VHS, Distruttivo, Noise, Combinato o Random, anche con audio glitchato!")

# File uploader per video
uploaded_file = st.file_uploader("📁 Carica un video", type=["mp4", "avi", "mov", "mkv"])

def frame_to_pil(frame):
    """Converte frame OpenCV (BGR) in PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_frame(pil_img):
    """Converte PIL Image (RGB) in frame OpenCV (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Funzioni degli effetti video (invariate) ---
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

# --- Nuove Funzioni per effetti audio ---
def glitch_audio(audio_clip, effect_type, params, duration_limit=None):
    """
    Applica effetti glitch all'audio clip.
    'effect_type' può essere 'vhs', 'distruttivo', 'noise', 'combined', 'random'.
    'params' conterrà i parametri specifici per l'effetto (o global_intensity per 'combined').
    """
    if not MOVIEPY_AVAILABLE:
        st.warning("MoviePy non disponibile, l'audio non verrà processato.")
        return audio_clip # Restituisce l'audio originale

    try:
        if duration_limit and audio_clip.duration > duration_limit:
            audio_clip = audio_clip.subclip(0, duration_limit)
        
        glitched_audio = audio_clip # Inizia con l'audio originale

        if effect_type == 'vhs':
            intensity = params.get("intensity", 1.0)
            # Aggiunge leggero rumore bianco e modula il volume
            noise_amplitude = 0.05 * intensity
            if noise_amplitude > 0:
                noise_clip = AudioFileClip(io.BytesIO(b'\x00' * int(audio_clip.duration * 44100 * 2)), fps=44100).audio_loop(audio_clip.duration).set_volume(noise_amplitude)
                glitched_audio = CompositeAudioClip([glitched_audio, noise_clip])
            # Piccole fluttuazioni di pitch o velocità
            if random.random() < 0.5 * intensity:
                glitched_audio = glitched_audio.fx(vfx.speedx, factor=random.uniform(1.0 - 0.05 * intensity, 1.0 + 0.05 * intensity))
            
        elif effect_type == 'distruttivo':
            intensity = params.get("intensity", 1.0)
            # Ripetizioni e tagli casuali
            if intensity > 0:
                num_cuts = int(5 * intensity)
                for _ in range(num_cuts):
                    if glitched_audio.duration < 0.1: # Evita loop infiniti su clip troppo corte
                        break
                    start_cut = random.uniform(0, glitched_audio.duration - 0.05)
                    end_cut = min(glitched_audio.duration, start_cut + random.uniform(0.01, 0.1 * intensity))
                    if end_cut > start_cut:
                        # Rimuove piccoli segmenti o li ripete
                        if random.random() < 0.7: # Rimuove
                            glitched_audio = glitched_audio.fx(vfx.fadeout, duration=end_cut-start_cut, start_time=start_cut).fx(vfx.fadein, duration=end_cut-start_cut, end_time=end_cut)
                            # Questa non è una vera rimozione, ma un effetto glitchy di taglio
                        else: # Ripete
                            segment = glitched_audio.subclip(start_cut, end_cut)
                            glitched_audio = concatenate_audioclips([glitched_audio.subclip(0, start_cut), segment, segment, glitched_audio.subclip(end_cut)])
                            if glitched_audio.duration > audio_clip.duration * 1.5: # Evita che l'audio diventi troppo lungo
                                glitched_audio = glitched_audio.subclip(0, audio_clip.duration * 1.5)


        elif effect_type == 'noise':
            intensity = params.get("intensity", 1.0)
            # Aggiunge rumore più aggressivo e distorsione occasionale
            noise_amplitude = 0.1 * intensity
            if noise_amplitude > 0:
                 # Crea un clip audio vuoto della stessa durata e riempilo di rumore
                noise_array = (np.random.rand(int(audio_clip.duration * audio_clip.fps)) * 2 - 1) * noise_amplitude
                # Converti in clip audio di moviepy
                noise_clip = AudioFileClip(io.BytesIO(noise_array.tobytes()), fps=audio_clip.fps).set_duration(audio_clip.duration)
                glitched_audio = CompositeAudioClip([glitched_audio, noise_clip])
            
            # Saturazione/Distorsione casuale
            if random.random() < 0.3 * intensity:
                glitched_audio = glitched_audio.fx(vfx.volumex, random.uniform(0.5, 2.0)) # Variazione di volume glitchy

        elif effect_type == 'combined':
            global_intensity = params.get("global_intensity", 1.0)
            current_audio = glitched_audio

            # Mappa l'intensità globale ai parametri audio
            audio_vhs_intensity = 0.5 + 1.5 * global_intensity
            audio_dist_intensity = 0.5 + 1.5 * global_intensity
            audio_noise_intensity = 0.5 + 1.5 * global_intensity

            if params.get("apply_vhs"):
                # Applica effetto VHS all'audio
                noise_amplitude = 0.05 * audio_vhs_intensity
                if noise_amplitude > 0:
                    noise_clip = AudioFileClip(io.BytesIO(b'\x00' * int(current_audio.duration * 44100 * 2)), fps=44100).audio_loop(current_audio.duration).set_volume(noise_amplitude)
                    current_audio = CompositeAudioClip([current_audio, noise_clip])
                if random.random() < 0.5 * audio_vhs_intensity:
                    current_audio = current_audio.fx(vfx.speedx, factor=random.uniform(1.0 - 0.05 * audio_vhs_intensity, 1.0 + 0.05 * audio_vhs_intensity))

            if params.get("apply_distruttivo"):
                # Applica effetto Distruttivo all'audio
                num_cuts = int(3 * audio_dist_intensity) # Meno tagli per evitare distorsioni troppo estreme
                for _ in range(num_cuts):
                    if current_audio.duration < 0.1: break
                    start_cut = random.uniform(0, current_audio.duration - 0.05)
                    end_cut = min(current_audio.duration, start_cut + random.uniform(0.01, 0.05 * audio_dist_intensity))
                    if end_cut > start_cut and random.random() < 0.5: # 50% di chance di ripetere
                        segment = current_audio.subclip(start_cut, end_cut)
                        current_audio = concatenate_audioclips([current_audio.subclip(0, start_cut), segment, segment, current_audio.subclip(end_cut)])
                        if current_audio.duration > audio_clip.duration * 1.2: # Limita la crescita dell'audio
                            current_audio = current_audio.subclip(0, audio_clip.duration * 1.2)


            if params.get("apply_noise"):
                # Applica effetto Noise all'audio
                noise_amplitude = 0.1 * audio_noise_intensity
                if noise_amplitude > 0:
                    noise_array = (np.random.rand(int(current_audio.duration * current_audio.fps)) * 2 - 1) * noise_amplitude
                    noise_clip = AudioFileClip(io.BytesIO(noise_array.tobytes()), fps=current_audio.fps).set_duration(current_audio.duration)
                    current_audio = CompositeAudioClip([current_audio, noise_clip])
                if random.random() < 0.3 * audio_noise_intensity:
                    current_audio = current_audio.fx(vfx.volumex, random.uniform(0.5, 2.0))

            glitched_audio = current_audio

        elif effect_type == 'random':
            random_level = params.get("random_level", 1.0)
            # Effetti audio casuali
            if random_level > 0:
                choice = random.choice(['noise', 'speed', 'cut'])
                if choice == 'noise':
                    noise_amplitude = 0.1 * random_level
                    if noise_amplitude > 0:
                        noise_array = (np.random.rand(int(glitched_audio.duration * glitched_audio.fps)) * 2 - 1) * noise_amplitude
                        noise_clip = AudioFileClip(io.BytesIO(noise_array.tobytes()), fps=glitched_audio.fps).set_duration(glitched_audio.duration)
                        glitched_audio = CompositeAudioClip([glitched_audio, noise_clip])
                elif choice == 'speed':
                     glitched_audio = glitched_audio.fx(vfx.speedx, factor=random.uniform(1.0 - 0.1 * random_level, 1.0 + 0.1 * random_level))
                elif choice == 'cut':
                    if glitched_audio.duration > 0.1:
                        start_cut = random.uniform(0, glitched_audio.duration - 0.05)
                        end_cut = min(glitched_audio.duration, start_cut + random.uniform(0.01, 0.05 * random_level))
                        if end_cut > start_cut:
                            segment = glitched_audio.subclip(start_cut, end_cut)
                            glitched_audio = concatenate_audioclips([glitched_audio.subclip(0, start_cut), segment, glitched_audio.subclip(end_cut)])
                            if glitched_audio.duration > audio_clip.duration * 1.1:
                                glitched_audio = glitched_audio.subclip(0, audio_clip.duration * 1.1)


        return glitched_audio.set_duration(audio_clip.duration) # Assicura che la durata dell'audio non ecceda quella originale, troncando o allungando se necessario.
    
    except Exception as e:
        st.warning(f"⚠️ Errore durante l'applicazione dell'effetto audio: {e}. L'audio non verrà modificato.")
        return audio_clip # Restituisce l'audio originale in caso di errore

def process_video(video_path, effect_type, params, max_frames=None, apply_audio_glitch=False):
    """Processa il video con l'effetto scelto, ora gestisce anche l'audio."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il video. Potrebbe essere danneggiato o non supportato.")
        return None, None # Restituisce due None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    if max_frames and max_frames > 0:
        actual_total_frames = min(total_frames, max_frames)
        duration_to_process = actual_total_frames / fps if fps > 0 else 0
    else:
        actual_total_frames = total_frames
        duration_to_process = duration_sec

    # Percorsi temporanei per video processato e audio (se presente)
    output_video_path = tempfile.mktemp(suffix='.mp4')
    output_audio_path = None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec per MP4
    
    try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error("❌ Impossibile inizializzare VideoWriter. Controlla i codec o i permessi di scrittura.")
            cap.release()
            return None, None
        
        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
                elif effect_type == 'combined':
                    current_frame_to_process = frame
                    global_intensity = params.get("global_intensity", 1.0)

                    vhs_intensity = 0.5 + 1.5 * global_intensity
                    vhs_scanlines = 0.5 + 1.5 * global_intensity
                    vhs_colors = 0.5 + 1.5 * global_intensity

                    dest_block_size = 0.5 + 1.5 * global_intensity
                    dest_num_blocks = 0.5 + 1.5 * global_intensity
                    dest_displacement = 0.5 + 1.5 * global_intensity

                    noise_intensity_val = 0.5 + 1.5 * global_intensity
                    noise_coverage_val = 0.5 + 1.5 * global_intensity
                    noise_chaos_val = min(1.0, 0.2 + 0.8 * global_intensity)

                    if params.get("apply_vhs"):
                        current_frame_to_process = glitch_vhs_frame(current_frame_to_process, vhs_intensity, vhs_scanlines, vhs_colors)
                    
                    if params.get("apply_distruttivo"):
                        current_frame_to_process = glitch_distruttivo_frame(current_frame_to_process, dest_block_size, dest_num_blocks, dest_displacement)
                    
                    if params.get("apply_noise"):
                        current_frame_to_process = glitch_noise_frame(current_frame_to_process, noise_intensity_val, noise_coverage_val, noise_chaos_val)
                    
                    processed_frame = current_frame_to_process

                elif effect_type == 'random':
                    effects = [
                        (glitch_vhs_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
                        (glitch_distruttivo_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
                        (glitch_noise_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))
                    ]
                    chosen_effect, p1, p2, p3 = random.choice(effects)
                    processed_frame = chosen_effect(frame, p1 * params[0], p2 * params[0], p3 * params[0])
                
                else:
                    st.warning(f"Tipo di effetto '{effect_type}' non riconosciuto.")
                    processed_frame = frame
            except Exception as e:
                st.warning(f"⚠️ Errore durante l'applicazione dell'effetto al frame {frame_count}: {e}. Skipping frame.")
                processed_frame = frame
            
            out.write(processed_frame)
            
            frame_count += 1
            progress = frame_count / actual_total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processando frame {frame_count}/{actual_total_frames}...")
        
    except Exception as e:
        st.error(f"Errore critico durante il processing del video: {e}")
        output_video_path = None
    finally:
        cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        progress_bar.empty()
        status_text.empty()
    
    # --- Gestione e Glitch dell'Audio ---
    if output_video_path and apply_audio_glitch and MOVIEPY_AVAILABLE:
        try:
            st.info("🎵 Elaborazione audio in corso...")
            video_clip = VideoFileClip(video_path)
            if video_clip.audio:
                original_audio = video_clip.audio
                
                # Applica glitch audio in base all'effetto video scelto
                audio_glitch_params = {}
                if effect_type == 'vhs':
                    audio_glitch_params["intensity"] = params[0] if params else 1.0 # Usa VHS intensity
                elif effect_type == 'distruttivo':
                    audio_glitch_params["intensity"] = params[0] if params else 1.0 # Usa Distruttivo block size
                elif effect_type == 'noise':
                    audio_glitch_params["intensity"] = params[0] if params else 1.0 # Usa Noise intensity
                elif effect_type == 'combined':
                    audio_glitch_params = params # Passa l'intero dizionario dei parametri combinati
                    audio_glitch_params["global_intensity"] = params.get("global_intensity", 1.0) # Assicurati che l'intensità globale sia passata

                elif effect_type == 'random':
                     audio_glitch_params["random_level"] = params[0] if params else 1.0


                glitched_audio_clip = glitch_audio(original_audio, effect_type, audio_glitch_params, duration_limit=duration_to_process)
                
                output_audio_path = tempfile.mktemp(suffix='.mp3') # Salva audio come mp3
                glitched_audio_clip.write_audiofile(output_audio_path, logger=None) # logger=None per ridurre output in console
                st.success("✅ Audio elaborato!")
            else:
                st.info("Video non contiene traccia audio.")
                video_clip.close() # Chiudi il clip video
        except Exception as e:
            st.error(f"❌ Errore durante l'elaborazione dell'audio: {e}. Il video finale non avrà audio glitchato.")
            output_audio_path = None
    elif apply_audio_glitch and not MOVIEPY_AVAILABLE:
        st.warning("Skipping audio glitch: MoviePy non disponibile.")

    # Combina video e audio (se l'audio è stato processato con successo)
    final_output_path = tempfile.mktemp(suffix='.mp4')
    if output_video_path and os.path.exists(output_video_path):
        if output_audio_path and os.path.exists(output_audio_path):
            try:
                st.info("🔗 Combinando video e audio...")
                video_clip_final = VideoFileClip(output_video_path)
                audio_clip_final = AudioFileClip(output_audio_path)
                
                # Imposta la durata dell'audio per corrispondere al video, se necessario
                if audio_clip_final.duration > video_clip_final.duration:
                    audio_clip_final = audio_clip_final.subclip(0, video_clip_final.duration)
                elif video_clip_final.duration > audio_clip_final.duration:
                    # Se il video è più lungo, allunga l'audio o lascia silenzio
                    pass # Lasciamo l'audio così com'è, si fermerà prima del video
                
                final_clip = video_clip_final.set_audio(audio_clip_final)
                final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", logger=None)
                
                video_clip_final.close()
                audio_clip_final.close()
                st.success("✅ Video e audio combinati con successo!")
                return final_output_path # Restituisce solo il percorso finale
            except Exception as e:
                st.error(f"❌ Errore durante la combinazione di video e audio: {e}. Verrà fornito solo il video.")
                # Elimina i file temporanei audio se la combinazione fallisce
                if output_audio_path and os.path.exists(output_audio_path):
                    os.unlink(output_audio_path)
                return output_video_path # Restituisce solo il video senza audio
        else:
            return output_video_path # Nessun audio da combinare, restituisce solo il video
    
    return None # Nulla è stato generato

# Logica principale dell'applicazione Streamlit
if uploaded_file is not None:
    video_path = None
    final_output_video_path = None # Variabile per il percorso finale
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Impossibile leggere il video caricato. Assicurati che non sia corrotto.")
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            st.stop()
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        st.success(f"✅ Video caricato!")
        st.info(f"📊 **Dettagli video:** {width}x{height} - {fps} FPS - {duration:.1f}s - {total_frames} frames")
        
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.image(first_frame_rgb, caption="🎬 Primo frame del video", use_container_width=True)
        
        st.markdown("### ⚙️ Impostazioni Processing")
        max_frames_default = min(300, total_frames)
        max_frames = st.number_input(
            "🎯 Massimo numero di frames da processare (0 = tutti)", 
            min_value=0, 
            max_value=total_frames, 
            value=max_frames_default,
            help="Limita il numero di frames per ridurre i tempi di processing. Un valore più basso rende il processo più veloce."
        )
        
        if max_frames == 0:
            max_frames_to_process = total_frames
        else:
            max_frames_to_process = max_frames
        
        processing_duration = (max_frames_to_process / fps) if fps > 0 else 0
        st.info(f"📅 Verranno processati {max_frames_to_process} frames ({processing_duration:.1f}s di video)")
        
        # Checkbox per il glitch audio
        apply_audio_glitch_checkbox = st.checkbox("🎵 Applica glitch all'audio", value=True, help="Attiva/disattiva gli effetti glitch anche sulla traccia audio.")

        # --- CONTROLLI EFFETTI: Reintrodotti controlli singoli + Combinato + Random ---
        st.markdown("### 🎛️ Controlli Effetti")
        
        # Tabs per i diversi effetti
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📺 VHS", "💥 Distruttivo", "🌀 Noise", "✨ Glitch Combinato", "🎲 Random"])
        
        with tab1:
            st.markdown("**Effetto VHS Glitch**")
            col1, col2, col3 = st.columns(3)
            with col1:
                vhs_intensity = st.slider("Intensità Distorsione", 0.0, 2.0, 1.0, 0.1, key="vhs_int_v2_1")
            with col2:
                vhs_scanlines = st.slider("Frequenza Scanlines", 0.0, 2.0, 1.0, 0.1, key="vhs_scan_v2_1")
            with col3:
                vhs_colors = st.slider("Separazione Colori", 0.0, 2.0, 1.0, 0.1, key="vhs_col_v2_1")
            
            if st.button("🎬 Genera Video VHS", key="btn_vhs_v2_1"):
                with st.spinner("📺 Processando video con effetto VHS... Questo potrebbe richiedere tempo."):
                    # params qui sono una tupla per gli effetti singoli
                    final_output_video_path = process_video(video_path, 'vhs', (vhs_intensity, vhs_scanlines, vhs_colors), max_frames_to_process, apply_audio_glitch_checkbox)
        
        with tab2:
            st.markdown("**Effetto Distruttivo**")
            col1, col2, col3 = st.columns(3)
            with col1:
                dest_blocks = st.slider("Dimensione Blocchi", 0.0, 2.0, 1.0, 0.1, key="dest_size_v2_1")
            with col2:
                dest_number = st.slider("Numero Blocchi", 0.0, 2.0, 1.0, 0.1, key="dest_num_v2_1")
            with col3:
                dest_displacement = st.slider("Spostamento", 0.0, 2.0, 1.0, 0.1, key="dest_disp_v2_1")
            
            if st.button("🎬 Genera Video Distruttivo", key="btn_dest_v2_1"):
                with st.spinner("💥 Processando video con effetto Distruttivo... Questo potrebbe richiedere tempo."):
                    final_output_video_path = process_video(video_path, 'distruttivo', (dest_blocks, dest_number, dest_displacement), max_frames_to_process, apply_audio_glitch_checkbox)
        
        with tab3:
            st.markdown("**Effetto Noise**")
            col1, col2, col3 = st.columns(3)
            with col1:
                noise_intensity = st.slider("Intensità Rumore", 0.0, 2.0, 1.0, 0.1, key="noise_int_v2_1")
            with col2:
                noise_coverage = st.slider("Copertura", 0.0, 2.0, 1.0, 0.1, key="noise_cov_v2_1")
            with col3:
                noise_chaos = st.slider("Caos", 0.0, 1.0, 0.5, 0.1, key="noise_chaos_v2_1")
            
            if st.button("🎬 Genera Video Noise", key="btn_noise_v2_1"):
                with st.spinner("🌀 Processando video con effetto Noise... Questo potrebbe richiedere tempo."):
                    final_output_video_path = process_video(video_path, 'noise', (noise_intensity, noise_coverage, noise_chaos), max_frames_to_process, apply_audio_glitch_checkbox)
        
        with tab4: # Nuova tab Glitch Combinato
            st.markdown("**Configura il tuo Glitch Combinato**")
            
            apply_vhs = st.checkbox("📺 Applica effetto VHS", value=True, key="chk_vhs_1")
            apply_distruttivo = st.checkbox("💥 Applica effetto Distruttivo", value=True, key="chk_dist_1")
            apply_noise = st.checkbox("🌀 Applica effetto Noise", value=True, key="chk_noise_1")
            
            global_intensity_combined = st.slider("Livello di Intensità Globale", 0.0, 2.0, 1.0, 0.1, 
                                                  help="Controlla l'intensità di tutti gli effetti attivi. 0.0 = minima, 2.0 = massima.",
                                                  key="global_intensity_combined_1")
            
            if st.button("🎬 Genera Video Combinato", key="btn_combined_1"):
                combined_params = {
                    "apply_vhs": apply_vhs,
                    "apply_distruttivo": apply_distruttivo,
                    "apply_noise": apply_noise,
                    "global_intensity": global_intensity_combined
                }
                
                with st.spinner("✨ Processando video con effetti combinati... Questo potrebbe richiedere tempo."):
                    final_output_video_path = process_video(video_path, 'combined', combined_params, max_frames_to_process, apply_audio_glitch_checkbox)
        
        with tab5: # Tab Random
            st.markdown("**Effetto Random (Combo Casuale)**")
            random_level = st.slider("Livello Casualità", 0.0, 2.0, 1.0, 0.1, key="random_lev_v2_1")
            st.info("🎲 Ogni frame avrà un effetto casuale diverso con intensità proporzionale al Livello Casualità!")
            
            if st.button("🎬 Genera Video Random", key="btn_random_v2_1"):
                with st.spinner("🎲 Processando video con effetti casuali... Questo potrebbe richiedere tempo."):
                    # Per l'effetto random, params è una tupla con solo il livello di casualità
                    final_output_video_path = process_video(video_path, 'random', (random_level,), max_frames_to_process, apply_audio_glitch_checkbox)
        
        # --- Se il video finale è stato generato con successo, mostra il pulsante di download ---
        if final_output_video_path and os.path.exists(final_output_video_path):
            st.download_button(
                "⬇️ Scarica Video Glitchato Finale",
                data=open(final_output_video_path, 'rb').read(),
                file_name="glitched_video_with_audio.mp4" if apply_audio_glitch_checkbox and MOVIEPY_AVAILABLE else "glitched_video.mp4",
                mime="video/mp4"
            )
            os.unlink(final_output_video_path) # Pulizia del file temporaneo dopo il download
            st.success("Video completato! Puoi scaricarlo qui sopra.")

    except Exception as e:
        st.error(f"Errore generale nell'elaborazione: {str(e)}")
        st.info("Assicurati che il file caricato sia un video valido (MP4, AVI, MOV, MKV) e prova a ridurre il numero di frames.")
    finally:
        # Assicurati che il file temporaneo del video originale venga eliminato
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
            
else:
    st.info("📁 Carica un video per iniziare!")
    st.markdown("""
    ### 📋 Istruzioni:
    1. **Carica un video** nei formati supportati (MP4, AVI, MOV, MKV).
    2. **Scegli le impostazioni** degli effetti video nelle tab dedicate (VHS, Distruttivo, Noise, Glitch Combinato, Random).
    3. **Attiva o disattiva il glitch audio** con la checkbox.
    4. Clicca il pulsante "Genera Video..." per l'effetto desiderato.
    5. **Scarica il risultato** una volta completato il processo.
    
    ### ⚠️ Note importanti:
    - L'elaborazione video può richiedere **molto tempo** e risorse, specialmente per video lunghi o con molti frames.
    - Utilizza il controllo "Massimo numero di frames da processare" per **ridurre i tempi** di elaborazione e i consumi di memoria.
    - Gli effetti audio sono sperimentali e possono variare.
    - Se riscontri problemi, prova a riavviare l'applicazione o a caricare un video più corto.
    """)

# Footer
st.markdown("---")
st.markdown("🎬🔥 **VideoDistruktor** - Glitcha i tuoi video!")
st.markdown("*💡 Perfetto per creare effetti visual disturbati e atmosfere cyberpunk!*")
