import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
import random
import io

# Importa OpenCV con gestione errori
try:
    import cv2
    OPENCV_AVAILABLE = True
    st.success("✅ OpenCV caricato correttamente!")
except ImportError as e:
    st.error(f"❌ Errore: OpenCV non trovato. Assicurati di averlo installato con 'pip install opencv-python'. Dettagli: {e}")
    st.stop() # Ferma l'esecuzione se OpenCV non è disponibile

# Configurazione della pagina
st.set_page_config(page_title="VideoDistruktor - Video Edition", layout="centered")

st.title("🎬🔥 VideoDistruktor")
st.write("Carica un video e genera versioni glitchate: VHS, Distruttivo e Random!")

# File uploader per video
uploaded_file = st.file_uploader("📁 Carica un video", type=["mp4", "avi", "mov", "mkv"])

def frame_to_pil(frame):
    """Converte frame OpenCV (BGR) in PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_frame(pil_img):
    """Converte PIL Image (RGB) in frame OpenCV (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Funzioni degli effetti ottimizzate (invariate, ma usate diversamente) ---

def glitch_vhs_frame(frame, intensity=1.0, scanline_freq=1.0, color_shift=1.0):
    """Applica effetto VHS a un singolo frame (ottimizzato)"""
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
    """Applica effetto distruttivo a un singolo frame (ottimizzato)"""
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
    """Applica effetto noise a un singolo frame (ottimizzato)"""
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

def process_video(video_path, effect_type, params, max_frames=None):
    """Processa il video con l'effetto scelto"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il video. Potrebbe essere danneggiato o non supportato.")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames and max_frames > 0:
        actual_total_frames = min(total_frames, max_frames)
    else:
        actual_total_frames = total_frames
    
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            st.error("❌ Impossibile inizializzare VideoWriter. Controlla i codec o i permessi di scrittura.")
            cap.release()
            return None
        
        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened() and frame_count < actual_total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = frame # Default: frame originale in caso di problemi
            try:
                # --- LOGICA MODIFICATA PER GLI EFFETTI ---
                if effect_type == 'combined':
                    current_frame_to_process = frame # Inizia con il frame originale
                    global_intensity = params.get("global_intensity", 1.0)

                    # Mappa l'intensità globale ai parametri specifici di ciascun effetto
                    # Questi valori sono stati scelti per dare un buon range di controllo
                    # Puoi modificarli per ottenere effetti diversi
                    vhs_intensity = 0.5 + 1.5 * global_intensity
                    vhs_scanlines = 0.5 + 1.5 * global_intensity
                    vhs_colors = 0.5 + 1.5 * global_intensity

                    dest_block_size = 0.5 + 1.5 * global_intensity
                    dest_num_blocks = 0.5 + 1.5 * global_intensity
                    dest_displacement = 0.5 + 1.5 * global_intensity

                    noise_intensity_val = 0.5 + 1.5 * global_intensity
                    noise_coverage_val = 0.5 + 1.5 * global_intensity
                    noise_chaos_val = min(1.0, 0.2 + 0.8 * global_intensity) # Caos max 1.0


                    if params.get("apply_vhs"):
                        current_frame_to_process = glitch_vhs_frame(current_frame_to_process, vhs_intensity, vhs_scanlines, vhs_colors)
                    
                    if params.get("apply_distruttivo"):
                        current_frame_to_process = glitch_distruttivo_frame(current_frame_to_process, dest_block_size, dest_num_blocks, dest_displacement)
                    
                    if params.get("apply_noise"):
                        current_frame_to_process = glitch_noise_frame(current_frame_to_process, noise_intensity_val, noise_coverage_val, noise_chaos_val)
                    
                    processed_frame = current_frame_to_process

                elif effect_type == 'random':
                    # L'effetto random rimane come prima
                    effects = [
                        (glitch_vhs_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
                        (glitch_distruttivo_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
                        (glitch_noise_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))
                    ]
                    # I parametri del random sono ancora scalati dal livello di casualità (params[0])
                    chosen_effect, p1, p2, p3 = random.choice(effects)
                    processed_frame = chosen_effect(frame, p1 * params[0], p2 * params[0], p3 * params[0])
                # --- FINE LOGICA MODIFICATA ---
                
                # Questa sezione `else` non sarà più raggiunta dai pulsanti
                # perché i controlli singoli sono stati rimossi.
                # Rimane solo se volessi riaggiungere un singolo effetto in futuro.
                else: 
                     st.warning(f"Tipo di effetto '{effect_type}' non riconosciuto.")
                     processed_frame = frame
            except Exception as e:
                st.warning(f"⚠️ Errore durante l'applicazione dell'effetto al frame {frame_count}: {e}. Skipping frame.")
                processed_frame = frame # Usa il frame originale se l'effetto fallisce
            
            out.write(processed_frame)
            
            frame_count += 1
            progress = frame_count / actual_total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processando frame {frame_count}/{actual_total_frames}...")
        
    except Exception as e:
        st.error(f"Errore critico durante il processing del video: {e}")
        output_path = None # Indica che non è stato generato un output valido
    finally:
        cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        progress_bar.empty()
        status_text.empty()
    
    return output_path

# Logica principale dell'applicazione Streamlit
if uploaded_file is not None:
    video_path = None
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
        
        # --- CONTROLLI EFFETTI: MODIFICATI ---
        st.markdown("### 🎛️ Controlli Effetti")
        
        # Tabs per i diversi effetti: ora 'Glitch Combinato' e 'Random'
        tab_combined, tab_random = st.tabs(["✨ Glitch Combinato", "🎲 Random"])
        
        with tab_combined:
            st.markdown("**Configura il tuo Glitch Combinato**")
            
            # Checkbox per attivare/disattivare gli effetti
            apply_vhs = st.checkbox("📺 Applica effetto VHS", value=True, key="chk_vhs")
            apply_distruttivo = st.checkbox("💥 Applica effetto Distruttivo", value=True, key="chk_dist")
            apply_noise = st.checkbox("🌀 Applica effetto Noise", value=True, key="chk_noise")
            
            # Slider per l'intensità globale
            global_intensity_combined = st.slider("Livello di Intensità Globale", 0.0, 2.0, 1.0, 0.1, 
                                                  help="Controlla l'intensità di tutti gli effetti attivi. 0.0 = minima, 2.0 = massima.",
                                                  key="global_intensity_combined")
            
            if st.button("🎬 Genera Video Combinato", key="btn_combined"):
                # Prepara i parametri per la funzione process_video
                combined_params = {
                    "apply_vhs": apply_vhs,
                    "apply_distruttivo": apply_distruttivo,
                    "apply_noise": apply_noise,
                    "global_intensity": global_intensity_combined
                }
                
                with st.spinner("✨ Processando video con effetti combinati... Questo potrebbe richiedere tempo."):
                    output_path_processed = process_video(video_path, 'combined', combined_params, max_frames_to_process)
                    
                    if output_path_processed and os.path.exists(output_path_processed):
                        with open(output_path_processed, 'rb') as f:
                            st.download_button(
                                "⬇️ Scarica Video Combinato",
                                f.read(),
                                "combined_glitch_video.mp4",
                                "video/mp4"
                            )
                        os.unlink(output_path_processed)
                    else:
                        st.error("❌ Errore nella generazione del video combinato.")
        
        with tab_random:
            st.markdown("**Effetto Random (Combo Casuale)**")
            random_level = st.slider("Livello Casualità", 0.0, 2.0, 1.0, 0.1, key="random_lev_v2")
            st.info("🎲 Ogni frame avrà un effetto casuale diverso con intensità proporzionale al Livello Casualità!")
            
            if st.button("🎬 Genera Video Random", key="btn_random_v2"):
                with st.spinner("🎲 Processando video con effetti casuali... Questo potrebbe richiedere tempo."):
                    # Per l'effetto random, params è una tupla con solo il livello di casualità
                    output_path_processed = process_video(video_path, 'random', (random_level,), max_frames_to_process)
                    
                    if output_path_processed and os.path.exists(output_path_processed):
                        with open(output_path_processed, 'rb') as f:
                            st.download_button(
                                "⬇️ Scarica Video Random",
                                f.read(),
                                "random_glitch_video.mp4",
                                "video/mp4"
                            )
                        os.unlink(output_path_processed)
                    else:
                        st.error("❌ Errore nella generazione del video Random.")
        
    except Exception as e:
        st.error(f"Errore generale nell'elaborazione: {str(e)}")
        st.info("Assicurati che il file caricato sia un video valido (MP4, AVI, MOV, MKV) e prova a ridurre il numero di frames.")
    finally:
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
            
else:
    st.info("📁 Carica un video per iniziare!")
    st.markdown("""
    ### 📋 Istruzioni:
    1. **Carica un video** nei formati supportati (MP4, AVI, MOV, MKV).
    2. **Seleziona gli effetti** nella tab "Glitch Combinato" e regola il "Livello di Intensità Globale". Oppure scegli l'effetto "Random".
    3. **Genera il video** e scarica il risultato.
    
    ### ⚠️ Note importanti:
    - L'elaborazione video può richiedere **molto tempo** e risorse, specialmente per video lunghi o con molti frames.
    - Utilizza il controllo "Massimo numero di frames da processare" per **ridurre i tempi** di elaborazione e i consumi di memoria.
    - Se riscontri problemi, prova a riavviare l'applicazione o a caricare un video più corto.
    """)

# Footer
st.markdown("---")
st.markdown("🎬🔥 **VideoDistruktor** - Glitcha i tuoi video!")
st.markdown("*💡 Perfetto per creare effetti visual disturbati e atmosfere cyberpunk!*")
