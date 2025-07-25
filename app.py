import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
import random
import cv2 # Importa OpenCV qui


# Configurazione della pagina
st.set_page_config(page_title="VideoDistruktor by loop507", layout="centered")

# Modifica del titolo con caratteri più piccoli per "by loop507"
st.markdown("<h1>🎬🔥 VideoDistruktor <span style='font-size:0.5em;'>by loop507</span></h1>", unsafe_allow_html=True)
st.write("Carica un video e genera versioni glitchate: VHS, Distruttivo, Noise, Combinato o Random!")

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
        if w < 20 or h < 20: # Aggiungi un controllo per dimensioni troppo piccole
            return frame
        max_total_blocks = min(10, w * h // 5000) # Limita il numero di blocchi per performance
        total_blocks = int(max(1, max_total_blocks * num_blocks))
        for i in range(total_blocks):
            # Limita le dimensioni dei blocchi in base alla dimensione del frame e block_size
            max_w_block = int(min(w // 8, 20 + 20 * block_size))
            max_h_block = int(min(h // 8, 20 + 20 * block_size))
            w_block = random.randint(min(3, max_w_block), max_w_block)
            h_block = random.randint(min(3, max_h_block), max_h_block)
            
            # Assicurati che x e y non vadano fuori dai limiti
            x = random.randint(0, max(0, w - w_block -1))
            y = random.randint(0, max(0, h - h_block -1))
            
            max_disp = int(min(w//10, h//10, 10 + 10 * displacement))
            dx = random.randint(-max_disp, max_disp)
            dy = random.randint(-max_disp, max_disp)
            
            x_new = np.clip(x + dx, 0, w - w_block)
            y_new = np.clip(y + dy, 0, h - h_block)
            
            # Controlli sui bordi per prevenire errori di slicing
            if h_block > 0 and w_block > 0 and y + h_block <= h and x + w_block <= w:
                block = arr[y:y+h_block, x:x+w_block].copy()
                if y_new + h_block <= h and x_new + w_block <= w:
                    arr[y_new:y_new+h_block, x_new:x_new+w_block] = block
        return arr
    except Exception as e:
        # st.warning(f"Errore in glitch_distruttivo_frame: {e}") # Per debug
        return frame

def glitch_noise_frame(frame, noise_intensity=1.0, coverage=1.0, chaos=1.0):
    try:
        arr = frame.copy().astype(np.int16) # Usa int16 per evitare overflow temporanei
        h, w, _ = arr.shape
        base_intensity = int(10 + (40 * noise_intensity)) # Regola l'intensità base del rumore
        
        if random.random() < coverage: # La probabilità di applicare rumore dipende dalla copertura
            if chaos < 0.4: # Rumore a bande orizzontali
                num_bands = int(2 + (5 * coverage))
                for _ in range(num_bands):
                    start_y = random.randint(0, h-1)
                    band_height = int(1 + (10 * noise_intensity)) # Altezza della banda
                    end_y = min(start_y + band_height, h)
                    band_noise = np.random.randint(-base_intensity, base_intensity, (end_y - start_y, w, 3))
                    arr[start_y:end_y] += band_noise
            elif chaos < 0.8: # Rumore sparso (pixel)
                num_pixels = int(w * h * 0.005 * coverage) # Numero di pixel influenzati
                for _ in range(num_pixels):
                    x = random.randint(0, w-1)
                    y = random.randint(0, h-1)
                    pixel_noise = np.random.randint(-base_intensity, base_intensity, 3)
                    arr[y, x] += pixel_noise
            else: # Rumore generale a blocchi o distorsione ampia
                general_intensity = int(base_intensity * 0.5) # Un po' meno intenso
                if random.random() < 0.2: # 20% di probabilità di un blocco di rumore più grande
                    noise_block_h = int(h * (0.1 + 0.2 * coverage))
                    if noise_block_h > 0:
                        start_y = random.randint(0, h - noise_block_h)
                        arr[start_y:start_y+noise_block_h] += np.random.randint(-general_intensity, general_intensity, (noise_block_h, w, 3))

        if chaos > 0.5: # Piccole distorsioni di canale colore
            channel = random.randint(0, 2) # Scegli un canale a caso (B, G, R)
            multiplier = random.uniform(0.8, 1.2) # Moltiplica per variare il colore
            arr[:,:,channel] = np.clip(arr[:,:,channel] * multiplier, 0, 255) # Applica e clippa
            
        return np.clip(arr, 0, 255).astype(np.uint8) # Clio finale a 0-255 e ritorna a uint8
    except Exception as e:
        # st.warning(f"Errore in glitch_noise_frame: {e}") # Per debug
        return frame


def process_video(video_path, effect_type, params, max_frames=None):
    """Processa il video con l'effetto scelto, senza gestione audio."""
    
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

    output_video_path = tempfile.mktemp(suffix='.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec per MP4
    
    try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
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
                    
                    # Estrai i parametri specifici per gli effetti combinati
                    vhs_intensity = params.get("vhs_intensity", 1.0)
                    vhs_scanlines = params.get("vhs_scanlines", 1.0)
                    vhs_colors = params.get("vhs_colors", 1.0)

                    dest_block_size = params.get("dest_block_size", 1.0)
                    dest_num_blocks = params.get("dest_num_blocks", 1.0)
                    dest_displacement = params.get("dest_displacement", 1.0)

                    noise_intensity_val = params.get("noise_intensity", 1.0)
                    noise_coverage_val = params.get("noise_coverage", 1.0)
                    noise_chaos_val = params.get("noise_chaos", 0.5)

                    if params.get("apply_vhs"):
                        current_frame_to_process = glitch_vhs_frame(current_frame_to_process, vhs_intensity, vhs_scanlines, vhs_colors)
                    
                    if params.get("apply_distruttivo"):
                        current_frame_to_process = glitch_distruttivo_frame(current_frame_to_process, dest_block_size, dest_num_blocks, dest_displacement)
                    
                    if params.get("apply_noise"):
                        current_frame_to_process = glitch_noise_frame(current_frame_to_process, noise_intensity_val, noise_coverage_val, noise_chaos_val)
                    
                    processed_frame = current_frame_to_process

                elif effect_type == 'random':
                    # Qui params è una tupla con solo il random_level
                    random_level = params[0] if params else 1.0
                    
                    effects = [
                        (glitch_vhs_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
                        (glitch_distruttivo_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)),
                        (glitch_noise_frame, random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))
                    ]
                    
                    chosen_effect, p1, p2, p3 = random.choice(effects)
                    # Scala i parametri dell'effetto scelto con il random_level
                    processed_frame = chosen_effect(frame, p1 * random_level, p2 * random_level, p3 * random_level)
                
                else:
                    st.warning(f"Tipo di effetto '{effect_type}' non riconosciuto.")
                    processed_frame = frame
            except Exception as e:
                st.warning(f"⚠️ Errore durante l'applicazione dell'effetto al frame {frame_count}: {e}. Skipping frame.")
                processed_frame = frame # Continua con il frame originale in caso di errore
            
            out.write(processed_frame)
            
            frame_count += 1
            progress = frame_count / actual_total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processando frame {frame_count}/{actual_total_frames}...")
        
        st.success("✅ Video processato!")
        return output_video_path
        
    except Exception as e:
        st.error(f"Errore critico durante il processing del video: {e}")
        return None
    finally:
        cap.release()
        if 'out' in locals() and out.isOpened(): # Controlla se 'out' esiste e se è aperto
            out.release()
        progress_bar.empty()
        status_text.empty()


# Logica principale dell'applicazione Streamlit
if uploaded_file is not None:
    video_path = None
    final_output_video_path = None # Variabile per il percorso finale
    try:
        # Salva il file caricato in un file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Analizza le proprietà del video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Impossibile leggere il video caricato. Assicurati che non sia corrotto o un formato supportato.")
            # Pulisci il file temporaneo in caso di errore
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            st.stop() # Ferma l'esecuzione dello script
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release() # Rilascia il VideoCapture dopo aver letto le info
        
        st.success(f"✅ Video caricato!")
        st.info(f"📊 **Dettagli video:** {width}x{height} - {fps} FPS - {duration:.1f}s - {total_frames} frames")
        
        # Mostra il primo frame come anteprima
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.image(first_frame_rgb, caption="🎬 Primo frame del video", use_container_width=True)
        
        st.markdown("### ⚙️ Impostazioni Processing")
        
        # Calcola un valore di default ragionevole per max_frames
        max_frames_default = min(300, total_frames) # Di default processa max 300 frames o tutti se meno di 300
        
        max_frames = st.number_input(
            "🎯 Massimo numero di frames da processare (0 = tutti)", 
            min_value=0, 
            max_value=total_frames, 
            value=max_frames_default, # Imposta il valore di default calcolato
            help="Limita il numero di frames per ridurre i tempi di processing. Un valore più basso rende il processo più veloce."
        )
        
        if max_frames == 0:
            max_frames_to_process = total_frames
        else:
            max_frames_to_process = max_frames
        
        processing_duration = (max_frames_to_process / fps) if fps > 0 else 0
        st.info(f"📅 Verranno processati {max_frames_to_process} frames ({processing_duration:.1f}s di video)")
        
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
                    final_output_video_path = process_video(video_path, 'vhs', (vhs_intensity, vhs_scanlines, vhs_colors), max_frames_to_process)
        
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
                    final_output_video_path = process_video(video_path, 'distruttivo', (dest_blocks, dest_number, dest_displacement), max_frames_to_process)
        
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
                    final_output_video_path = process_video(video_path, 'noise', (noise_intensity, noise_coverage, noise_chaos), max_frames_to_process)
        
        with tab4: # Nuova tab Glitch Combinato
            st.markdown("**Configura il tuo Glitch Combinato**")
            
            apply_vhs = st.checkbox("📺 Applica effetto VHS", value=True, key="chk_vhs_1")
            if apply_vhs:
                st.markdown("--- VHS Settings ---")
                col_vhs1, col_vhs2, col_vhs3 = st.columns(3)
                with col_vhs1:
                    vhs_intensity_c = st.slider("VHS Intensità Distorsione", 0.0, 2.0, 1.0, 0.1, key="vhs_int_c")
                with col_vhs2:
                    vhs_scanlines_c = st.slider("VHS Frequenza Scanlines", 0.0, 2.0, 1.0, 0.1, key="vhs_scan_c")
                with col_vhs3:
                    vhs_colors_c = st.slider("VHS Separazione Colori", 0.0, 2.0, 1.0, 0.1, key="vhs_col_c")
            
            st.markdown("---") # Separatore
            
            apply_distruttivo = st.checkbox("💥 Applica effetto Distruttivo", value=True, key="chk_dist_1")
            if apply_distruttivo:
                st.markdown("--- Distruttivo Settings ---")
                col_dest1, col_dest2, col_dest3 = st.columns(3)
                with col_dest1:
                    dest_blocks_c = st.slider("Distr. Dimensione Blocchi", 0.0, 2.0, 1.0, 0.1, key="dest_size_c")
                with col_dest2:
                    dest_number_c = st.slider("Distr. Numero Blocchi", 0.0, 2.0, 1.0, 0.1, key="dest_num_c")
                with col_dest3:
                    dest_displacement_c = st.slider("Distr. Spostamento", 0.0, 2.0, 1.0, 0.1, key="dest_disp_c")
            
            st.markdown("---") # Separatore

            apply_noise = st.checkbox("🌀 Applica effetto Noise", value=True, key="chk_noise_1")
            if apply_noise:
                st.markdown("--- Noise Settings ---")
                col_noise1, col_noise2, col_noise3 = st.columns(3)
                with col_noise1:
                    noise_intensity_c = st.slider("Noise Intensità Rumore", 0.0, 2.0, 1.0, 0.1, key="noise_int_c")
                with col_noise2:
                    noise_coverage_c = st.slider("Noise Copertura", 0.0, 2.0, 1.0, 0.1, key="noise_cov_c")
                with col_noise3:
                    noise_chaos_c = st.slider("Noise Caos", 0.0, 1.0, 0.5, 0.1, key="noise_chaos_c")
            
            st.markdown("---") # Separatore
            
            if st.button("🎬 Genera Video Combinato", key="btn_combined_1"):
                combined_params = {
                    "apply_vhs": apply_vhs,
                    "vhs_intensity": vhs_intensity_c if apply_vhs else 1.0,
                    "vhs_scanlines": vhs_scanlines_c if apply_vhs else 1.0,
                    "vhs_colors": vhs_colors_c if apply_vhs else 1.0,

                    "apply_distruttivo": apply_distruttivo,
                    "dest_block_size": dest_blocks_c if apply_distruttivo else 1.0,
                    "dest_num_blocks": dest_number_c if apply_distruttivo else 1.0,
                    "dest_displacement": dest_displacement_c if apply_distruttivo else 1.0,

                    "apply_noise": apply_noise,
                    "noise_intensity": noise_intensity_c if apply_noise else 1.0,
                    "noise_coverage": noise_coverage_c if apply_noise else 1.0,
                    "noise_chaos": noise_chaos_c if apply_noise else 0.5
                }
                
                with st.spinner("✨ Processando video con effetti combinati... Questo potrebbe richiedere tempo."):
                    final_output_video_path = process_video(video_path, 'combined', combined_params, max_frames_to_process)
        
        with tab5: # Tab Random
            st.markdown("**Effetto Random (Combo Casuale)**")
            random_level = st.slider("Livello Casualità", 0.0, 2.0, 1.0, 0.1, key="random_lev_v2_1")
            st.info("🎲 Ogni frame avrà un effetto casuale diverso con intensità proporzionale al Livello Casualità!")
            
            if st.button("🎬 Genera Video Random", key="btn_random_v2_1"):
                with st.spinner("🎲 Processando video con effetti casuali... Questo potrebbe richiedere tempo."):
                    # Per l'effetto random, params è una tupla con solo il livello di casualità
                    final_output_video_path = process_video(video_path, 'random', (random_level,), max_frames_to_process)
        
        # --- Se il video finale è stato generato con successo, mostra il pulsante di download ---
        if final_output_video_path and os.path.exists(final_output_video_path):
            st.download_button(
                "⬇️ Scarica Video Glitchato",
                data=open(final_output_video_path, 'rb').read(),
                file_name="glitched_video.mp4",
                mime="video/mp4"
            )
            # Pulizia del file temporaneo dopo il download
            os.unlink(final_output_video_path) 
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
    3. Clicca il pulsante "Genera Video..." per l'effetto desiderato.
    4. **Scarica il risultato** una volta completato il processo.
    
    ### ⚠️ Note importanti:
    - L'elaborazione video può richiedere **molto tempo** e risorse, specialmente per video lunghi o con molti frames.
    - Utilizza il controllo "Massimo numero di frames da processare" per **ridurre i tempi** di elaborazione e i consumi di memoria.
    - Se riscontri problemi, prova a riavviare l'applicazione o a caricare un video più corto.
    """)

# Footer
st.markdown("---")
st.markdown("🎬🔥 **VideoDistruktor** - Glitcha i tuoi video!")
st.markdown("*💡 Perfetto per creare effetti visual disturbati e atmosfere cyberpunk!*")
