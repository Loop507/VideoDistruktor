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
    OPENCV_AVAILABLE = False
    st.error(f"❌ Errore OpenCV: {e}")
    st.stop()

# Configurazione della pagina
st.set_page_config(page_title="GlitchLabLoop507 - Video Edition", layout="centered")

st.title("🎬🔥 GlitchLabLoop507 - Video Edition")
st.write("Carica un video e genera versioni glitchate: VHS, Distruttivo e Random!")

# File uploader per video
uploaded_file = st.file_uploader("📁 Carica un video", type=["mp4", "avi", "mov", "mkv"])

def frame_to_pil(frame):
    """Converte frame OpenCV in PIL Image"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_frame(pil_img):
    """Converte PIL Image in frame OpenCV"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def glitch_vhs_frame(frame, intensity=1.0, scanline_freq=1.0, color_shift=1.0):
    """Applica effetto VHS a un singolo frame"""
    try:
        # Converti frame in array numpy
        arr = frame.copy()
        h, w, _ = arr.shape
        
        # Parametri basati sui controlli utente
        base_intensity = int(10 + (25 * intensity))
        freq1 = 3 + (9 * scanline_freq)
        freq2 = 1 + (5 * scanline_freq)
        
        # Scanlines distorte
        for y in range(0, h, 2):  # Skip alcune righe per performance
            shift = int(base_intensity * np.sin(y / freq1) + (base_intensity//2) * np.sin(y / freq2))
            if shift != 0:
                arr[y:y+1, :, :] = np.roll(arr[y:y+1, :, :], shift, axis=1)
            
            # Aggiungi rumore occasionale
            if random.random() < (0.1 + 0.1 * intensity):
                noise_intensity = int(5 + (15 * intensity))
                noise = np.random.randint(-noise_intensity, noise_intensity, (1, w, 3))
                arr[y:y+1, :, :] = np.clip(arr[y:y+1, :, :] + noise, 0, 255)
        
        # Separazione canali colore
        b, g, r = cv2.split(arr)
        shift_multiplier = color_shift
        r_shift = int(10 * shift_multiplier + random.randint(0, int(15 * shift_multiplier)))
        b_shift = int(-10 * shift_multiplier + random.randint(int(-15 * shift_multiplier), 0))
        g_shift = int(random.randint(int(-5 * shift_multiplier), int(5 * shift_multiplier)))
        
        r = np.roll(r, r_shift, axis=1)
        b = np.roll(b, b_shift, axis=1)
        g = np.roll(g, g_shift, axis=0)
        
        # Saturazione colori
        sat_range = 0.2 * color_shift
        r_sat = 1.0 + random.uniform(0, sat_range)
        g_sat = 1.0 - random.uniform(0, sat_range * 0.8)
        b_sat = 1.0 + random.uniform(-sat_range * 0.5, sat_range)
        
        r = np.clip(r * r_sat, 0, 255)
        g = np.clip(g * g_sat, 0, 255)
        b = np.clip(b * b_sat, 0, 255)
        
        return cv2.merge([b, g, r])
    except Exception as e:
        return frame

def glitch_distruttivo_frame(frame, block_size=1.0, num_blocks=1.0, displacement=1.0):
    """Applica effetto distruttivo a un singolo frame"""
    try:
        arr = frame.copy()
        h, w, _ = arr.shape
        
        if w < 50 or h < 50:
            return frame
        
        # Numero di blocchi ridotto per performance video
        base_blocks = min(30, w * h // 3000)
        total_blocks = int(base_blocks * (0.5 + 1.0 * num_blocks))
        
        for i in range(total_blocks):
            # Dimensioni blocchi
            base_max_w = min(40, w // 6)
            base_max_h = min(40, h // 6)
            
            max_block_w = int(base_max_w * (0.3 + 1.2 * block_size))
            max_block_h = int(base_max_h * (0.3 + 1.2 * block_size))
            
            w_block = random.randint(max(3, max_block_w//3), max_block_w)
            h_block = random.randint(max(3, max_block_h//3), max_block_h)
            
            # Posizione iniziale
            x = random.randint(0, max(0, w - w_block))
            y = random.randint(0, max(0, h - h_block))
            
            # Spostamento
            base_displacement = min(w//8, h//8)
            max_displacement = int(base_displacement * displacement)
            dx = random.randint(-max_displacement, max_displacement)
            dy = random.randint(-max_displacement, max_displacement)
            
            # Copia e sposta il blocco
            if y + h_block <= h and x + w_block <= w:
                block = arr[y:y+h_block, x:x+w_block].copy()
                
                x_new = np.clip(x + dx, 0, w - w_block)
                y_new = np.clip(y + dy, 0, h - h_block)
                
                # Distorsione occasionale
                if random.random() < (0.1 + 0.3 * displacement):
                    distortion = int(3 + 8 * displacement)
                    block = np.roll(block, random.randint(-distortion, distortion), axis=1)
                
                arr[y_new:y_new+h_block, x_new:x_new+w_block] = block
        
        return arr
    except Exception as e:
        return frame

def glitch_noise_frame(frame, noise_intensity=1.0, coverage=1.0, chaos=1.0):
    """Applica effetto noise a un singolo frame"""
    try:
        arr = frame.copy().astype(np.int16)
        h, w, _ = arr.shape
        
        base_intensity = int(20 + (60 * noise_intensity))
        coverage_factor = 0.3 + (0.5 * coverage)
        
        # Tipo di noise basato sul chaos
        if chaos < 0.3:
            noise_type = 'bands'
        elif chaos < 0.6:
            noise_type = 'pixels'
        else:
            noise_type = 'mixed'
        
        if noise_type == 'bands':
            # Rumore a bande (ridotto per performance)
            num_bands = int(3 + (8 * coverage))
            for _ in range(num_bands):
                start_y = random.randint(0, h-1)
                band_height = int(2 + (15 * noise_intensity))
                end_y = min(start_y + band_height, h)
                
                band_noise = np.random.randint(-base_intensity, base_intensity, (end_y - start_y, w, 3))
                arr[start_y:end_y] += band_noise
        
        elif noise_type == 'pixels':
            # Rumore pixel casuali (ridotto)
            base_pixels = w * h // 60
            num_pixels = int(base_pixels * coverage * (0.5 + 0.5 * chaos))
            pixel_positions = [(random.randint(0, w-1), random.randint(0, h-1)) for _ in range(num_pixels)]
            
            for x, y in pixel_positions:
                pixel_noise = np.random.randint(-base_intensity, base_intensity, 3)
                arr[y, x] += pixel_noise
        
        else:  # mixed
            # Effetto misto leggero
            general_intensity = int(base_intensity * 0.3)
            # Solo su alcune zone per performance
            zones = int(h * coverage_factor / 10)
            for _ in range(zones):
                start_y = random.randint(0, h-10)
                end_y = min(start_y + 10, h)
                zone_noise = np.random.randint(-general_intensity, general_intensity, (end_y - start_y, w, 3))
                arr[start_y:end_y] += zone_noise
        
        # Saturazione casuale dei canali
        if chaos > 0.4:
            channel = random.randint(0, 2)
            multiplier = 0.7 + (0.6 * chaos)
            multiplier = random.uniform(1/multiplier, multiplier)
            arr[:,:,channel] = np.clip(arr[:,:,channel] * multiplier, 0, 255)
        
        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception as e:
        return frame

def process_video(video_path, effect_type, params, max_frames=None):
    """Processa il video con l'effetto scelto"""
    cap = cv2.VideoCapture(video_path)
    
    # Proprietà del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    # Configurazione output
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Applica l'effetto glitch
            if effect_type == 'vhs':
                processed_frame = glitch_vhs_frame(frame, *params)
            elif effect_type == 'distruttivo':
                processed_frame = glitch_distruttivo_frame(frame, *params)
            elif effect_type == 'noise':
                processed_frame = glitch_noise_frame(frame, *params)
            else:  # random
                # Scelta casuale per ogni frame (opzionale)
                effects = [
                    (glitch_vhs_frame, random.random(), random.random(), random.random()),
                    (glitch_distruttivo_frame, random.random(), random.random(), random.random()),
                    (glitch_noise_frame, random.random(), random.random(), random.random())
                ]
                chosen_effect, p1, p2, p3 = random.choice(effects)
                processed_frame = chosen_effect(frame, p1 * params[0], p2 * params[0], p3 * params[0])
            
            out.write(processed_frame)
            
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processando frame {frame_count}/{total_frames}")
    
    finally:
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
    
    return output_path

# Logica principale
if uploaded_file is not None:
    try:
        # Salva il video temporaneamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Informazioni sul video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        st.success(f"✅ Video caricato!")
        st.info(f"📊 **Dettagli video:** {width}x{height} - {fps} FPS - {duration:.1f}s - {total_frames} frames")
        
        # Mostra preview del primo frame
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.image(first_frame_rgb, caption="🎬 Primo frame del video", use_container_width=True)
        
        # Controlli per limitare i frame (per performance)
        st.markdown("### ⚙️ Impostazioni Processing")
        max_frames = st.number_input(
            "🎯 Massimo numero di frames da processare (0 = tutti)", 
            min_value=0, 
            max_value=total_frames, 
            value=min(300, total_frames),  # Default 300 frames o meno
            help="Limita il numero di frames per ridurre i tempi di processing"
        )
        
        if max_frames == 0:
            max_frames = total_frames
        
        processing_duration = (max_frames / fps) if fps > 0 else 0
        st.info(f"📅 Verranno processati {max_frames} frames ({processing_duration:.1f}s di video)")
        
        # --- CONTROLLI EFFETTI ---
        st.markdown("### 🎛️ Controlli Effetti")
        
        # Tabs per i diversi effetti
        tab1, tab2, tab3, tab4 = st.tabs(["📺 VHS", "💥 Distruttivo", "🌀 Noise", "🎲 Random"])
        
        with tab1:
            st.markdown("**Effetto VHS Glitch**")
            col1, col2, col3 = st.columns(3)
            with col1:
                vhs_intensity = st.slider("Intensità Distorsione", 0.0, 2.0, 1.0, 0.1, key="vhs_int")
            with col2:
                vhs_scanlines = st.slider("Frequenza Scanlines", 0.0, 2.0, 1.0, 0.1, key="vhs_scan")
            with col3:
                vhs_colors = st.slider("Separazione Colori", 0.0, 2.0, 1.0, 0.1, key="vhs_col")
            
            if st.button("🎬 Genera Video VHS", key="btn_vhs"):
                with st.spinner("📺 Processando video con effetto VHS..."):
                    output_path = process_video(video_path, 'vhs', (vhs_intensity, vhs_scanlines, vhs_colors), max_frames)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Scarica Video VHS",
                            f.read(),
                            "vhs_glitch_video.mp4",
                            "video/mp4"
                        )
                    os.unlink(output_path)
        
        with tab2:
            st.markdown("**Effetto Distruttivo**")
            col1, col2, col3 = st.columns(3)
            with col1:
                dest_blocks = st.slider("Dimensione Blocchi", 0.0, 2.0, 1.0, 0.1, key="dest_size")
            with col2:
                dest_number = st.slider("Numero Blocchi", 0.0, 2.0, 1.0, 0.1, key="dest_num")
            with col3:
                dest_displacement = st.slider("Spostamento", 0.0, 2.0, 1.0, 0.1, key="dest_disp")
            
            if st.button("🎬 Genera Video Distruttivo", key="btn_dest"):
                with st.spinner("💥 Processando video con effetto Distruttivo..."):
                    output_path = process_video(video_path, 'distruttivo', (dest_blocks, dest_number, dest_displacement), max_frames)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Scarica Video Distruttivo",
                            f.read(),
                            "distruttivo_glitch_video.mp4",
                            "video/mp4"
                        )
                    os.unlink(output_path)
        
        with tab3:
            st.markdown("**Effetto Noise**")
            col1, col2, col3 = st.columns(3)
            with col1:
                noise_intensity = st.slider("Intensità Rumore", 0.0, 2.0, 1.0, 0.1, key="noise_int")
            with col2:
                noise_coverage = st.slider("Copertura", 0.0, 2.0, 1.0, 0.1, key="noise_cov")
            with col3:
                noise_chaos = st.slider("Caos", 0.0, 1.0, 0.5, 0.1, key="noise_chaos")
            
            if st.button("🎬 Genera Video Noise", key="btn_noise"):
                with st.spinner("🌀 Processando video con effetto Noise..."):
                    output_path = process_video(video_path, 'noise', (noise_intensity, noise_coverage, noise_chaos), max_frames)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Scarica Video Noise",
                            f.read(),
                            "noise_glitch_video.mp4",
                            "video/mp4"
                        )
                    os.unlink(output_path)
        
        with tab4:
            st.markdown("**Effetto Random (Combo Casuale)**")
            random_level = st.slider("Livello Casualità", 0.0, 2.0, 1.0, 0.1, key="random_lev")
            st.info("🎲 Ogni frame avrà un effetto casuale diverso!")
            
            if st.button("🎬 Genera Video Random", key="btn_random"):
                with st.spinner("🎲 Processando video con effetti casuali..."):
                    output_path = process_video(video_path, 'random', (random_level,), max_frames)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Scarica Video Random",
                            f.read(),
                            "random_glitch_video.mp4",
                            "video/mp4"
                        )
                    os.unlink(output_path)
        
        # Pulizia file temporaneo
        os.unlink(video_path)
    
    except Exception as e:
        st.error(f"Errore nel caricamento del video: {str(e)}")
        st.info("Assicurati che il file sia un video valido (MP4, AVI, MOV, MKV)")

else:
    st.info("📁 Carica un video per iniziare!")
    st.markdown("""
    ### 📋 Istruzioni:
    1. **Carica un video** nei formati supportati (MP4, AVI, MOV, MKV)
    2. **Regola i parametri** degli effetti usando i controlli
    3. **Scegli l'effetto** desiderato e clicca per generare
    4. **Scarica il risultato** una volta completato
    
    ### ⚠️ Note importanti:
    - Il processing può richiedere tempo in base alla lunghezza del video
    - Limita il numero di frames per ridurre i tempi di elaborazione
    - Video più grandi richiederanno più tempo e memoria
    """)

# Footer
st.markdown("---")
st.markdown("🎬🔥 **GlitchLabLoop507 - Video Edition** - Glitcha i tuoi video!")
st.markdown("*💡 Perfetto per creare effetti visual disturbati e atmosfere cyberpunk!*")
