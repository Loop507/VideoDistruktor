# 🎬🔥 VideoDistruktor

**Trasforma i tuoi video in opere d'arte glitchate con effetti video e audio professionali!**

## ✨ Caratteristiche

- **6 Effetti Glitch**: VHS, Distruttivo, Noise, Combinato, Broken TV, Random
- **Audio Glitch**: Effetti audio sincronizzati con video (con FFmpeg)
- **Controlli Avanzati**: Parametri personalizzabili per ogni effetto
- **Formati Supportati**: MP4, AVI, MOV, MKV
- **Web-Based**: Nessuna installazione richiesta

## 🚀 Demo Live

👉 **[Prova VideoDistruktor](https://your-app-url.streamlit.app)**

## 🎭 Effetti Disponibili

| Effetto | Descrizione | Perfetto per |
|---------|-------------|--------------|
| 📼 **VHS** | Scanline, color shift, vintage noise | Video retrò, vaporwave |
| 💥 **Distruttivo** | Blocchi spostati, frammentazione | Arte sperimentale, horror |
| 📺 **Noise** | Disturbi digitali, pixel damage | Cyberpunk, glitch art |
| 🌟 **Combinato** | Multipli effetti insieme | Massimo impatto visivo |
| 📻 **Broken TV** | Linee spostate, flicker | Horror, atmosfere inquietanti |
| 🎲 **Random** | Effetto casuale ogni volta | Sperimentazione creativa |

## 📦 Installazione

### Deploy su Streamlit Cloud

1. **Fork questo repository**
2. **Crea questi file nel repository:**

```bash
# requirements.txt
streamlit
numpy
Pillow
opencv-python-headless
scipy
librosa
soundfile
ffmpeg-python
```

```bash
# packages.txt
ffmpeg
```

3. **Deploy su [Streamlit Cloud](https://share.streamlit.io)**

### Installazione Locale

```bash
git clone https://github.com/tuousername/videodistributor.git
cd videodistributor
pip install -r requirements.txt
streamlit run app.py
```

**Nota**: Per gli effetti audio, installa FFmpeg sul tuo sistema.

## 🎮 Come Usare

1. **Carica video** (MP4, AVI, MOV, MKV)
2. **Scegli effetto** dal menu dropdown
3. **Regola parametri** con gli slider
4. **Attiva audio glitch** (se disponibile)
5. **Processa** e scarica il risultato!

## ⚙️ Parametri Ottimali

### 📼 VHS (Vintage)
- Intensità: `0.8-1.2`
- Scanline: `1.0-1.5`  
- Color Shift: `0.5-1.0`

### 💥 Distruttivo (Arte)
- Blocchi: `1.0-2.0`
- Numero: `1.2-2.0`
- Spostamento: `1.5-2.5`

### 📺 Noise (Cyberpunk)
- Intensità: `1.0-2.0`
- Copertura: `1.5-2.5`
- Chaos: `1.0-2.0`

## 🎵 Effetti Audio

| Effetto | Audio Features |
|---------|----------------|
| VHS | Wow & Flutter, Tape Hiss |
| Distruttivo | Skip, Reverse, Chaos |
| Noise | Digital Artifacts, Bit Crush |
| Broken TV | Static, Channel Separation |

## 🔧 Troubleshooting

**❌ "Impossibile aprire video"**
- Usa formato MP4 con codec H.264
- Riduci dimensioni file

**⚠️ "FFmpeg non disponibile"**  
- Aggiungi `ffmpeg` al `packages.txt` (Streamlit Cloud)
- Installa FFmpeg localmente

**🐌 Processing lento**
- Imposta limite frame per test
- Usa parametri più bassi

## 🎨 Esempi Creative

```python
# Video musicali retrò
VHS(intensity=1.0, scanline=1.2, color_shift=0.8)

# Horror intenso  
Broken_TV(shift=2.0, flicker=2.5) + Distruttivo(chaos=1.5)

# Arte digitale
Noise(intensity=2.0, coverage=2.5, chaos=2.0)

# Massimo impatto
Combinato(tutti_attivi=True, parametri_medi=1.5)
```

## 📱 Social Media

- **Instagram Stories**: VHS leggero (0.5-0.8)
- **TikTok**: Random/Combinato per attenzione
- **YouTube**: Distruttivo per intro/outro
- **Twitter**: Noise per GIF artistiche

## 🤝 Contributi

Contributi benvenuti! Apri un issue o pull request.

## 📄 Licenza

MIT License - vedi [LICENSE](LICENSE)

## 🙏 Credits

- **Sviluppato da**: loop507
- **Basato su**: Streamlit, OpenCV, librosa
- **Ispirato da**: Glitch art e cultura digitale

---

**⭐ Se ti piace VideoDistruktor, lascia una stella!**

**🔥 Crea. Distruggi. Rigenera. Ripeti.**
