import sounddevice as sd
import queue
import vosk
import sys
import pyttsx3
import requests
import json
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, Canvas
import numpy as np
import math
from PIL import Image, ImageTk


MODEL_PATH = "vosk-model-it-0.22"


model = vosk.Model(MODEL_PATH)


q = queue.Queue()


tts_lock = threading.Lock()
tts_in_progress = False
stop_event = threading.Event()


mic_activity = 0
tts_activity = 0
current_color = "#1E90FF"  
angle_offset = 0.0


tts_volume = 0.8
mic_enabled = True
log_visible = True
input_visible = True
canvas_expanded = False


def callback(indata, frames, time, status):
    global mic_activity
    if status:
        print("Status:", status, file=sys.stderr)
    if not mic_enabled:
        return
    data = bytes(indata)
    q.put(data)
    audio_data = np.frombuffer(indata, dtype=np.int16)
    if len(audio_data) > 0:
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
        mic_activity = min(rms / 500, 2.0)


def run_tts(text):
    global tts_in_progress, tts_activity, current_color, tts_volume
    try:
        current_color = "#32CD32"
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.setProperty('volume', tts_volume)
        for voice in engine.getProperty('voices'):
            if "italian" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.startLoop(False)
        engine.say(text)
        while engine.isBusy():
            engine.iterate()
            tts_activity = 0.5 + 0.5 * math.sin(time.time() * 5)
            if stop_event.is_set():
                engine.stop()
                break
            time.sleep(0.1)
    except Exception as e:
        print(f"Errore durante il TTS: {e}")
    finally:
        engine.endLoop()
        with tts_lock:
            tts_in_progress = False
            stop_event.clear()
            tts_activity = 0
        current_color = "#1E90FF"

def ollm(text):
    global tts_in_progress
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "gemma3:4b"
    payload = {"model": MODEL_NAME, "prompt": text, "stream": False, "system": "Sei un assistente personale di nome Jarvis che parla in italiano"}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            response_text = response.json().get('response', '')
            print(f"❖ Input: {text}")
            print(f"Risposta di Ollama: {response_text}")
            with tts_lock:
                if tts_in_progress:
                    return
                tts_in_progress = True
            threading.Thread(target=run_tts, args=(response_text,), daemon=True).start()
        else:
            print(f"Errore Ollama: {response.status_code}")
    except Exception as e:
        print(f"Errore nella richiesta a Ollama: {e}")

class JarvisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.command_map = {
            "prova": self.facc_prov,
        }
        self.title("JARVIS - Assistente Virtuale")
        self.geometry("800x600")
        self.configure(bg='#0A0A1A')
        self.tk_setPalette(background='#0A0A1A', foreground='white', activeBackground='#1E90FF', activeForeground='white')

        main_frame = tk.Frame(self, bg='#0A0A1A')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

       
        ctrl_frame = tk.Frame(main_frame, bg='#0A0A1A')
        ctrl_frame.pack(fill=tk.X, pady=(0,10))
        tk.Button(ctrl_frame, text="Mic On/Off", command=self.toggle_mic).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="STOP TTS", command=self.stop_tts).pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl_frame, text="Volume:").pack(side=tk.LEFT, padx=(20,2))
        self.vol_slider = tk.Scale(ctrl_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, command=self.set_volume)
        self.vol_slider.set(tts_volume)
        self.vol_slider.pack(side=tk.LEFT)
        tk.Button(ctrl_frame, text="Show/Hide Log", command=self.toggle_log).pack(side=tk.RIGHT, padx=5)
        tk.Button(ctrl_frame, text="Show/Hide Input", command=self.toggle_input).pack(side=tk.RIGHT)

       
        self.input_frame = tk.Frame(main_frame, bg='#0A0A1A')
        self.input_frame.pack(fill=tk.X)
        self.input_entry = tk.Entry(self.input_frame, font=("Consolas", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        tk.Button(self.input_frame, text="Send", command=self.send_input).pack(side=tk.LEFT)

        
        self.title_label = tk.Label(main_frame, text="JARVIS", font=("Arial", 24, "bold"), fg="#1E90FF", bg='#0A0A1A')
        self.title_label.pack()
        self.canvas = Canvas(main_frame, width=300, height=300, bg='#0A0A1A', highlightthickness=0)
        self.canvas.pack(pady=(10, 20))

        
        self.log_frame = tk.LabelFrame(main_frame, text="Log di Sistema", font=("Arial", 10), bg='#0A0A1A', fg='white', padx=10, pady=10)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_area = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, width=60, height=10, bg='#111122', fg='white', insertbackground='white', font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.config(state=tk.DISABLED)

        sys.stdout = self.TextRedirector(self.log_area, "stdout")
        sys.stderr = self.TextRedirector(self.log_area, "stderr")

        self.after(50, self.update_animation)
        threading.Thread(target=self.run_audio, daemon=True).start()

    class TextRedirector:
        def __init__(self, widget, tag="stdout"):
            self.widget = widget
            self.tag = tag
        def write(self, text):
            self.widget.config(state=tk.NORMAL)
            self.widget.insert(tk.END, text, (self.tag,))
            self.widget.config(state=tk.DISABLED)
            self.widget.see(tk.END)
        def flush(self): pass

    def facc_prov(self):
        print("Ha funzionato")


    
    def handle_command(self, text: str) -> bool:
        text = text.lower()
        for phrase, func in self.command_map.items():
            if phrase in text:
                func()    
                return True
        return False

    def toggle_mic(self):
        global mic_enabled
        mic_enabled = not mic_enabled
        status = "enabled" if mic_enabled else "disabled"
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break
        print(f"🎤 Microfono {status}")

    def stop_tts(self):
        global stop_event
        if tts_in_progress:
            stop_event.set()
            print("⏹️ TTS interrotto manualmente")

    def set_volume(self, val):
        global tts_volume
        tts_volume = float(val)
        print(f"🔊 Volume impostato a {tts_volume:.2f}")

    def toggle_log(self):
        global log_visible, canvas_expanded
        if log_visible:
            self.log_frame.pack_forget()
            self.canvas.pack_forget()
            self.title_label.config(font=("Arial", 48, "bold"))
            self.canvas.pack(fill=tk.BOTH, expand=True)
            canvas_expanded = True
        else:
            self.canvas.pack_forget()
            self.title_label.config(font=("Arial", 24, "bold"))
            self.canvas.config(width=300, height=300)
            self.canvas.pack(pady=(10, 20))
            self.log_frame.pack(fill=tk.BOTH, expand=True)
            canvas_expanded = False
        log_visible = not log_visible

    def toggle_input(self):
        global input_visible
        if input_visible:
            self.input_frame.pack_forget()
        else:
            self.input_frame.pack(fill=tk.X)
        input_visible = not input_visible

    

    def send_input(self):
        text = self.input_entry.get().strip()
        if not text:
            return


        self.input_entry.delete(0, tk.END)

   
        def job():
            if not self.handle_command(text):
                ollm(text)

        threading.Thread(target=job, daemon=True).start()



    def update_animation(self):
        global angle_offset
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        cx, cy = w//2, h//2
        base_radius = min(w, h)*0.2 if canvas_expanded else 60
        pulse = mic_activity*20 + tts_activity*15
        angle_offset = (angle_offset + 2) % 360
        for ring in range(3):
            radius = base_radius + ring*(base_radius*0.3) + pulse
            segments, seg_ang = 24, 360/24*0.3
            width = max(2, int(base_radius*0.03))
            for i in range(segments):
                start = (360/segments)*i + angle_offset*(1+ring*0.3)
                self.canvas.create_arc(cx-radius, cy-radius, cx+radius, cy+radius,
                                       start=start, extent=seg_ang,
                                       style=tk.ARC, outline=current_color, width=width)
        self.canvas.create_oval(cx-base_radius*1.1, cy-base_radius*1.1,
                                cx+base_radius*1.1, cy+base_radius*1.1,
                                outline=current_color, width=max(2, int(base_radius*0.04)))
        font_size = int((base_radius*0.8))
        self.canvas.create_text(cx, cy, text="", font=("Arial", font_size, "bold"), fill="white")
        self.after(33, self.update_animation)

    def run_audio(self):
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
            print("🎤 Inizia a parlare (CTRL+C per uscire)...\n")
            rec = vosk.KaldiRecognizer(model, 16000)
            while True:
                try:
                    data = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip().lower()
                    if text:
                        print(f"🗣️ Riconosciuto: {text}")
                        if text == "stop":
                            self.stop_tts()
                        elif not self.handle_command(text):
                            ollm(text)

                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "").lower()
                    if "stop" in partial and tts_in_progress:
                        self.stop_tts()

if __name__ == "__main__":
    app = JarvisGUI()
    app.mainloop()
