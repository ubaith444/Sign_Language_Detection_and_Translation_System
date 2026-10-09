import queue
import threading
import time
from typing import Optional
from src.config import TTS_RATE, TTS_VOLUME

class SpeechSynthesizer:
    """
    Offline Text-To-Speech engine utilizing pyttsx3 (SAPI5 on Windows).
    Executes in a dedicated background daemon worker thread to prevent
    blocking the video capture and user interface pipelines.
    """

    def __init__(self, rate: int = TTS_RATE, volume: float = TTS_VOLUME):
        self.rate = rate
        self.volume = volume
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = True
        self._start_worker()

    def _start_worker(self):
        """Starts background worker thread for TTS."""
        self.worker_thread = threading.Thread(target=self._run_tts_worker, daemon=True)
        self.worker_thread.start()

    def _run_tts_worker(self):
        """Worker loop processing speech queue."""
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[SpeechSynthesizer] Note: pyttsx3 local audio driver initialization: {e}")

        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.2)
                if text is None:
                    break

                text_clean = str(text).strip()
                if not text_clean:
                    self.speech_queue.task_done()
                    continue

                if engine is not None:
                    try:
                        engine.say(text_clean)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"[SpeechSynthesizer] Playback error: {e}")
                        # Re-initialize engine if SAPI5 COM thread invalidated
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty("rate", self.rate)
                            engine.setProperty("volume", self.volume)
                            engine.say(text_clean)
                            engine.runAndWait()
                        except Exception:
                            pass
                else:
                    # Fallback log output if driver not present
                    print(f"[SpeechSynthesizer Vocalization]: {text_clean}")

                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SpeechSynthesizer] Worker error: {e}")

    def speak(self, text: str, block: bool = False):
        """Queues text to be spoken aloud."""
        if not text or not str(text).strip():
            return

        self.speech_queue.put(str(text).strip())
        if block:
            self.speech_queue.join()

    def stop(self):
        """Stops the synthesizer."""
        self.is_running = False
        self.speech_queue.put(None)
