"""
============================================================
  🤖 RoboBot 1.0 — Speech-to-Speech + PDF RAG
============================================================
  Organisation: ROBOTWALA — Igniting Young Minds for Tomorrow's Tech
  Address:       201, Apollo Avenue, Old Palasia, Indore, M.P., India
  Website:       www.robotwala.org

  Stack:
    STT      → Groq Whisper (whisper-large-v3)
    LLM      → Groq LLaMA 3.3 70B
    TTS      → Microsoft Edge TTS (edge-tts, free)
    RAG      → Local PDF + TF-IDF + Cosine Similarity

  RAG Design:
    • PDF auto-loaded from project folder at startup
      Expected filename: finalrobotwala_Profile.pdf
    • Chunked: 300-word sliding window, 50-word overlap
    • Indexed: TF-IDF (sklearn) — zero network, ~1ms retrieval
    • Top-3 chunks injected into system prompt silently
    • Falls back to LLM general knowledge if no good match

  Interruption pipeline: preserved from original AcroBot design
============================================================
"""

import os
import asyncio
import tempfile
import queue
import time
import re
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

# ── RAG imports ────────────────────────────────────────────
import pdfplumber                                              # pip install pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer  # pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

STT_MODEL  = "whisper-large-v3"
CHAT_MODEL = "llama-3.3-70b-versatile"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 150
INTERRUPTION_THRESHOLD = 0.06
NOISE_FLOOR = 0.02
MIN_INTERRUPT_SECS = 1.2
END_OF_SPEECH_SECS = 0.7

ENERGY_THRESHOLD     = 0.018
SILENCE_AFTER_SPEECH = 0.8
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.05

IDLE_TIMEOUT         = 10.0
IDLE_POLL_TIMEOUT    = 30.0

WAKE_WORDS = ["hello", "hey", "hello robobot", "hey robobot", "robobot",
              "robotwala", "hello robotwala", "hey robotwala"]

# CHANGE 1: Stop words — only these trigger interruption
STOP_WORDS = ["stop", "stop it", "ruk", "ruko", "bas", "band karo", "chup"]

# ── RAG config ─────────────────────────────────────────────
PDF_FILENAME       = "finalrobotwala_Profile.pdf"   # place this file next to the script
RAG_CHUNK_WORDS    = 300
RAG_CHUNK_OVERLAP  = 50
RAG_TOP_K          = 3
RAG_MIN_SCORE      = 0.07   # slightly permissive for a focused org profile

# ──────────────────────────────────────────────
#  SYSTEM PROMPTS
# ──────────────────────────────────────────────

# CHANGE 2: Removed "institute", CHANGE 3: Allow general answers
SYSTEM_EN_BASE = (
    "Your name is RoboBot. You are the official AI assistant of ROBOTWALA — "
    "an AI, Robotics, Drone, 3D Printing, Coding and STEM education company "
    "based at 201, Apollo Avenue, Old Palasia, Indore, Madhya Pradesh, India. "
    "The company was co-founded by Prof. Imran Baig and is mentored by "
    "Dr. Zaheeruddin Babar. "
    "Help students, parents, schools, colleges, and businesses with questions "
    "about Robotwala's courses, workshops, services, leadership, philosophy, "
    "location, digital presence, and any other Robotwala-related queries. "
    "You can also answer general knowledge questions, science, technology, "
    "and educational questions when asked. "
    "Keep responses concise and conversational — 2 to 3 sentences maximum. "
    "No bullet points or markdown. "
    "Never mention documents, PDFs, retrieval, or sources. "
    "Answer naturally as if you simply know the information."
)

# CHANGE 2: Removed "institute", CHANGE 3: Allow general answers
SYSTEM_HI_BASE = (
    "Aapka naam RoboBot hai. Aap ROBOTWALA ke official AI assistant hain — "
    "yeh ek AI, Robotics, Drone, 3D Printing, Coding aur STEM education company hai, "
    "jo 201, Apollo Avenue, Old Palasia, Indore, Madhya Pradesh mein sthit hai. "
    "Is company ke co-founder Prof. Imran Baig hain aur Dr. Zaheeruddin Babar "
    "Investor & Mentor hain. "
    "Students, parents, schools, colleges aur businesses ko Robotwala ke courses, "
    "workshops, services, leadership, philosophy, location aur kisi bhi "
    "Robotwala se judi jaankari mein madad karein. "
    "Aap general knowledge, science, technology aur educational sawaalon ka jawab bhi de sakte hain. "
    "Hamesha Roman/Latin script mein jawab dein — Devanagari bilkul mat use karein. "
    "Apne uttar chhote aur batcheet ke andaz mein rakhein — 2 se 3 sentences maximum. "
    "Koi bullet points ya markdown nahi. "
    "Documents, PDF, ya retrieval ka zikr kabhi mat karo. "
    "Naturally jawab do jaise yeh jaankari tumhe pehle se pata hai."
)

# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    SPEAKING  = "speaking"

# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

history: dict = {
    "en": [],
    "hi": [],
}

last_user_text = ""

pygame.mixer.init()


# ══════════════════════════════════════════════
#  RAG SYSTEM
# ══════════════════════════════════════════════

class RobotwalaRAG:
    """
    Lightweight local RAG using TF-IDF + cosine similarity.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.loaded = False

    def load_pdf(self, pdf_path: str) -> bool:
        try:
            print(f"📄 Loading PDF: {pdf_path}")
            pages = []

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text.strip())

            if not pages:
                print("⚠️  PDF loaded but no text extracted — may be image-based.")
                return False

            full_text = "\n\n".join(pages)
            print(f"   Extracted {len(full_text)} characters from {len(pages)} pages")

            self._build_index(full_text)
            return True

        except Exception as e:
            print(f"⚠️  PDF load failed: {e}")
            return False

    def _build_index(self, text: str):
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        all_words = []
        for para in paragraphs:
            all_words.extend(para.split())

        chunks = []
        start = 0
        while start < len(all_words):
            end = min(start + RAG_CHUNK_WORDS, len(all_words))
            chunk = " ".join(all_words[start:end])
            if len(chunk.split()) >= 20:
                chunks.append(chunk)
            start += RAG_CHUNK_WORDS - RAG_CHUNK_OVERLAP

        self.chunks = chunks
        print(f"   Created {len(chunks)} chunks")

        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        self.loaded = True
        print("   ✅ TF-IDF index built — RAG ready")

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> str:
        if not self.loaded or not self.chunks:
            return ""

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = scores.argsort()[::-1][:top_k]

            relevant = []
            for idx in top_indices:
                if scores[idx] >= RAG_MIN_SCORE:
                    relevant.append(self.chunks[idx])

            if not relevant:
                return ""

            return "\n\n---\n\n".join(relevant)

        except Exception:
            return ""

    def build_system_prompt(self, query: str, lang: str) -> str:
        base = SYSTEM_HI_BASE if lang == "hi" else SYSTEM_EN_BASE
        context = self.retrieve(query)

        if not context:
            return base

        if lang == "hi":
            context_block = (
                "\n\nYeh jaankari tumhare paas already hai — isko naturally use karo:\n\n"
                + context
            )
        else:
            context_block = (
                "\n\nHere is relevant information you already know — use it naturally:\n\n"
                + context
            )

        return base + context_block


# Global RAG instance — loaded once at startup
rag = RobotwalaRAG()


def auto_load_pdf():
    # Search in multiple locations so the PDF is always found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, PDF_FILENAME),          # same folder as script
        r"D:\Robotwala\RW final.pdf",                   # hardcoded project folder
        os.path.join(script_dir, "RW final.pdf"),         # alternate name in script dir
        os.path.join(script_dir, "RW_final.pdf"),         # underscore variant
        r"D:\Robotwala\RW_final.pdf",                   # underscore in project folder
    ]

    for pdf_path in search_paths:
        if os.path.exists(pdf_path):
            print(f"✅ PDF found: {pdf_path}")
            rag.load_pdf(pdf_path)
            return

    print(f"⚠️  PDF not found. Searched in:")
    for p in search_paths:
        print(f"   • {p}")
    print("   RAG disabled — bot will rely on built-in Robotwala knowledge only.")


# ══════════════════════════════════════════════
#  AUDIO PIPELINE
# ══════════════════════════════════════════════

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()

    speech_buffer: list            = []
    pre_buffer:    list            = []
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock    = time.time()
                silence_start = None

                if not recording:
                    recording = True
                    speech_buffer = list(pre_buffer)

                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)

                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None


# ══════════════════════════════════════════════
#  TRANSCRIPTION
# ══════════════════════════════════════════════

def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            temperature=0,
            prompt=(
                "This conversation is about Robotwala — an AI and Robotics company in Indore. "
                "Topics include: robotics, AI, drones, 3D printing, coding, STEM, "
                "Prof. Imran Baig, courses, workshops, fees, admission, location. "
                "This conversation contains Hindi and English. "
                "Keep Hindi in Hindi words and English in English words. "
                "Do not translate Hindi to English. "
                "Use Roman Hindi for Hindi speech. "
                "Do not generate Urdu, Arabic, Tamil, Telugu, Bengali, or other languages. "
                "Ignore filler sounds like hmm, umm, ahh, mm-hmm, okay, accha. "
                "Transcribe fast speech accurately."
            ),
            response_format="json",
        )

    os.unlink(tmp_path)

    raw_text = (result.text or "").strip()
    text = clean_transcription(raw_text)

    if is_only_fillers(raw_text):
        return "", "en"

    lang = "en"

    for ch in raw_text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            lang = "hi"
            break
        if 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    hindi_words = {
        "hai", "haan", "nahi", "kya", "kaise",
        "kitna", "kitni", "accha", "theek",
        "courses", "fees", "admission",
        "placement", "robotwala", "branch", "mera", "meri", "mere",
        "aap", "tum", "hum",
        "kab", "ka", "ki",
        "mein", "main",
        "kar", "karna",
        "bol", "bata",
        "sakta", "sakti",
        "jana", "chahiye",
        "founder", "sikhna", "seekhna",
    }

    words = raw_text.lower().split()
    if any(word in hindi_words for word in words):
        lang = "hi"

    return text, lang


def clean_transcription(text: str) -> str:
    fillers = {
        "hmm", "hmmm", "hmmmm", "umm", "um", "ummm",
        "uh", "uhh", "uhhh", "ah", "ahh", "ahhh",
        "mmm", "mm", "huh", "huhh", "er", "erm", "like",
        "you know", "i mean", "sort of", "kind of",
        "accha", "acha", "achha", "haan", "han", "hmm na",
        "matlab", "toh", "to", "arey", "arre", "acha ji",
        "theek", "theek hai", "haanji", "ji", "bolo", "sunna", "dekho",
        "mm-hmm", "mhm", "hmm-hmm", "ahm", "ahmm", "ahmmm",
        "ahm-ahmm", "uh-huh", "huh-uh",
        "okay", "ok", "okk", "okayy", "right", "yeah",
        "yep", "ya", "yup",
        "hmmm...", "umm...", "uh...", "accha...", "okay...", "ahm-ahmm...",
    }

    words = text.split()
    cleaned = [w for w in words if w.lower().strip(".,!?") not in fillers]
    return " ".join(cleaned).strip()


def is_only_fillers(text: str) -> bool:
    if not text:
        return True
    cleaned = clean_transcription(text)
    return cleaned.strip() == ""


# ══════════════════════════════════════════════
#  WAKE WORD
# ══════════════════════════════════════════════

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# CHANGE 1: Helper to detect stop command
def is_stop_command(text: str) -> bool:
    lower = text.lower().strip().strip(".,!?")
    return any(stop == lower or lower.startswith(stop + " ") for stop in STOP_WORDS)


# ══════════════════════════════════════════════
#  AI REPLY — RAG context injection
# ══════════════════════════════════════════════

def get_ai_reply(user_text: str, lang: str) -> str:
    global last_user_text

    lang_history = history[lang]

    contextual_input = user_text

    if last_user_text:
        short_followups = [
            "haan", "nahi", "fees", "courses", "drone", "robot",
            "founder", "address", "contact", "website",
            "kya", "kitna", "kitni",
            "yes", "no", "what", "which", "who", "where", "when",
        ]
        if (
            len(user_text.split()) <= 5
            or any(w in user_text.lower() for w in short_followups)
        ):
            contextual_input = (
                f"Previous user query: {last_user_text}\n"
                f"Current follow-up query: {user_text}"
            )

    system = rag.build_system_prompt(contextual_input, lang)

    lang_history.append({
        "role": "user",
        "content": contextual_input
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            *lang_history,
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.5,
    )

    reply = response.choices[0].message.content.strip()

    lang_history.append({
        "role": "assistant",
        "content": reply
    })

    last_user_text = user_text
    return reply


# ══════════════════════════════════════════════
#  TTS / SPEAK
# ══════════════════════════════════════════════

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return TTS_VOICE_HI
        if 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts(text: str, path: str, voice: str):
    text = str(text).strip()
    if not text:
        text = "Sorry, I could not understand."
    text = text.replace("*", "").replace("#", "").replace("`", "")

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(path)
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
        fallback = edge_tts.Communicate(
            text="Sorry, there was a voice generation problem.",
            voice="en-US-JennyNeural",
        )
        await fallback.save(path)


# speak() returns (audio, is_new_question):
#   (None, False)       — finished naturally, no interruption
#   (audio, False)      — stopped by stop-word, no new question to answer
#   (audio, True)       — interrupted by a new question; caller should answer it
def speak(text: str, lang: str = "en"):
    voice = pick_voice(text, lang)
    print(f"   🔊 Voice → {voice}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        asyncio.run(_tts(text, tmp_path, voice))
    except Exception as e:
        print(f"⚠️ Async TTS Error: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None, False

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    blocksize = int(SAMPLE_RATE * CHUNK_SECS)
    speech_chunks = []
    recording = False
    silence_start = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        latency="low",
    ) as stream:

        while True:
            audio_chunk, overflowed = stream.read(blocksize)
            audio_chunk = np.where(np.abs(audio_chunk) < NOISE_FLOOR, 0, audio_chunk)
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

            if rms >= INTERRUPTION_THRESHOLD:
                if not recording:
                    # Do NOT stop playback yet — collect speech first
                    recording = True
                silence_start = None
                speech_chunks.append(audio_chunk)

            elif recording:
                speech_chunks.append(audio_chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= 0.55:
                    speech_duration = len(speech_chunks) * CHUNK_SECS
                    if speech_duration < 0.6:
                        speech_chunks = []
                        recording = False
                        silence_start = None
                        continue

                    # Transcribe what was said during playback
                    audio_arr = np.concatenate(speech_chunks, axis=0)
                    try:
                        heard_text, _ = transcribe(audio_arr)
                    except Exception:
                        heard_text = ""

                    print(f"   🎤 Heard during playback: {heard_text!r}")

                    # Stop playback in both cases (stop command OR new question)
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                    if is_stop_command(heard_text):
                        print("   🛑 Stop command detected — halting.")
                        return audio_arr, False
                    elif heard_text:
                        # New question detected — return audio for answering
                        print("   ❓ New question detected — switching.")
                        return audio_arr, True
                    else:
                        # Could not understand — keep playing (restart audio)
                        speech_chunks = []
                        recording = False
                        silence_start = None
                        pygame.mixer.music.load(tmp_path)
                        pygame.mixer.music.play()
                        continue

            elif not pygame.mixer.music.get_busy():
                break

            pygame.time.wait(15)

    pygame.mixer.music.unload()
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return None, False


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 60)
    print("  RoboBot 1.0 🤖  |  ROBOTWALA — Indore")
    print("  Igniting Young Minds for Tomorrow's Tech")
    print("=" * 60)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' or 'Robotwala' to wake up")
    print("    🔊 SPEAKING   — playing response; say 'Stop' to interrupt")
    rag_status = f"✅ {len(rag.chunks)} chunks" if rag.loaded else "❌ disabled"
    print(f"  RAG: {rag_status}")
    print(f"  PDF: {PDF_FILENAME}")
    print("  Ctrl+C to quit")
    print("=" * 60 + "\n")


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════

def main():
    auto_load_pdf()
    print_banner()

    state = State.LISTENING
    reply = ""
    lang  = "en"

    # CHANGE 4: Opening greeting changed to exactly what was requested
    speak(
        "Hello, I'm Robotwala AI Assistant. What can I help you with today?",
        lang="en",
    )

    try:
        while True:

            # ════════════════════════════════════════════════════
            #  IDLE
            # ════════════════════════════════════════════════════
            if state == State.IDLE:
                print(f"\n{state_label(state)}  — say 'Hello' or 'Robotwala' to activate...")

                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)

                if audio is None:
                    continue

                print("🔍 Checking for wake word...")
                wake_text, _ = transcribe(audio)
                print(f"   Heard: {wake_text!r}")

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    speak("Hello! I am listening. How can I help you with Robotwala?", lang="en")
                else:
                    print("   Not a wake word — staying idle.")

                continue

            # ════════════════════════════════════════════════════
            #  LISTENING
            # ════════════════════════════════════════════════════
            if state == State.LISTENING:
                print(f"\n{state_label(state)}  "
                      f"— silence for {int(IDLE_TIMEOUT)}s → idle")

                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    speak(
                        "Going to sleep now. Say 'Hello' or 'Robotwala' when you need me.",
                        lang="en",
                    )
                    continue

                print("🔍 Transcribing...")
                user_text, lang = transcribe(audio)

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")

                print("🤔 Thinking...")
                reply = get_ai_reply(user_text, lang)
                print(f"   AI  [{lang.upper()}] › {reply}")

                state = State.SPEAKING
                continue

            # ════════════════════════════════════════════════════
            #  SPEAKING
            # ════════════════════════════════════════════════════
            if state == State.SPEAKING:
                print(f"\n{state_label(state)}")

                interrupted_audio, is_new_question = speak(reply, lang)

                if interrupted_audio is not None and is_new_question:
                    # User asked a new question mid-answer — transcribe and answer it
                    print("🔍 Transcribing new question...")
                    user_text, lang = transcribe(interrupted_audio)
                    if user_text:
                        print(f"   You [{lang.upper()}] › {user_text}")
                        print("🤔 Thinking...")
                        reply = get_ai_reply(user_text, lang)
                        print(f"   AI  [{lang.upper()}] › {reply}")
                        state = State.SPEAKING
                    else:
                        state = State.LISTENING
                    continue

                # Stop command or finished naturally — go back to listening
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down RoboBot. Goodbye!")


if __name__ == "__main__":
    main()
