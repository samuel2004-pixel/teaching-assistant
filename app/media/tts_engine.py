import os
import uuid
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

AUDIO_DIR = "data/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

def text_to_speech(text: str) -> str:
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # 🔒 Force non-streaming audio generation
    audio_bytes = client.text_to_speech.generate(
        voice=VOICE_ID,
        text=text,
        model="eleven_monolingual_v1",
        voice_settings={
            "stability": 0.92,
            "similarity_boost": 0.95,
            "style": 0.15,
            "use_speaker_boost": True
        }
    )

    # ✅ audio_bytes is GUARANTEED bytes
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filepath
