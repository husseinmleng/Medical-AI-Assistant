from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def text_to_speech_to_file(text: str, output_path: Path = Path("output.mp3"), model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
    """
    Converts the given text into speech using OpenAI's TTS API and saves it to a file.
    """
    client = OpenAI()  # Assumes your OPENAI_API_KEY is set in the environment

    # Use OpenAI's audio/speech endpoint with streaming support
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(output_path)
    print(f"Saved TTS output to: {output_path}")

if __name__ == "__main__":
    sample_text = ".  رفعتلك الصورة, حلل الاشاعة   وطمني   على   النتيجة"
    output_file = Path("x_ray.wav")
    text_to_speech_to_file(sample_text, output_file)
