import base64
import json
import os
import wave
import uuid
import pika
import openai
import ffmpeg
import datetime
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from dotenv import load_dotenv
load_dotenv()

# API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")  # Your OpenAI key
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN") # Your Hugging Face token
)

RABBIT_URL = 'amqp://localhost'
QUEUE_NAME = 'audio_chunks'

CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM
FRAME_RATE = 44100
TRANSCRIPT_FILE = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
TEMP_AUDIO = "temp_audio.wav"
TEMP_SEGMENT = "temp_segment.wav"

def save_wav(data_bytes, path):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(FRAME_RATE)
        wf.writeframes(data_bytes)

def diarize_and_transcribe(wav_path):
    transcripts = []

    with ProgressHook() as hook:
        diarization = pipeline(wav_path, hook=hook)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        adjusted_end = turn.end + (1.1 - duration) if duration < 1.0 else turn.end

        ffmpeg.input(wav_path, ss=turn.start, to=adjusted_end).output(TEMP_SEGMENT).run(overwrite_output=True)

        if not os.path.exists(TEMP_SEGMENT) or os.path.getsize(TEMP_SEGMENT) == 0:
            continue

        with open(TEMP_SEGMENT, "rb") as audio_file:
            result = openai.Audio.transcribe(model="whisper-1", file=audio_file)

        transcripts.append({
            "speaker": speaker,
            "text": result["text"],
            "start": turn.start,
            "end": adjusted_end
        })

    return transcripts

def process_message(message_body):
    try:
        message = json.loads(message_body.decode())
        chunk_data = base64.b64decode(message["audio_chunk"])
        save_wav(chunk_data, TEMP_AUDIO)

        print(f"🔍 Diarizing + transcribing chunk {message['chunk_id']} from session {message['session_id']}...")

        transcripts = diarize_and_transcribe(TEMP_AUDIO)

        with open(TRANSCRIPT_FILE, "a") as f:
            for t in transcripts:
                line = f"[{t['speaker']}] {t['text'].strip()}"
                print(line)
                f.write(line + "\n")

    except Exception as e:
        print(f"❌ Error processing chunk: {e}")

    finally:
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)
        if os.path.exists(TEMP_SEGMENT):
            os.remove(TEMP_SEGMENT)

def consume_one_by_one():
    connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URL))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=False)

    print("🌀 Waiting for chunks from RabbitMQ...")

    try:
        while True:
            method_frame, _, body = channel.basic_get(queue=QUEUE_NAME, auto_ack=False)
            if body:
                process_message(body)
                channel.basic_ack(method_frame.delivery_tag)
            else:
                import time
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Exiting on keyboard interrupt.")
    finally:
        connection.close()

if __name__ == "__main__":
    consume_one_by_one()
