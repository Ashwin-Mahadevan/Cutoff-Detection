from dotenv import load_dotenv
load_dotenv()

import json
from google.genai import Client, types
from typing import Literal, List
from pydantic import BaseModel

class TranscriptEntry(BaseModel):
    role: Literal["Main Agent", "Testing Agent"]
    content: str
    start_time: float
    end_time: float

client = Client()

def load_audio(id: int):
    return client.files.upload(file=f'data/case-{id}/audio.wav')

def load_transcript(id: int) -> List[TranscriptEntry]:
    with open(f'data/case-{id}/transcript.json', 'r') as file:
        raw = json.load(file)
    entries = []
    for entry in raw:
        try:
            entries.append(TranscriptEntry(**entry))
        except Exception:
            continue
    return entries

def fmt_time(t: float) -> str:
    minutes = int(t // 60)
    seconds = int(t % 60)

    return f"{minutes}:{seconds:02d}"

def format_transcript(transcript: List[TranscriptEntry]):
    lines = [f"{fmt_time(entry.start_time)}-{fmt_time(entry.end_time)} {entry.role}: {entry.content}" for entry in transcript]
    return "\n".join(lines)

def find_cutoffs(audio: types.File, transcript_text: str):
    instructions = '''
    You are a quality assurance agent working for a telephone company.
    Occasionally, due to technical issues, the connection may be temporarily lost.
    You will be given a recording of the phone call and the full transcript with timestamps.
    Your job is to determine if and when the connection was lost due to a technical issue (a cutoff).
    Do not include timestamps where a speaker simply finishes speaking.
    Do not include timestamps where one speaker interrupts the other.
    We are only interested in timestamps where they are cut off by a technical issue, not when they are interrupted by the other speaker.
    Return a list of timestamps where the connection was lost due to a technical issue.
    If there aren't any, return an empty list.
    '''

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[instructions, transcript_text, audio],
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[str],
        }
    )

    assert response.text is not None

    response_json = json.loads(response.text)

    assert isinstance(response_json, list)
    assert all(isinstance(item, str) for item in response_json)

    return response_json

if __name__ == "__main__":
    for case_id in range(1, 6):
        print(f"=== Case {case_id} ===")
        audio = load_audio(case_id)
        transcript = load_transcript(case_id)
        transcript_text = format_transcript(transcript)

        result = find_cutoffs(audio, transcript_text)
        if result:
            print(f"  Cutoffs found at: {result}")
        else:
            print("  No cutoff found.")

        print()
