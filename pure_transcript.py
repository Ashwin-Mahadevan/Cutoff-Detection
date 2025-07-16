import json
from typing import Literal, List, Iterator
from pydantic import BaseModel, ValidationError
from google.genai import Client

from dotenv import load_dotenv
load_dotenv()

import time

def mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS string to seconds as float."""
    minutes, seconds = map(int, mmss.split(":"))
    return minutes * 60 + seconds

class TranscriptEntry(BaseModel):
    role: Literal["Main Agent", "Testing Agent"]
    content: str
    time: str  # MM:SS format

class TranscriptSegment(BaseModel):
    messages: list[TranscriptEntry]


def load_transcript(id: int) -> Iterator[TranscriptEntry]:
    with open(f'data/case-{id}/transcript.json', 'rb') as file:
        raw_transcript = json.load(file)
    for entry in raw_transcript:
        try:
            # Parse and keep only the 'time' field (MM:SS)
            yield TranscriptEntry(
                role=entry["role"],
                content=entry["content"],
                time=entry["time"]
            )
        except (ValidationError, KeyError):
            continue

def join_transcript(id: int) -> str:
    transcript = list(load_transcript(id))
    lines = [f"{entry.role}: {entry.content}" for entry in transcript]
    return "\n\n".join(lines)


def segment_transcript(id: int, gap_threshold: float = 1.0) -> Iterator[TranscriptSegment]:
    current_segment: List[TranscriptEntry] = []
    prev_time_sec = None
    for entry in load_transcript(id):
        entry_time_sec = mmss_to_seconds(entry.time)
        if (
            prev_time_sec is not None
            and (entry_time_sec - prev_time_sec) > gap_threshold
            and len(current_segment) > 0
        ):
            yield TranscriptSegment(messages=current_segment)
            current_segment = []
        current_segment.append(entry)
        prev_time_sec = entry_time_sec

    if current_segment:
        yield TranscriptSegment(messages=current_segment)

client = Client()

def detect_cutoff(segment: TranscriptSegment) -> bool:
    ''' Detects if the segment contains a cutoff. '''
    # Join the segment into a readable transcript
    transcript_text = "\n".join(f"{entry.role}: {entry.content}" for entry in segment.messages)
    instructions = '''
    You are a quality assurance agent working for a telephone company.
    Occasionally, due to technical issues, phone connections may be temporarily lost.
    Our testing agent is calling a customer to test the phone system to see if it is working properly.
    You will be given a transcript for a segment of this call.
    Your job is to determine if the segment contains a point where the connection was lost due to a technical issue (a cutoff).
    Do not include cases where a speaker simply finishes speaking, is interrupted by the other speaker, or where the call ends naturally.
    Only return true if there is a clear technical cutoff in the segment.
    Respond with only 'true' or 'false'.
    In cases where the Testing Agent is speaking, return false.
    '''
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-06-17",
        contents=[instructions, transcript_text],
        config={
            'response_mime_type': 'text/plain',
        }
    )

    if response.text is not None:
        answer = response.text.strip().lower()
        return answer == 'true'
    return False


if __name__ == "__main__":
    for case_id in 1, 2, 3, 4, 5:
        for segment in segment_transcript(case_id):
            if detect_cutoff(segment):
                print(f"Case {case_id} | First Time: {segment.messages[0].time} | Last Time: {segment.messages[-1].time}")
                print("Transcript:")
                for entry in segment.messages:
                    print(f"  {entry.role}: {entry.content}")
                print("\n" + "-"*40 + "\n")

            time.sleep(7)
