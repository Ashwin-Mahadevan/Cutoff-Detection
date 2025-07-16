import json
from typing import Literal, List, Iterator
from pydantic import BaseModel, ValidationError
from google.genai import Client, types

from dotenv import load_dotenv

load_dotenv()

import time
import librosa
import soundfile as sf
import numpy as np


def mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS string to seconds as float."""
    minutes, seconds = map(int, mmss.split(":"))
    return minutes * 60 + seconds


class TranscriptEntry(BaseModel):
    role: Literal["Main Agent", "Testing Agent"]
    content: str
    start_time: float
    end_time: float


class TranscriptSegment(BaseModel):
    messages: list[TranscriptEntry]


def load_transcript(id: int) -> Iterator[TranscriptEntry]:
    with open(f"data/case-{id}/transcript.json", "rb") as file:
        raw_transcript = json.load(file)
    for entry in raw_transcript:
        try:
            yield TranscriptEntry(
                role=entry["role"],
                content=entry["content"],
                start_time=entry["start_time"],
                end_time=entry["end_time"],
            )
        except (ValidationError, KeyError):
            continue


def join_transcript(id: int) -> str:
    transcript = list(load_transcript(id))
    lines = [f"{entry.role}: {entry.content}" for entry in transcript]
    return "\n\n".join(lines)


def segment_transcript(
    id: int, gap_threshold: float = 1.0
) -> Iterator[TranscriptSegment]:
    current_segment: List[TranscriptEntry] = []
    prev_time_sec = None
    for entry in load_transcript(id):
        if (
            prev_time_sec is not None
            and (entry.start_time - prev_time_sec) > gap_threshold
            and len(current_segment) > 0
        ):
            yield TranscriptSegment(messages=current_segment)
            current_segment = []
        current_segment.append(entry)
        prev_time_sec = entry.end_time

    if current_segment:
        yield TranscriptSegment(messages=current_segment)


client = Client()


def fmt_message(message: TranscriptEntry) -> str:
    role = "Agent" if message.role == "Main Agent" else "User"
    return f"{role}: {message.content}"


instructions = """
You are a quality assurance agent.
Occasionally, due to technical issues, your messages to the user may be cut off.
Your job is to determine if this has happened.
If your message to the user is cut off, return true, Otherwise, return false.
We only care about the agent's messages being cut off, not the user's messages.

--

User: Hello!
Agent: Hi! How can I help you today?

false
 
--

User: It's January tenth, twenty-twenty-one.
Agent: Got it, let me

true

--

Agent: No worries, I can repeat myself.

false

--

Agent: That's great to hear! I'm glad to hear that you're feeling better! Would you like

true

--

"""


def detect_cutoff(segment: TranscriptSegment) -> bool:
    """Detects if the segment contains a cutoff."""

    prompt = instructions + "\n".join(
        fmt_message(message) for message in segment.messages
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": Literal["true", "false"],
        },
    )

    assert response.parsed is not None
    assert isinstance(response.parsed, str)

    result = response.parsed.strip().lower()

    if result not in ["true", "false"]:
        raise ValueError(f"Invalid response: {result, type(result)}")

    return result == "true"


if __name__ == "__main__":
    for case_id in 1, 2, 3, 4, 5:
        for segment in segment_transcript(case_id):
            if not any(entry.role == "Main Agent" for entry in segment.messages):
                continue

            if detect_cutoff(segment):
                print(
                    f"Case {case_id} | First Time: {segment.messages[0].start_time} | Last Time: {segment.messages[-1].end_time}"
                )
                print("Transcript:")
                for entry in segment.messages:
                    print(f"  {entry.role}: {entry.content}")
                print("\n" + "-" * 40 + "\n")
