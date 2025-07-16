import json
import soundfile as sf
from typing import Literal, List, Iterator
import librosa
from pydantic import BaseModel, ValidationError
from google.genai import Client, types

from dotenv import load_dotenv

load_dotenv()


class TranscriptEntry(BaseModel):
    role: Literal["Main Agent", "Testing Agent"]
    content: str
    start_time: float
    end_time: float


def load_transcript(id: int) -> Iterator[TranscriptEntry]:
    with open(f"data/case-{id}/transcript.json", "rb") as file:
        raw_transcript = json.load(file)

    assert isinstance(raw_transcript, list)

    for entry in raw_transcript:
        assert isinstance(entry, dict)

        try:
            yield TranscriptEntry(**entry)
        except (ValidationError, KeyError):
            continue


def join_transcript(id: int) -> str:
    transcript = list(load_transcript(id))
    lines = [f"{entry.role}: {entry.content}" for entry in transcript]
    return "\n\n".join(lines)


client = Client()


def fmt_message(message: TranscriptEntry) -> str:
    role = "User A" if message.role == "Main Agent" else "User B"
    return f"{role}: {message.content}"


potential_cutoff_instructions = """
You are a quality assurance agent for a messging application.
Occasionally, due to technical issues, the end of a message may be cut off.
Your job is to determine if this has happened.
If the message has been cut off AT THE VERY END, return true.
Otherwise, return false.

"""


def detect_potential_cutoff(message: TranscriptEntry) -> bool:
    """Detects if the segment contains a cutoff."""

    if message.role != "Main Agent":
        return False

    prompt = potential_cutoff_instructions + fmt_message(message)

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


def cut_audio(inpath: str, outpath: str, start_time: float, end_time: float):
    assert end_time > start_time

    # Load the audio file
    audio, sample_rate = librosa.load(inpath, sr=None)

    # Convert to mono
    audio = librosa.to_mono(audio)

    # Convert time to samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Clamp values to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    # Ensure we have a valid segment
    if start_sample >= end_sample:
        raise ValueError(
            "Invalid time range: start_time must be less than end_time and both must be within audio bounds"
        )

    # Extract the segment
    audio_segment = audio[start_sample:end_sample]

    # Save the trimmed audio
    sf.write(outpath, audio_segment, sample_rate)


validate_cutoff_instructions = """
You are a quality assurance agent for a voice call application.
To preserve anonymity, you will only examine a short segment of the call.
Your job is to determine if the audio cuts out in the middle of this segment.
You will be provided both a partial transcript and the partial audio file.
Return true if the audio cuts out in the middle of the segment, otherwise return false.
"""


if __name__ == "__main__":
    for case_id in 1, 2, 3, 4, 5:
        transcript = list(load_transcript(case_id))

        for i, message in enumerate(transcript):
            if detect_potential_cutoff(message):
                context_messages = transcript[max(0, i - 2) : i + 2]

                start_time = min(message.start_time for message in context_messages)
                end_time = max(message.end_time for message in context_messages)

                if i < 3:
                    start_time = 0

                if i > len(transcript) - 2:
                    end_time = float("inf")

                print(f"Potential Cutoff: Case {case_id} Message {i}")

                cut_audio(
                    f"data/case-{case_id}/audio.wav",
                    f"data/case-{case_id}/audio_{i}.wav",
                    start_time,
                    end_time,
                )

                audio_file = client.files.upload(
                    file=f"data/case-{case_id}/audio_{i}.wav",
                )

                partial_transcript = "Transcript: \n\n" + "\n".join(
                    fmt_message(message) for message in context_messages
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        f"Your job is to determine if the audio file has a missing segment. If it does, return true. Otherwise, return false.",
                        partial_transcript,
                        audio_file,
                    ],
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

                if result == "true":
                    print("cutoff confirmed")
