from dotenv import load_dotenv
load_dotenv()

import json
from google.genai import Client, types
from typing import Literal
from pydantic import BaseModel


client = Client()

with open("data/transcript.json", 'rb') as file:
    transcript = json.load(file)

with open("data/audio.wav", 'rb') as file:
    audio_bytes = file.read()

audio_part = types.Part.from_bytes(
    data=audio_bytes,
    mime_type="audio/wav"
)


class SingleCutoffFoundResponse(BaseModel):
    found: Literal["true"]
    timestamp: str

class SingleCutoffNotFoundResponse(BaseModel):
    found: Literal["false"]


def find_cutoff_single(audio_bytes: bytes):

    instructions = '''
    You are a quality assurance agent working for a telephone company.
    Occasionally, due to technical issues, the connection may be temporarily lost.
    You will be given a recording of the phone call.
    Your job is to determine if and when the connection was lost.
    If multiple cutoffs exist, return the first one.
    Do not include timestamps where a speaker simply finishes speaking.
    Do not include timestamps where one speaker interrupts the other.
    Do not include timestamps where the phone call ends.
    We are only interested in timestamps where they are cut off by a technical issue, not when they are interrupted by the other speaker.
    '''

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[instructions, audio_part],
        config={
            'response_mime_type': 'application/json',
            'response_schema': SingleCutoffFoundResponse | SingleCutoffNotFoundResponse,
        }
    )

    return response.text


class MultipleCutoffResponse(BaseModel):
    timestamp: str

def find_cutoff_multiple(audio_bytes: bytes):
    instructions = '''
    You are a quality assurance agent working for a telephone company.
    Occasionally, due to technical issues, the connection may be temporarily lost.
    You will be given a recording of the phone call.
    Your job is to create a list of timestamps where the connection was lost, if any exist.
    Do not include timestamps where a speaker simply finishes their sentence.
    Do not include timestamps where one speaker interrupts the other.
    Do not include timestamps where the phone call ends.
    We are only interested in timestamps where they are cut off by a technical issue, not when they are interrupted by the other speaker.
    '''

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[instructions, audio_part],
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[MultipleCutoffResponse],
        }
    )

    return response.text


if __name__ == "__main__":
    print("Single Cutoff:")
    print(find_cutoff_single(audio_bytes))
    print()

    print("Multiple Cutoff:")
    print(find_cutoff_multiple(audio_bytes))
    print()
