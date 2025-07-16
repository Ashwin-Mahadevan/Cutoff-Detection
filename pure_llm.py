from dotenv import load_dotenv
load_dotenv()

import json
from google.genai import Client, types
from typing import Literal
from pydantic import BaseModel


client = Client()

def load_case(id: int):
    with open(f'data/case-{id}/transcript.json', 'rb') as file:
        transcript = json.load(file)
    
    audio = client.files.upload(file=f'data/case-{id}/audio.wav')

    return transcript, audio

class SingleCutoffFoundResponse(BaseModel):
    found: Literal["true"]
    timestamp: str


class SingleCutoffNotFoundResponse(BaseModel):
    found: Literal["false"]


def find_cutoff_single(audio: types.File):

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[instructions, audio],
        config={
            'response_mime_type': 'application/json',
            'response_schema': SingleCutoffFoundResponse | SingleCutoffNotFoundResponse,
        }
    )

    assert response.text is not None
    # Parse and validate the response as JSON using Pydantic
    response_json = json.loads(response.text)
    if response_json.get("found") == "true":
        return SingleCutoffFoundResponse(**response_json)
    else:
        return SingleCutoffNotFoundResponse(**response_json)


class MultipleCutoffResponse(BaseModel):
    timestamp: str

def find_cutoff_multiple(audio: types.File):
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[instructions, audio],
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[MultipleCutoffResponse],
        }
    )

    assert response.text is not None
    # Parse and validate the response as JSON using Pydantic
    response_json = json.loads(response.text)
    return [MultipleCutoffResponse(**item) for item in response_json]


if __name__ == "__main__":
    for case_id in range(1, 6):
        print(f"=== Case {case_id} ===")
        transcript, audio = load_case(case_id)

        print("Single Cutoff:")
        single_result = find_cutoff_single(audio)
        if hasattr(single_result, 'found') and single_result.found == "true":
            print(f"  Cutoff found at: {single_result.timestamp}")
        else:
            print("  No cutoff found.")
        print()

        print("Multiple Cutoff:")
        multiple_results = find_cutoff_multiple(audio)
        if multiple_results:
            for idx, cutoff in enumerate(multiple_results, 1):
                print(f"  {idx}. {cutoff.timestamp}")
        else:
            print("  None found.")
        print()
