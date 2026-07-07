from google import genai
import json

from pydantic import BaseModel
from typing import List

client = genai.Client()

class Section(BaseModel):
    type: str
    occurrence: int
    text_anchor: str

class SongStructure(BaseModel):
    title: str
    sections: List[Section]

html_code = """"""
response = client.models.generate_content(
    model="gemini-3.5-flash",
contents=f"Extract the song details from this HTML: {html_code}, and return the result in JSON format. Make sure to thoroughly parse the HTML and extract all relevant information, including where a verse/chorus/bridge is, the occurrence (eg 1,2) and their corresponding text anchor: this is the exact first full line of lyrics in that section, excluding the chords, Make sure it's long enough to be uniquely identifiable from other verses. Do not include any other text.. The value should be present within the HTML and not made up. If a verse/chorus/bridge is not present, return an empty string for that value. The JSON should be given by SongStructure schema. If the song structure is not present, return an empty string for that value. Do a final check and delete any sections with an empty string for text_anchor",
    config={
        "response_mime_type": "application/json",
        "response_schema": SongStructure,
    }
)
print(response.text)