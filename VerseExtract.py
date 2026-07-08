from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

class SectionLandmark(BaseModel):
    type: str = Field(description="e.g., intro, verse, chorus, bridge")
    text_anchor: str = Field(description="The exact first line of lyric text for this section")

class SongStructure(BaseModel):
    sections: List[SectionLandmark]

client = genai.Client()

def parseSongStructure(page_text):
    prompt = (
        "Analyze this song sheet text. Extract every distinct structural section in order "
        "(Verse 1, Chorus, Verse 2, etc.) and identify a clean, exact line of text/lyrics "
        "at the very start of that section to act as a positioning anchor."
    )
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[prompt, page_text],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SongStructure,
        ),
    )
    
    return response.text  # <-- Clean JSON string returned to app.py