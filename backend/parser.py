import os
from pathlib import Path
from google import genai
from models import SongStructure
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))  # Load backend/.env.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(
    api_key=api_key
)


def parseSongStructure(page_text: str) -> SongStructure:
    """Parse the page text with Gemini."""

    prompt = f"""
Extract the song details from the following webpage text.
Rules:
- Ignore menus, adverts, comments and anything not part of the song.
- Extract the song title.
- Extract every Verse, Chorus, Bridge, Intro, Outro, Instrumental, Tag or Vamp that exists.
- occurrence should start at 1 and increment for repeated sections.
- text_anchor must be the exact first complete lyric line of that section.
- if a text_anchor for a VERSE is the same as another text_ancho of a previous VERSE include more of the lyrics until it is unique.
- Do NOT include chords.
- Do NOT invent information.
- Do NOT paraphrase.
- If a section has no valid text anchor, do not include it.
- Return ONLY valid JSON following the SongStructure schema.
- REMOVE DUPLICATE CHORUS/INSTRUMENTAL SECTIONS. If a chorus is repeated, only include the first occurrence.

Page Text:

{page_text}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SongStructure,
        },
    )

    if response.parsed is None:
        raise ValueError("Gemini returned no parsed song structure")

    return response.parsed