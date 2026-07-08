from pydantic import BaseModel
from typing import List


class Section(BaseModel):
    type: str
    occurrence: int
    text_anchor: str


class SongStructure(BaseModel):
    title: str
    sections: List[Section]