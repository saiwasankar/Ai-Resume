from typing import Literal

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    company: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=160)
    location: str = Field(default="", max_length=160)
    url: str = Field(default="", max_length=2048)
    description: str = Field(default="", max_length=20000)


class JobUpdate(BaseModel):
    status: Literal["saved", "applied", "interview", "offer", "rejected", "archived"]
