from dataclasses import dataclass, field
from typing import List

from . import HTMLPage
from . import MediaContent
from . import TextPage


@dataclass
class SiteContent:
    HTMLPages: List[HTMLPage] = field(default_factory=list)
    TextPages: List[TextPage] = field(default_factory=list)
    MediaContentList: List[MediaContent] = field(default_factory=list)