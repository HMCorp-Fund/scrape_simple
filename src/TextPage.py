from dataclasses import dataclass


@dataclass
class TextPage:
    RootUrl: str
    ParentUrl: str
    Url: str
    DepthLevel: int
    PageText: str