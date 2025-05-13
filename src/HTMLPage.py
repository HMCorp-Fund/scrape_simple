from dataclasses import dataclass


@dataclass
class HTMLPage:
    RootUrl: str
    ParentUrl: str
    Url: str
    DepthLevel: int
    PageHTMLCode: str