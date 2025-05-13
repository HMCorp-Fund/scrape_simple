class HTMLPage:
    """Represents an HTML page with its content."""
    def __init__(self, url, title="", content="", links=None, parent_url=""):
        self.url = url
        self.title = title
        self.content = content
        self.links = links or []
        self.parent_url = parent_url
        
    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "links": self.links,
            "parent_url": self.parent_url
        }