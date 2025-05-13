class TextPage:
    """Represents a text page with its content."""
    def __init__(self, url, title="", content="", simplified_content="", parent_url=""):
        self.url = url
        self.title = title
        self.content = content
        self.simplified_content = simplified_content
        self.parent_url = parent_url
        
    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "simplified_content": self.simplified_content,
            "parent_url": self.parent_url
        }