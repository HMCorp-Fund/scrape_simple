class MediaContent:
    """Represents media content (images, videos, etc.)."""
    def __init__(self, url, media_type="", description="", parent_url=""):
        self.url = url
        self.media_type = media_type
        self.description = description
        self.parent_url = parent_url
        
    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "url": self.url,
            "media_type": self.media_type,
            "description": self.description,
            "parent_url": self.parent_url
        }