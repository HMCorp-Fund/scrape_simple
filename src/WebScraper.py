import os
import io
import time
import requests # type: ignore
from urllib.parse import urlparse, urljoin
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup # type: ignore
from PIL import Image # type: ignore
import transformers
from . import SiteContent
from . import HTMLPage
from . import TextPage
from . import MediaContent
from . import TorManager

# Update the LLMLingua import
try:
    from llmlingua import PromptCompressor as LLMLingua  # type: ignore # For newer versions
except ImportError:
    try:
        from llmlingua import LLMLingua  # type: ignore # For older versions
    except ImportError:
        print("Warning: LLMLingua package not found or incompatible. Text simplification will be disabled.")
        LLMLingua = None

class WebScraper:
    def __init__(self, root_url: str, max_depth: int, simplify: bool = False, use_existing_tor: bool = True, 
                 preloaded_models: Optional[Dict[str, Any]] = None):
        self.root_url = root_url
        self.max_depth = max_depth
        self.simplify = simplify
        self.visited_urls = set()
        self.media_urls = set()
        self.domain = urlparse(root_url).netloc
        self.site_content = SiteContent()
        self.tor_manager = TorManager()
        self.use_existing_tor = use_existing_tor
        
        # Use preloaded models if available
        if preloaded_models:
            print("Using preloaded models...")
            self.image_processor = preloaded_models.get("image_processor")
            self.image_model = preloaded_models.get("image_model")
            self.llm_lingua = preloaded_models.get("llm_lingua")
        else:
            # Initialize LLMLingua if simplification is enabled
            if self.simplify:
                try:
                    from llmlingua import PromptCompressor as LLMLingua  # type: ignore
                    print("Initializing LLMLingua...")
                    self.llm_lingua = LLMLingua()
                except ImportError:
                    try:
                        from llmlingua import LLMLingua  # type: ignore
                        print("Initializing LLMLingua...")
                        self.llm_lingua = LLMLingua()
                    except ImportError:
                        print("Warning: LLMLingua package not found or incompatible. Text simplification will be disabled.")
                        self.llm_lingua = None
            
            # Initialize image captioning model
            print("Loading image captioning model...")
            self.image_processor = transformers.AutoImageProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base", 
                use_fast=True
            )
            self.image_model = transformers.BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain as root_url."""
        parsed_url = urlparse(url)
        return parsed_url.netloc == self.domain or parsed_url.netloc == ''
    
    def should_process_media(self, url: str) -> bool:
        """Determine if the URL points to media content worth processing."""
        # Check file extension for common media types
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', 
                     '.mp4', '.webm', '.mov', '.pdf', '.mp3', '.wav']
        
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Skip tiny icons and web design elements
        if 'icon' in path or 'favicon' in path:
            return False
            
        return any(path.endswith(ext) for ext in extensions)
    
    def get_media_description(self, media_url: str) -> str:
        """Generate description for media content using the image captioning model."""
        try:
            response = requests.get(media_url, stream=True, timeout=10)
            if response.status_code != 200:
                return "Could not retrieve media content"
                
            # For images, use the captioning model
            if any(media_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                image = Image.open(io.BytesIO(response.content))
                inputs = self.image_processor(images=image, return_tensors="pt")
                outputs = self.image_model.generate(**inputs, max_length=30)
                description = self.image_processor.batch_decode(outputs, skip_special_tokens=True)[0]
                return description[:200]
            else:
                # For other media types, return basic info
                content_type = response.headers.get('Content-Type', 'unknown')
                size_bytes = len(response.content)
                size_kb = size_bytes / 1024
                return f"{os.path.basename(media_url)} - {content_type}, size: {size_kb:.1f} KB"
                
        except Exception as e:
            return f"Media content (description failed: {str(e)[:100]})"
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract readable text content from BeautifulSoup object."""
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.extract()
            
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Apply LLMLingua simplification if enabled
        if self.simplify and LLMLingua and text:
            try:
                compressed_text = self.llm_lingua.compress(text)
                return compressed_text
            except Exception as e:
                print(f"Error applying text simplification: {e}")
                return text
        return text

    def simplify_text(self, text):
        """Simplify text using LLMLingua if available."""
        if not hasattr(self, '_lingua') and LLMLingua is not None:
            self._lingua = LLMLingua()
        
        if hasattr(self, '_lingua'):
            try:
                compressed = self._lingua.compress_prompt(text)
                return compressed.get('compressed_prompt', text)
            except Exception as e:
                print(f"Text simplification failed: {e}")
        
        return text  # Return original text if simplification isn't available
    
    def extract_links(self, soup: BeautifulSoup, parent_url: str) -> List[str]:
        """Extract all links from the page and normalize them."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(parent_url, href)
            
            # Skip fragments, mailto, tel, javascript, etc.
            if '#' in absolute_url or any(protocol in absolute_url for protocol in ['mailto:', 'tel:', 'javascript:']):
                continue
                
            if self.is_same_domain(absolute_url):
                links.append(absolute_url)
        return links
    
    def extract_media(self, soup: BeautifulSoup, parent_url: str):
        """Extract media content URLs from the page."""
        # Process images
        for img in soup.find_all('img', src=True):
            src = img.get('src', '')
            if src:
                abs_url = urljoin(parent_url, src)
                if self.should_process_media(abs_url) and abs_url not in self.media_urls:
                    self.media_urls.add(abs_url)
                    description = self.get_media_description(abs_url)
                    self.site_content.MediaContentList.append(
                        MediaContent(Url=abs_url, Description=description)
                    )
        
        # Process video, audio, and other media
        media_tags = soup.find_all(['video', 'audio', 'source', 'iframe'])
        for tag in media_tags:
            src = tag.get('src', '')
            if src:
                abs_url = urljoin(parent_url, src)
                if self.should_process_media(abs_url) and abs_url not in self.media_urls:
                    self.media_urls.add(abs_url)
                    description = self.get_media_description(abs_url)
                    self.site_content.MediaContentList.append(
                        MediaContent(Url=abs_url, Description=description)
                    )
    
    def crawl(self, url, parent_url="", depth=0):
        """Crawl a URL to given depth and collect content."""
        if depth > self.max_depth or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        print(f"Crawling ({depth}/{self.max_depth}): {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Process HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            
            # Create HTMLPage object with correct parameter names
            html_page = HTMLPage(
                url=url,
                title=title,
                content=response.text,
                links=self.extract_links(soup, url),
                parent_url=parent_url
            )
            self.site_content.add_html_page(html_page)
            
            # Extract text content
            text_content = self.extract_text(soup)
            if text_content:
                simplified_content = ""
                if self.simplify and hasattr(self, 'llm_lingua') and self.llm_lingua:
                    simplified_content = self.simplify_text(text_content)
                    
                text_page = TextPage(
                    url=url,
                    title=title,
                    content=text_content,
                    simplified_content=simplified_content,
                    parent_url=parent_url
                )
                self.site_content.add_text_page(text_page)
            
            # Extract and process media content
            self.process_media(soup, url)
            
            # Follow links if we haven't reached max depth
            if depth < self.max_depth:
                for link in html_page.links:
                    if link not in self.visited_urls:
                        time.sleep(1)  # Be nice to the server
                        self.crawl(link, url, depth + 1)
                        
        except Exception as e:
            print(f"Error crawling {url}: {e}")
    
    def start(self):
        """Start the scraping process."""
        try:
            # Connect to Tor, using existing process if available
            self.tor_manager.start_tor(use_existing=self.use_existing_tor)
            
            # Begin crawling from the root URL
            self.crawl(self.root_url, "", 0)
            
            return self.site_content
            
        finally:
            # Always stop Tor when done (if we started it)
            self.tor_manager.stop_tor()