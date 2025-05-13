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
import re

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
                    import torch # type: ignore
                    # Check for GPU availability and set device
                    device = "cpu"  # Default to CPU
                    try:
                        if torch.cuda.is_available():
                            device = "cuda"
                        else:
                            print("No GPU available, using CPU for LLMLingua")
                    except:
                        print("Could not check for CUDA, using CPU for LLMLingua")
                        
                    try:
                        from llmlingua import PromptCompressor as LLMLingua  # type: ignore
                        print("Initializing LLMLingua...")
                        self.llm_lingua = LLMLingua(device_map=device)
                    except ImportError:
                        try:
                            from llmlingua import LLMLingua  # type: ignore
                            print("Initializing LLMLingua...")
                            self.llm_lingua = LLMLingua(device_map=device)
                        except ImportError:
                            print("Warning: LLMLingua package not found or incompatible. Text simplification will be disabled.")
                            self.llm_lingua = None
                except Exception as e:
                    print(f"Error initializing LLMLingua: {e}")
                    print("Text simplification will be disabled.")
                    self.llm_lingua = None

            # Initialize image captioning model
            print("Loading image captioning model...")
            try:
                self.image_processor = transformers.AutoImageProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base", 
                    use_fast=True
                )
                self.image_model = transformers.BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base", 
                    device_map="cpu"  # Force CPU usage
                )
            except Exception as e:
                print(f"Error initializing image model: {e}")
                self.image_processor = None
                self.image_model = None
        
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
    
    def get_media_description(self, media_url: str, img_element=None, media_element=None, parent_soup=None) -> str:
        """Generate enhanced description for media content using alt tags, surrounding text, and ML models."""
        try:
            filename = os.path.basename(media_url)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Collect information from HTML elements if available
            alt_text = ""
            title_text = ""
            figcaption_text = ""
            nearby_text = ""
            
            # Get alt text, title, and nearby context
            element = img_element or media_element
            if element:
                # Get alt text
                if element.get('alt'):
                    alt_text = element.get('alt').strip()
                
                # Get title attribute
                if element.get('title'):
                    title_text = element.get('title').strip()
                    
                # Look for figcaption in parent figure
                parent_figure = element.find_parent('figure')
                if parent_figure and parent_figure.find('figcaption'):
                    figcaption_text = parent_figure.find('figcaption').get_text().strip()
                
                # Get surrounding paragraphs for context
                if parent_soup:
                    # Find the closest paragraph or heading
                    prev_sibling = element.find_previous_sibling(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if prev_sibling:
                        nearby_text = prev_sibling.get_text().strip()
                    
                    # If no previous sibling with text, try parent's previous sibling
                    if not nearby_text and element.parent:
                        parent_prev = element.parent.find_previous_sibling(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                        if parent_prev:
                            nearby_text = parent_prev.get_text().strip()
            
            # If we have good HTML metadata, use it directly
            html_description = ""
            if alt_text:
                html_description += f"{alt_text}"
            if title_text and title_text != alt_text:
                html_description += f" {title_text}"
            if figcaption_text and figcaption_text not in html_description:
                html_description += f" {figcaption_text}"
            
            # If we have a good description from HTML, return it
            if len(html_description.strip()) > 10:  # Reasonable length
                return html_description.strip()
                
            # Try to extract meaning from filenames for icons and common elements
            if file_ext == '.svg' or 'icon' in filename.lower():
                # Remove file extension, then split by common separators
                base_name = os.path.splitext(filename)[0]
                words = re.split(r'[-_\s]', base_name)
                words = [w for w in words if w and not w.isdigit()]  # Remove empty or numeric parts
                
                # Check for common icon patterns
                icon_types = {
                    'calendar': 'Calendar',
                    'check': 'Checkmark',
                    'fill': 'Filled',
                    'outline': 'Outlined',
                    'user': 'User',
                    'person': 'Person',
                    'mail': 'Email',
                    'phone': 'Phone',
                    'arrow': 'Arrow',
                    'star': 'Star',
                    'search': 'Search',
                    'home': 'Home',
                    'plus': 'Plus',
                    'minus': 'Minus',
                    'menu': 'Menu',
                    'download': 'Download',
                    'upload': 'Upload',
                    'share': 'Share',
                    'play': 'Play',
                    'pause': 'Pause',
                }
                
                description_parts = []
                for word in words:
                    # Check for exact matches in icon_types dictionary
                    if word.lower() in icon_types:
                        description_parts.append(icon_types[word.lower()])
                    # Check for partial matches
                    else:
                        for key, value in icon_types.items():
                            if key in word.lower():
                                description_parts.append(value)
                                break
                        else:
                            # If no match found in icon_types, use the word as is
                            if len(word) > 2:  # Avoid very short words
                                description_parts.append(word.capitalize())
                
                # Construct a meaningful description
                if description_parts:
                    icon_desc = " ".join(description_parts)
                    return f"Icon: {icon_desc}"
            
            # Try to download and analyze the media
            try:
                response = requests.get(media_url, stream=True, timeout=10)
                if response.status_code != 200:
                    if html_description:
                        return html_description
                    return f"Could not retrieve media content: {filename}"
                    
                # For images, use the captioning model
                img_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                if any(file_ext == ext for ext in img_exts):
                    try:
                        image = Image.open(io.BytesIO(response.content))
                        inputs = self.image_processor(images=image, return_tensors="pt")
                        outputs = self.image_model.generate(**inputs, max_length=50)
                        ml_description = self.image_processor.batch_decode(outputs, skip_special_tokens=True)[0]
                        
                        # Combine ML description with HTML context if available
                        if html_description:
                            combined_desc = f"{ml_description} - {html_description}"
                            return combined_desc
                        
                        # If the ML description is too generic, enhance it with filename or nearby text
                        generic_phrases = ["an image", "a picture", "an icon", "a graphic"]
                        if any(phrase in ml_description.lower() for phrase in generic_phrases):
                            # Try using nearby text first
                            if nearby_text and len(nearby_text) < 100:  # Keep it reasonably short
                                return f"{ml_description} - Context: {nearby_text}"
                            
                            # Otherwise use filename
                            words = re.split(r'[-_\s]', os.path.splitext(filename)[0])
                            words = [w.capitalize() for w in words if w and len(w) > 2 and not w.isdigit()]
                            if words:
                                return f"{ml_description} ({' '.join(words)})"
                            
                        return ml_description
                        
                    except Exception as e:
                        # If captioning fails but we have HTML metadata, use that
                        if html_description:
                            return html_description
                            
                        # Otherwise fall back to filename-based description
                        words = re.split(r'[-_\s]', os.path.splitext(filename)[0])
                        words = [w.capitalize() for w in words if w and len(w) > 2 and not w.isdigit()]
                        if words:
                            return f"Image showing: {' '.join(words)}"
                        
                        # Last resort: use nearby text if available
                        if nearby_text:
                            return f"Image related to: {nearby_text[:100]}..."
                            
                        # Very last resort: basic info
                        content_type = response.headers.get('Content-Type', 'unknown')
                        size_kb = len(response.content) / 1024
                        return f"Image file: {filename} ({size_kb:.1f} KB)"
                    
                # For SVGs, we need special handling
                elif file_ext == '.svg':
                    # If we have HTML metadata, use that
                    if html_description:
                        return html_description
                    
                    # Try to extract any text from SVG
                    svg_content = response.content.decode('utf-8', errors='ignore')
                    if '<text' in svg_content:
                        # Extract text elements from SVG
                        soup = BeautifulSoup(svg_content, 'xml')
                        text_elements = soup.find_all('text')
                        if text_elements:
                            svg_texts = ' '.join(text.get_text() for text in text_elements)
                            if svg_texts.strip():
                                return f"SVG graphic with text: {svg_texts}"
                    
                    # Use nearby text as context if available
                    if nearby_text:
                        return f"SVG graphic related to: {nearby_text[:100]}..."
                    
                    # Fall back to filename-based description
                    words = re.split(r'[-_\s]', os.path.splitext(filename)[0])
                    readable_name = ' '.join(w.capitalize() for w in words if w and len(w) > 1)
                    if readable_name:
                        return f"SVG graphic: {readable_name}"
                    else:
                        return f"SVG graphic: {filename}"
                        
                # For other media types
                else:
                    # If we have HTML metadata, use that
                    if html_description:
                        return html_description
                    
                    # Otherwise return improved info
                    content_type = response.headers.get('Content-Type', 'unknown')
                    size_bytes = len(response.content)
                    size_kb = size_bytes / 1024
                    
                    # Use nearby text if available
                    context = ""
                    if nearby_text:
                        context = f" - Related to: {nearby_text[:100]}"
                    
                    # Classify media type more meaningfully
                    media_category = "Media file"
                    if 'audio' in content_type:
                        media_category = "Audio file"
                    elif 'video' in content_type:
                        media_category = "Video file" 
                    elif 'pdf' in content_type or file_ext == '.pdf':
                        media_category = "PDF document"
                    elif 'text' in content_type:
                        media_category = "Text document"
                    
                    # Extract meaningful name from filename
                    name_part = os.path.splitext(filename)[0]
                    words = re.split(r'[-_\s]', name_part)
                    readable_name = ' '.join(w.capitalize() for w in words if w and len(w) > 1)
                    
                    if readable_name:
                        return f"{media_category}: {readable_name} ({size_kb:.1f} KB){context}"
                    else:
                        return f"{media_category}: {filename} ({size_kb:.1f} KB){context}"
            
            except Exception as e:
                # If all else fails, use HTML metadata if available
                if html_description:
                    return html_description
                
                # Or use nearby text if available
                if nearby_text:
                    return f"Media related to: {nearby_text[:100]}"
                
                return f"Media content (description failed: {str(e)[:100]})"
        
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
        """Extract media content URLs from the page with improved descriptions."""
        # Process images
        for img in soup.find_all('img', src=True):
            src = img.get('src', '')
            if src:
                abs_url = urljoin(parent_url, src)
                if self.should_process_media(abs_url) and abs_url not in self.media_urls:
                    self.media_urls.add(abs_url)
                    # Pass the img element to get alt tags and context
                    description = self.get_media_description(abs_url, img_element=img, parent_soup=soup)
                    media_content = MediaContent(
                        url=abs_url,
                        media_type="image",
                        description=description,
                        parent_url=parent_url
                    )
                    self.site_content.add_media(media_content)
        
        # Process video, audio, and other media
        for tag_name in ['video', 'audio', 'source', 'iframe']:
            for tag in soup.find_all(tag_name):
                src = tag.get('src', '')
                if src:
                    abs_url = urljoin(parent_url, src)
                    if self.should_process_media(abs_url) and abs_url not in self.media_urls:
                        self.media_urls.add(abs_url)
                        # Pass the element for context
                        description = self.get_media_description(abs_url, media_element=tag, parent_soup=soup)
                        media_content = MediaContent(
                            url=abs_url,
                            media_type=tag_name,
                            description=description,
                            parent_url=parent_url
                        )
                        self.site_content.add_media(media_content)
    
    def crawl(self, url, parent_url="", depth=0):
        """Crawl a URL to given depth and collect content."""
        # Only check if we've visited this URL in *this* run
        if depth > self.max_depth or url in self.visited_urls:
            return
        
        # Mark this URL as visited in this run
        self.visited_urls.add(url)
        print(f"Crawling ({depth}/{self.max_depth}): {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Process HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            
            # Create HTMLPage object
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
            self.extract_media(soup, url)
            
            # Follow links if we haven't reached max depth
            if depth < self.max_depth:
                for link in html_page.links:
                    if link not in self.visited_urls:  # Only check against current run's visited URLs
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