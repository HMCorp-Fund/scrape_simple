import time
import random
import requests # type: ignore
from urllib.parse import urlparse, urljoin, unquote
import os.path
from typing import List, Optional, Dict, Any, Set
from bs4 import BeautifulSoup # type: ignore
from . import SiteContent
from . import HTMLPage
from . import TextPage
from . import MediaContent
from . import TorManager
import re
import os
import io
import stem.control

class WebScraper:
    def __init__(self, root_url: str, max_depth: int, use_existing_tor: bool = True, 
                 simplify_ru: bool = False, min_media_size: int = 10240,
                 ai_describe_media: bool = False, skip_media: bool = False,
                 max_retries: int = 3):
        self.root_url = root_url
        self.max_depth = max_depth
        self.visited_urls = set()
        self.processed_media_urls = set()
        self.domain = urlparse(root_url).netloc
        self.site_content = SiteContent()
        self.tor_manager = TorManager()
        self.use_existing_tor = use_existing_tor
        self.simplify_ru = simplify_ru
        self.ru_simplifier = None  # Initialize to None by default
        
        # Initialize Russian text simplifier if needed
        if self.simplify_ru:
            try:
                from .utils import RussianTextSimplifier
                self.ru_simplifier = RussianTextSimplifier()
                print("Russian text simplification enabled")
            except Exception as e:
                print(f"Warning: Could not initialize Russian text simplification: {e}")
                print("Russian text simplification will be disabled")
                self.simplify_ru = False
        
        self.min_media_size = min_media_size  # Minimum media size in bytes
        self.ai_describe_media = ai_describe_media
        self.image_captioner = None
        self.skip_media = skip_media  # Flag to control media extraction
        self.max_retries = max_retries  # Number of retries for failed requests
        
        # Generate random user agents to rotate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:94.0) Gecko/20100101 Firefox/94.0'
        ]
        
        # Initialize AI image captioner only if needed and media extraction is enabled
        if self.ai_describe_media and not self.skip_media:
            try:
                self._initialize_image_captioner()
                print("AI image description enabled")
            except Exception as e:
                print(f"Warning: Could not initialize AI image description: {e}")
                self.ai_describe_media = False

    def _initialize_image_captioner(self):
        """Initialize the image captioning model."""
        try:
            from transformers.models.blip import BlipProcessor
            from transformers.models.blip import BlipForConditionalGeneration
            from PIL import Image # type: ignore
            
            # Store these as class attributes for use later
            self.Image = Image
            # Explicitly set use_fast=True to use the faster processor
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            # If processor returns a tuple, unpack it
            if isinstance(self.processor, tuple):
                self.processor = self.processor[0]
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.image_captioner = True
            print("AI image captioning model loaded successfully")
        except Exception as e:
            print(f"Error loading image captioning model: {e}")
            print("Please install the required packages: pip install transformers pillow torch")
            self.image_captioner = None
            raise

    def generate_ai_description(self, image_url: str) -> str:
        """Generate a description of the image using AI."""
        if not self.image_captioner:
            return ""
            
        try:
            # Download the image
            response = requests.get(image_url, timeout=10, stream=True)
            if response.status_code != 200:
                return ""
                
            # Load as PIL image
            image = self.Image.open(io.BytesIO(response.content))
            
            # Generate caption
            inputs = self.processor(image, return_tensors="pt")
            output = self.model.generate(**inputs, max_length=30)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            print(f"Error generating AI description for {image_url}: {e}")
            return ""

    def _has_cyrillic(self, text):
        """Check if text contains Cyrillic characters (for Russian detection)"""
        return bool(re.search('[а-яА-Я]', text))
    
    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain as root_url."""
        parsed_url = urlparse(url)
        return parsed_url.netloc == self.domain or parsed_url.netloc == ''
    
    def get_media_file_size(self, url: str) -> int:
        """Get the file size of a media URL in bytes."""
        try:
            # Get browser-like headers
            headers = self.get_request_headers()
            
            # Use HEAD request with headers to efficiently get content length
            head = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            if head.status_code == 200 and 'content-length' in head.headers:
                return int(head.headers['content-length'])
            
            # If HEAD request doesn't have content-length, try a small GET request
            get = requests.get(url, headers=headers, timeout=10, stream=True)
            if get.status_code == 200 and 'content-length' in get.headers:
                return int(get.headers['content-length'])
                
            return 0
        except Exception as e:
            print(f"Error checking size of {url}: {e}")
            return 0
    
    def get_filename_from_url(self, url: str) -> str:
        """Extract a clean filename from a URL."""
        parsed = urlparse(url)
        path = unquote(parsed.path)  # Handle URL encoding
        filename = os.path.basename(path)
        
        # If filename is empty or just a slash, use the last path component
        if not filename or filename == '/':
            parts = [p for p in path.split('/') if p]
            if parts:
                filename = parts[-1]
            else:
                return "unnamed_file"
                
        # Strip query parameters if present
        if '?' in filename:
            filename = filename.split('?')[0]
            
        return filename
    
    def get_media_description(self, url: str, img_element=None, parent_soup=None) -> str:
        """Generate a meaningful description for media content."""
        # First try AI description if enabled (for images only)
        if self.ai_describe_media:
            ai_description = self.generate_ai_description(url)
            if ai_description:
                return ai_description
        
        # Try alt text for images
        if img_element and img_element.get('alt'):
            alt_text = img_element.get('alt').strip()
            if alt_text and alt_text.lower() != "image" and len(alt_text) > 1:
                return alt_text
                
        # Try title attribute
        if img_element and img_element.get('title'):
            title_text = img_element.get('title').strip()
            if title_text:
                return title_text
        
        # Try figcaption if it exists
        if img_element and parent_soup:
            figure = img_element.find_parent('figure')
            if figure:
                figcaption = figure.find('figcaption')
                if figcaption and figcaption.text.strip():
                    return figcaption.text.strip()
        
        # Use filename as fallback
        filename = self.get_filename_from_url(url)
        
        # Clean up the filename
        if filename:
            # Remove extension
            filename = os.path.splitext(filename)[0]
            # Replace dashes and underscores with spaces
            filename = filename.replace('-', ' ').replace('_', ' ')
            # Capitalize words
            filename = ' '.join(word.capitalize() for word in filename.split())
            
            if filename and len(filename) > 1:
                return filename
        
        # Last resort
        return "Media file"
    
    def extract_media(self, soup: BeautifulSoup, page_url: str):
        """Extract and process media files above minimum size."""
        # Track new media found in this page
        media_found = 0
        
        # 1. Process all image tags
        for img in soup.find_all('img'):
            # Check all possible image attributes
            for attr in ['src', 'data-src', 'data-original', 'data-lazy-src']:
                if img.get(attr):
                    img_url = urljoin(page_url, img[attr])
                    if img_url not in self.processed_media_urls:
                        self.processed_media_urls.add(img_url)
                        
                        # Check file size
                        file_size = self.get_media_file_size(img_url)
                        if file_size >= self.min_media_size:
                            description = self.get_media_description(img_url, img_element=img, parent_soup=soup)
                            
                            # Create MediaContent object
                            media_content = MediaContent(
                                url=img_url,
                                media_type="image",
                                description=description,
                                parent_url=page_url
                            )
                            
                            self.site_content.add_media(media_content)
                            media_found += 1
                    break  # Found an image source, no need to check others
            
            # Handle srcset attribute
            if img.get('srcset'):
                srcset = img['srcset']
                # Extract the highest resolution image from srcset
                highest_res_url = None
                highest_res = 0
                
                # Parse the srcset attribute
                srcset_parts = re.findall(r'([^\s,]+)(?:\s+(\d+)w)?(?:,\s*)?', srcset)
                for url, width in srcset_parts:
                    if width and int(width) > highest_res:
                        highest_res = int(width)
                        highest_res_url = url
                    elif not width and not highest_res_url:  # Default if no width specified
                        highest_res_url = url
                
                if highest_res_url:
                    img_url = urljoin(page_url, highest_res_url)
                    if img_url not in self.processed_media_urls:
                        self.processed_media_urls.add(img_url)
                        
                        # Check file size
                        file_size = self.get_media_file_size(img_url)
                        if file_size >= self.min_media_size:
                            description = self.get_media_description(img_url, img_element=img, parent_soup=soup)
                            
                            # Create MediaContent object
                            media_content = MediaContent(
                                url=img_url,
                                media_type="image",
                                description=description,
                                parent_url=page_url
                            )
                            
                            self.site_content.add_media(media_content)
                            media_found += 1
        
        # 2. Process video elements
        for video in soup.find_all('video'):
            if video.get('src'):
                video_url = urljoin(page_url, video['src'])
                if video_url not in self.processed_media_urls:
                    self.processed_media_urls.add(video_url)
                    
                    # Check file size
                    file_size = self.get_media_file_size(video_url)
                    if file_size >= self.min_media_size:
                        description = video.get('title') or video.get('alt') or self.get_filename_from_url(video_url)
                        
                        # Create MediaContent object
                        media_content = MediaContent(
                            url=video_url,
                            media_type="video",
                            description=description,
                            parent_url=page_url
                        )
                        
                        self.site_content.add_media(media_content)
                        media_found += 1
            
            # Check source elements within video
            for source in video.find_all('source'):
                if source.get('src'):
                    video_url = urljoin(page_url, source['src'])
                    if video_url not in self.processed_media_urls:
                        self.processed_media_urls.add(video_url)
                        
                        # Check file size
                        file_size = self.get_media_file_size(video_url)
                        if file_size >= self.min_media_size:
                            description = source.get('title') or video.get('title') or self.get_filename_from_url(video_url)
                            
                            # Create MediaContent object
                            media_content = MediaContent(
                                url=video_url,
                                media_type="video",
                                description=description,
                                parent_url=page_url
                            )
                            
                            self.site_content.add_media(media_content)
                            media_found += 1
        
        # 3. Look for CDN patterns in HTML
        cdn_patterns = [
            r'https?://images\..*?/cdn-cgi/imagedelivery/[^"\')\s]+',
            r'https?://.*?\.cloudfront\.net/[^"\')\s]+\.(jpe?g|png|gif|svg|webp)',
            r'https?://.*?\.amazonaws\.com/[^"\')\s]+\.(jpe?g|png|gif|svg|webp)'
        ]
        
        html = str(soup)
        for pattern in cdn_patterns:
            for match in re.finditer(pattern, html):
                img_url = match.group(0)
                # Clean up URL if it has trailing quotes or syntax
                img_url = re.sub(r'["\')\s].*$', '', img_url)
                
                if img_url not in self.processed_media_urls:
                    self.processed_media_urls.add(img_url)
                    
                    # Check file size
                    file_size = self.get_media_file_size(img_url)
                    if file_size >= self.min_media_size:
                        description = self.get_filename_from_url(img_url)
                        
                        # Create MediaContent object
                        media_content = MediaContent(
                            url=img_url,
                            media_type="image",
                            description=description,
                            parent_url=page_url
                        )
                        
                        self.site_content.add_media(media_content)
                        media_found += 1
                        
        # 4. Check for elements with background images in style attributes
        elements_with_bg = soup.find_all(lambda tag: tag.has_attr('style') and 
                                        ('background' in tag['style'] or 'url(' in tag['style']))
        for element in elements_with_bg:
            style = element['style']
            urls = re.findall(r'url\([\'"]?([^\'";\)]+)', style)
            for extracted_url in urls:
                img_url = urljoin(page_url, extracted_url)
                if img_url not in self.processed_media_urls:
                    self.processed_media_urls.add(img_url)
                    
                    # Check file size
                    file_size = self.get_media_file_size(img_url)
                    if file_size >= self.min_media_size:
                        description = element.get('alt') or element.get('title') or self.get_filename_from_url(img_url)
                        
                        # Create MediaContent object
                        media_content = MediaContent(
                            url=img_url,
                            media_type="image",
                            description=description,
                            parent_url=page_url
                        )
                        
                        self.site_content.add_media(media_content)
                        media_found += 1
        
        if media_found > 0:
            print(f"Found {media_found} media files above minimum size on {page_url}")
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract readable text content from BeautifulSoup object with improved non-Latin support."""
        # Remove script, style and other non-content elements
        for element_to_remove in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element_to_remove.decompose()
        
        # Handle character encoding issues - this is critical for Cyrillic text
        try:
            # Use a direct approach to extract text
            content = []
            
            # Try to get main content first
            main_content = soup.find(['main', 'article', 'div', 'body'])
            
            # Get all significant text elements
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            
            # Process each text element
            for elem in text_elements:
                # Skip empty elements
                if not elem.text.strip():
                    continue
                
                # Skip elements in navigation, footer, etc.
                parent_nav = elem.find_parent(['nav', 'footer', 'header'])
                if parent_nav:
                    continue
                
                # Skip elements likely to be navigation
                classes = elem.get('class', [])
                if isinstance(classes, list):
                    class_str = ' '.join(str(c) for c in classes).lower()
                    if any(term in class_str for term in ['nav', 'menu', 'footer', 'header']):
                        continue
                
                # Get text content with proper encoding
                text = elem.get_text(strip=True)
                
                # Only add non-empty content
                if text and len(text) > 1:
                    content.append(text)
            
            # If we haven't found enough text, try getting content from divs
            if not content or sum(len(c) for c in content) < 100:
                for div in soup.find_all('div'):
                    # Skip empty divs
                    if not div.text.strip():
                        continue
                    
                    # Skip divs likely to be navigation or menus
                    classes = div.get('class', [])
                    if isinstance(classes, list):
                        class_str = ' '.join(str(c) for c in classes).lower()
                        if any(term in class_str for term in ['nav', 'menu', 'footer', 'header']):
                            continue
                    
                    # Get text and add if substantial
                    text = div.get_text(strip=True)
                    if text and len(text) > 30:  # Only add if reasonable length
                        content.append(text)
            
            # Combine the content
            full_text = '\n\n'.join(content)
            
            # Clean up the text without destroying unicode characters
            # Replace multiple whitespace with a single space
            full_text = re.sub(r'[ \t]+', ' ', full_text)
            # Replace multiple newlines with two newlines
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            
            return full_text
        except Exception as e:
            print(f"Error during text extraction: {e}")
            return ""
    
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

    def get_new_tor_identity(self):
        """Get a new Tor identity (IP address) by requesting a new circuit."""
        try:
            with stem.control.Controller.from_port(port=self.tor_manager.control_port) as controller:
                controller.authenticate()
                controller.signal(stem.Signal.NEWNYM)
                # Wait for the new identity to be established
                time.sleep(5)
                print("New Tor identity established")
                return True
        except Exception as e:
            print(f"Error getting new Tor identity: {e}")
            return False
    
    def get_request_headers(self):
        """Generate browser-like headers for HTTP requests."""
        user_agent = random.choice(self.user_agents)
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        return headers
    
    def crawl(self, url, parent_url="", depth=0):
        """Crawl a URL to given depth and collect content."""
        # Only check if we've visited this URL in *this* run
        if depth > self.max_depth or url in self.visited_urls:
            return
        
        # Mark this URL as visited in this run
        self.visited_urls.add(url)
        print(f"Crawling ({depth}/{self.max_depth}): {url}")
        
        # Add a small random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # Implement retry logic with Tor IP rotation
        for attempt in range(self.max_retries):
            try:
                # Get fresh headers for each attempt
                headers = self.get_request_headers()
                
                # Make the request with headers
                response = requests.get(url, headers=headers, timeout=30)
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
                
                # Extract media files only if media extraction is not skipped
                if not self.skip_media:
                    self.extract_media(soup, url)
            
                # Extract text content
                text_content = self.extract_text(soup)
                if text_content:
                    # Apply Russian simplification if enabled AND the simplifier exists
                    simplified_content = text_content
                    if self.simplify_ru and hasattr(self, 'ru_simplifier') and self.ru_simplifier and self._has_cyrillic(text_content):
                        try:
                            original_length = len(text_content)
                            simplified_content = self.ru_simplifier.simplify_text(text_content)
                            new_length = len(simplified_content)
                            
                            # Check if simplification produced reasonable results
                            if simplified_content and new_length >= original_length * 0.5:
                                print(f"Applied Russian text simplification for {url}")
                            else:
                                print(f"Russian simplification produced poor results, using original text")
                                simplified_content = text_content
                        except Exception as e:
                            print(f"Error applying Russian text simplification: {e}")
                            simplified_content = text_content
                    
                    text_page = TextPage(
                        url=url,
                        title=title,
                        content=text_content,
                        simplified_content=simplified_content,
                        parent_url=parent_url
                    )
                    self.site_content.add_text_page(text_page)
                
                # Follow links if we haven't reached max depth
                if depth < self.max_depth:
                    for link in html_page.links:
                        if link not in self.visited_urls:  # Only check against current run's visited URLs
                            time.sleep(1)  # Be nice to the server
                            self.crawl(link, url, depth + 1)
                
                # If we've reached here, the request was successful
                break
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') else 0
                
                # For 403/429 errors, try rotating IP and retrying
                if status_code in (403, 429) and attempt < self.max_retries - 1:
                    print(f"Received {status_code} error. Attempt {attempt+1}/{self.max_retries}, rotating Tor identity...")
                    self.get_new_tor_identity()
                    continue
                print(f"Error crawling {url}: {e}")
                break
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                break
    
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