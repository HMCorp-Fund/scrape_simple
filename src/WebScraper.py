import time
import requests # type: ignore
from urllib.parse import urlparse, urljoin
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup # type: ignore
from . import SiteContent
from . import HTMLPage
from . import TextPage
from . import TorManager
import re

class WebScraper:
    def __init__(self, root_url: str, max_depth: int, simplify: bool = False, use_existing_tor: bool = True, 
                 preloaded_models: Optional[Dict[str, Any]] = None):
        self.root_url = root_url
        self.max_depth = max_depth
        self.visited_urls = set()
        self.domain = urlparse(root_url).netloc
        self.site_content = SiteContent()
        self.tor_manager = TorManager()
        self.use_existing_tor = use_existing_tor
        
    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain as root_url."""
        parsed_url = urlparse(url)
        return parsed_url.netloc == self.domain or parsed_url.netloc == ''
    
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
                text_page = TextPage(
                    url=url,
                    title=title,
                    content=text_content,
                    simplified_content=text_content,  # Same as content without simplification
                    parent_url=parent_url
                )
                self.site_content.add_text_page(text_page)
            
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