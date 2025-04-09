import os
import requests
from datetime import datetime
import re
from typing import Optional
import logging
from bs4 import BeautifulSoup

# Constants
GUIDELINES_FILE = 'coding_guidelines.md'
# Time format in the header of the guidelines file
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
# Default refresh interval in days (can be overridden by environment variable)
DEFAULT_REFRESH_INTERVAL_DAYS = 7
# Header pattern for finding the timestamp in the file
TIMESTAMP_HEADER_PATTERN = r'^<!-- Last updated: ([\d\- :]+) -->'

class GuidelinesManager:
    def __init__(self, jira_username: Optional[str] = None, jira_api_token: Optional[str] = None, confluence_url: Optional[str] = None):
        """
        Initialize the GuidelinesManager.
        
        Args:
            jira_username: The Jira/Confluence username (email)
            jira_api_token: The Jira/Confluence API token
            confluence_url: The URL to the Confluence page with coding guidelines
        """
        # Default to environment variables if not provided
        self.jira_username = jira_username or os.environ.get('JIRA_USERNAME')
        self.jira_api_token = jira_api_token or os.environ.get('JIRA_API_TOKEN')
        self.confluence_url = confluence_url or os.environ.get('CONFLUENCE_URL')
        
        # Get refresh interval from environment variable or use default
        try:
            self.refresh_interval_days = int(os.environ.get('REFRESH_INTERVAL_DAYS', DEFAULT_REFRESH_INTERVAL_DAYS))
            print(f"Guidelines refresh interval set to {self.refresh_interval_days} days")
        except ValueError:
            self.refresh_interval_days = DEFAULT_REFRESH_INTERVAL_DAYS
            print(f"Invalid REFRESH_INTERVAL_DAYS value, using default: {DEFAULT_REFRESH_INTERVAL_DAYS} days")
        
    def guidelines_exist(self) -> bool:
        """Check if the coding guidelines file exists."""
        return os.path.exists(GUIDELINES_FILE)
    
    def _extract_timestamp_from_file(self) -> Optional[datetime]:
        """
        Extract the timestamp from the guidelines file header.
        
        Returns:
            datetime object if timestamp was found, None otherwise
        """
        if not self.guidelines_exist():
            return None
            
        try:
            with open(GUIDELINES_FILE, 'r') as f:
                first_line = f.readline().strip()
                match = re.match(TIMESTAMP_HEADER_PATTERN, first_line)
                if match:
                    timestamp_str = match.group(1)
                    return datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
        except (IOError, ValueError) as e:
            logging.warning(f"Error extracting timestamp from guidelines file: {e}")
        
        return None
    
    def needs_refresh(self) -> bool:
        """
        Check if the guidelines need to be refreshed.
        
        Returns:
            True if guidelines don't exist or are older than the refresh interval,
            False otherwise
        """
        if not self.guidelines_exist():
            return True
            
        timestamp = self._extract_timestamp_from_file()
        if not timestamp:
            return True
            
        now = datetime.now()
        delta = now - timestamp
        
        return delta.days >= self.refresh_interval_days
    
    def fetch_from_confluence(self) -> Optional[str]:
        """
        Fetch guidelines content from Confluence.
        
        Returns:
            The guidelines content as a string, or None if fetching failed
        """
        if not all([self.jira_username, self.jira_api_token, self.confluence_url]):
            logging.warning("Missing Confluence credentials or URL. Cannot fetch guidelines.")
            return None
            
        try:
            # Extract page ID from URL if it's a Confluence cloud URL
            page_id = None
            if 'viewpage.action' in self.confluence_url:
                # For server/data center
                page_id_match = re.search(r'pageId=(\d+)', self.confluence_url)
                if page_id_match:
                    page_id = page_id_match.group(1)
            elif '/pages/' in self.confluence_url:
                # For cloud
                page_id_match = re.search(r'/pages/\d+/(\d+)', self.confluence_url)
                if page_id_match:
                    page_id = page_id_match.group(1)
            
            if not page_id:
                logging.error(f"Could not extract page ID from Confluence URL: {self.confluence_url}")
                return None
                
            # Determine if it's cloud or server/data center
            base_url = self.confluence_url.split('/display/')[0] if '/display/' in self.confluence_url else self.confluence_url.split('/wiki/')[0]
            
            # Create REST API URL
            if 'atlassian.net' in base_url:
                # Cloud
                api_url = f"{base_url}/wiki/rest/api/content/{page_id}?expand=body.storage"
            else:
                # Server/Data Center
                api_url = f"{base_url}/rest/api/content/{page_id}?expand=body.storage"
                
            response = requests.get(
                api_url,
                auth=(self.jira_username, self.jira_api_token),
                headers={"Accept": "application/json"}
            )
            
            if response.status_code != 200:
                logging.error(f"Failed to fetch from Confluence. Status code: {response.status_code}")
                logging.error(f"Response: {response.text[:500]}...")
                return None
                
            content_data = response.json()
            html_content = content_data['body']['storage']['value']
            
            # Convert HTML to Markdown
            return self._format_html_to_markdown(html_content)
            
        except Exception as e:
            logging.error(f"Error fetching guidelines from Confluence: {e}")
            return None
    
    def _format_html_to_markdown(self, html_content: str) -> str:
        """
        Format HTML content from Confluence to Markdown format.
        
        Args:
            html_content: HTML content from Confluence
            
        Returns:
            Formatted markdown content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Initialize markdown content with timestamp header
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        markdown = f"<!-- Last updated: {timestamp} -->\n\n# Coding Guidelines\n\n"
        
        # Process headings
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(tag.name[1])
            markdown += f"{'#' * level} {tag.get_text().strip()}\n\n"
            
            # Process the content until the next heading
            sibling = tag.next_sibling
            while sibling and not (hasattr(sibling, 'name') and sibling.name and sibling.name.startswith('h')):
                if hasattr(sibling, 'name'):
                    if sibling.name == 'p':
                        markdown += f"{sibling.get_text().strip()}\n\n"
                    elif sibling.name == 'ul':
                        for li in sibling.find_all('li', recursive=False):
                            markdown += f"* {li.get_text().strip()}\n"
                        markdown += "\n"
                    elif sibling.name == 'ol':
                        for i, li in enumerate(sibling.find_all('li', recursive=False), 1):
                            markdown += f"{i}. {li.get_text().strip()}\n"
                        markdown += "\n"
                    elif sibling.name == 'pre' or sibling.name == 'code':
                        code = sibling.get_text().strip()
                        markdown += f"```\n{code}\n```\n\n"
                    elif sibling.name == 'table':
                        # Process table headers
                        headers = sibling.find_all('th')
                        if headers:
                            header_row = "| " + " | ".join(th.get_text().strip() for th in headers) + " |\n"
                            separator = "| " + " | ".join(["---"] * len(headers)) + " |\n"
                            markdown += header_row + separator
                            
                        # Process table rows
                        for row in sibling.find_all('tr'):
                            cells = row.find_all('td')
                            if cells:
                                row_text = "| " + " | ".join(td.get_text().strip() for td in cells) + " |\n"
                                markdown += row_text
                        markdown += "\n"
                sibling = sibling.next_sibling
        
        return markdown
    
    def update_guidelines(self) -> bool:
        """
        Update the guidelines file with content from Confluence.
        
        Returns:
            True if the update was successful, False otherwise
        """
        guidelines_content = self.fetch_from_confluence()
        
        if not guidelines_content:
            logging.warning("Could not fetch guidelines from Confluence.")
            return False
            
        try:
            with open(GUIDELINES_FILE, 'w') as f:
                f.write(guidelines_content)
            logging.info(f"Successfully updated guidelines in {GUIDELINES_FILE}")
            return True
        except IOError as e:
            logging.error(f"Error writing guidelines to file: {e}")
            return False
    
    def check_and_update_if_needed(self) -> bool:
        """
        Check if guidelines need to be refreshed and update them if necessary.
        
        Returns:
            True if guidelines exist and are up to date (or were successfully updated),
            False otherwise
        """
        if self.needs_refresh():
            print(f"Guidelines need to be refreshed (older than {self.refresh_interval_days} days or don't exist)")
            return self.update_guidelines()
        else:
            print("Guidelines are up to date")
            return True 