import os
import re
import csv
import json
import tarfile
import sys # Added for command-line arguments
import xml.etree.ElementTree as ET
import urllib.request
from typing import List, Dict, Optional, Any, Tuple

import requests
import nltk
from requests.exceptions import RequestException

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)


# --- Configuration and Constants ---

class Config:
    """
    Configuration class for file paths and parameters.
    Default values are provided, but FILE_PATH can be overridden via sys.argv.
    """
    # Default file path (can be overridden by command-line argument)
    DEFAULT_FILE_PATH: str = 'french_case_reports_pmc_ids.txt' 
    FILE_PATH: str = DEFAULT_FILE_PATH # Actual value to be set in main()

    DOWNLOAD_DIR: str = 'PMC_tarballs'  # Directory for downloaded tarballs
    XML_DIR: str = 'XML_extracted'  # Directory for extracted XML files
    OUTPUT_CSV: str = 'pmc_extracted_data.csv'  # Final output CSV file

    # Global XML tag/element definitions
    P_NODES: List[str] = ['p', 'list-item', 'disp-quote', 'AbstractText']
    SEC_NODES: List[str] = ['sec', 'list']
    INLINE_ELEMENTS: List[str] = ['italic', 'bold', 'sup', 'strike', 'sub', 'sc', 'named-content', 'underline',
                                  'statement', 'monospace', 'roman', 'overline', 'styled-content']
    # Regex patterns
    LABEL_PATTERN: re.Pattern = re.compile(r'^[0-9]\.?[0-9]?\.?[0-9]?\.? ?')
    # Various whitespace characters
    SPACE_CHARS: str = r"[\u3000\u2009\u2002\u2003\u00a0\u200a\xa0\xa0]"
    PMC_PATTERN: re.Pattern = re.compile(r'\bPMC\d+\b')


# --- Utility Functions ---

def create_directory(directory: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Cleans text by normalizing whitespace, removing specific artifacts,
    and resolving common formatting issues.
    """
    text = re.sub(Config.SPACE_CHARS, ' ', text).replace(u'\u2010', '-').strip()
    text = re.sub(r" +", ' ', text)  # Collapse multiple spaces
    text = re.sub(r"\n+", "\n", text)  # Collapse multiple newlines

    # Regex patterns to clean up common empty or erroneous parenthetical/bracketed content
    patterns_to_remove = [
        r"\(,\s*and\s*\)", r"\(,\s*et\s*\)", r"\(\s*,\s*(,\s*)+\)",
        r"\(\s*,\s*\)", r"\[\s*,\s*\]", r"\(\s*et\s*\)", r"\[\s*et\s*\]",
        r"\(\s*and\s*\)", r"\[\s*and\s*\]", r"\(\s*\)", r"\[\s*\]",
        r"\(-?\)", r"\( A\)", r"\(A \)", r"\(A\)", r"\[-\]",
        r"\(,\s*\){2,}", r"\(\s*,\s*\)", r"\[\s*,\s*\]",
        r"\(\s*,\s*,\s*\)", r"\[\s*,\s*,\s*\]", r"\(A, B\)", r"\( A, B\)", r"\( A,B\)"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)

    text = re.sub(r",+", ",", text)  # Collapse multiple commas
    text = re.sub(r"\s*,\s*", ", ", text)  # Standardize space after commas

    # Final replacements for common artifacts
    text = text.replace('[, , , ]', "").replace('[, , ]', "").replace('[,]', "")
    text = text.replace('(, ', "(").replace(' ,)', ")").replace('(,)', '')

    return text.strip()


def decode_escaped_characters(text: str) -> str:
    """Decodes escaped characters (e.g., XML entities if needed) in text."""
    return text.encode('utf_8', 'ignore').decode('utf-8')


# --- Core Extractor Class ---

class PMCExtractor:
    """
    Manages the pipeline for downloading, extracting, and parsing PMC XML files.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def _get_tarball_link(self, pmcid: str) -> Optional[str]:
        """Fetch the FTP link for the tarball using the PMC utility API."""
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            link_element = root.find(".//link")
            if link_element is not None:
                href_attribute = link_element.get("href")
                return href_attribute
            print(f"-> Warning: No download link found in API response for {pmcid}.")
            return None
        except RequestException as e:
            print(f"-> Error: HTTP/Network error for {pmcid}: {e}")
            return None
        except ET.ParseError as e:
            print(f"-> Error: Failed to parse XML API response for {pmcid}: {e}")
            return None

    def _download_tarball(self, tarball_link: str, pmcid: str) -> Optional[str]:
        """Download one tarball file from a given link."""
        filename = f"{pmcid}.tar.gz"
        filepath = os.path.join(self.config.DOWNLOAD_DIR, filename)
        try:
            urllib.request.urlretrieve(tarball_link, filepath)
            return filepath
        except Exception as e:
            print(f"-> Error: Failed to download {pmcid}.tar.gz: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return None

    def download_tarballs(self) -> List[str]:
        """Main step 1: Download all tarballs based on PMCIDs in the input file."""
        print(f"Using input file: {self.config.FILE_PATH}")
        if not os.path.exists(self.config.FILE_PATH):
            print(f"Error: PMCID file '{self.config.FILE_PATH}' not found. Please provide a valid file.")
            return []

        with open(self.config.FILE_PATH, 'r', encoding='utf-8') as file:
            text = file.read()
        
        pmc_ids: List[str] = Config.PMC_PATTERN.findall(text)
        if not pmc_ids:
            print("Warning: No PMCIDs found in the input file.")
            return []
            
        print(f"Found {len(pmc_ids)} PMCIDs to process.")
        create_directory(self.config.DOWNLOAD_DIR)
        
        downloaded_files: List[str] = []
        for i, pmc_id in enumerate(pmc_ids):
            print(f"Processing [{i+1}/{len(pmc_ids)}]: {pmc_id}...", end='\r', flush=True)
            tarball_link = self._get_tarball_link(pmc_id)
            if tarball_link:
                filepath = self._download_tarball(tarball_link, pmc_id)
                if filepath:
                    downloaded_files.append(filepath)
        
        print(f"\n[1/4] Downloaded {len(downloaded_files)} tarballs successfully.")
        return downloaded_files

    def extract_xml_from_tar(self) -> None:
        """Main step 2: Extract .nxml files from downloaded tarballs."""
        if not os.path.exists(self.config.DOWNLOAD_DIR):
            print(f"Warning: Download directory '{self.config.DOWNLOAD_DIR}' not found. Skipping extraction.")
            return

        create_directory(self.config.XML_DIR)
        
        extracted_count = 0
        tar_files = [f for f in os.listdir(self.config.DOWNLOAD_DIR) if f.endswith('.tar.gz')]
        
        for filename in tar_files:
            tar_filepath = os.path.join(self.config.DOWNLOAD_DIR, filename)
            try:
                with tarfile.open(tar_filepath, "r:gz") as tar:
                    nxml_members = [m for m in tar.getmembers() if m.name.endswith('.nxml')]
                    if not nxml_members:
                        continue
                    
                    nxml_member = nxml_members[0]
                    pmcid = filename.split('.')[0]
                    extraction_path = os.path.join(self.config.XML_DIR, pmcid)
                    
                    tar.extract(nxml_member, path=extraction_path)
                    
                    original_name = os.path.join(extraction_path, nxml_member.name)
                    new_name = os.path.join(extraction_path, f"{pmcid}.nxml")
                    if os.path.exists(original_name):
                        os.rename(original_name, new_name)
                    
                    extracted_count += 1
            except tarfile.TarError as e:
                print(f"-> Error: Failed to open or extract tarball {filename}: {e}")
            except Exception as e:
                print(f"-> Error: An unexpected error occurred during extraction of {filename}: {e}")

        print(f"[2/4] Extracted XML from {extracted_count} tarballs.")


    def _get_element_text(self, element: ET.Element) -> str:
        """Recursively extracts text from a given XML element and its children."""
        text = element.text if element.text else ""

        for child in element:
            child_text = ""
            if child.tag in self.config.INLINE_ELEMENTS:
                child_text += child.text if child.text else ""
            
            if child.tag in self.config.SEC_NODES or child.tag in self.config.P_NODES:
                child_text += self._get_element_text(child) + ' '
            
            text += child_text
            text += child.tail if child.tail else ""

        return clean_text(text)

    def _get_section_title(self, sec_element: ET.Element) -> str:
        """Extracts the title from a section element."""
        title_element = sec_element.find("title")
        if title_element is not None:
            title_text = self._get_element_text(title_element)
            return Config.LABEL_PATTERN.sub('', title_text).strip()
        return ""

    def _parse_section(self, sec_element: ET.Element) -> Tuple[str, List[Dict[str, str]]]:
        """Parses a section element to extract its main text and recursively find subsections."""
        section_text_parts: List[str] = []
        subsections: List[Dict[str, str]] = []

        for child in sec_element:
            if child.tag in self.config.P_NODES:
                text = self._get_element_text(child)
                if text:
                    section_text_parts.append(text)
            elif child.tag in self.config.SEC_NODES:
                subsection_title = self._get_section_title(child)
                if not subsection_title:
                    continue
                    
                subsection_text, _ = self._parse_section(child) 
                
                subsections.append({
                    "title": subsection_title,
                    "text": subsection_text
                })
        
        main_text = "\n\n".join(section_text_parts)
        return main_text, subsections


    def extract_text_from_xml_folder(self) -> List[Dict[str, Any]]:
        """Main step 3: Extract structured text from XML files."""
        if not os.path.exists(self.config.XML_DIR):
            print(f"Error: XML directory '{self.config.XML_DIR}' not found. Cannot proceed with parsing.")
            return []

        extracted_data: List[Dict[str, Any]] = []
        parsed_count = 0
        
        for pmcid in os.listdir(self.config.XML_DIR):
            pmcid_path = os.path.join(self.config.XML_DIR, pmcid)
            if not os.path.isdir(pmcid_path):
                continue
                
            nxml_files = [f for f in os.listdir(pmcid_path) if f.endswith('.nxml')]
            if not nxml_files:
                continue

            file_path = os.path.join(pmcid_path, nxml_files[0])
            
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    xml_content = file.read()
                
                root_element = ET.fromstring(xml_content)
                body = root_element.find(".//body")

                if body is None:
                    continue

                sections = body.findall("./sec")
                next_section_data: Optional[Dict[str, Any]] = None
                found_intro = False
                
                for sec in sections:
                    title = self._get_section_title(sec)
                    lower_title = title.lower().strip()
                    
                    if lower_title in ("introduction", "intro"):
                        found_intro = True
                        continue
                    
                    if found_intro:
                        if lower_title in ("discussion", "conclusion", "references"):
                            break

                        if next_section_data is None:
                            main_text, subsections = self._parse_section(sec)
                            
                            next_section_data = {
                                "title": title,
                                "text": main_text,
                                "subsections": subsections
                            }
                            # Only capture the very first section post-intro (and its subsections)
                            break 
                            
                if next_section_data is not None:
                    extracted_data.append({
                        "pmcid": pmcid,
                        "case_report": next_section_data
                    })
                    parsed_count += 1
                    
            except FileNotFoundError:
                print(f"-> Error: XML file not found for {pmcid} at {file_path}")
            except ET.ParseError as e:
                print(f"-> Error: Failed to parse XML for {pmcid}: {e}")
            except Exception as e:
                print(f"-> Error: An unexpected error occurred while parsing {pmcid}: {e}")

        print(f"[3/4] Extracted text from {parsed_count} XML files.")
        return extracted_data

    def write_to_csv(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Main step 4: Flattens and writes extracted data to a CSV file.
        """
        processed_data: List[Dict[str, str]] = []
        
        for entry in data:
            case_report = entry.get('case_report', {})
            
            full_text = case_report.get('text', '')
            
            subsections = case_report.get('subsections', [])
            for subsection in subsections:
                if subsection.get('text'):
                    # Append subsection content with a clear separator
                    full_text += f"\n\n## {subsection['title']}\n" + subsection['text']
            
            final_text = decode_escaped_characters(full_text)
            
            if final_text.strip():
                processed_data.append({
                    'pmcid': entry['pmcid'],
                    'case_report_title': case_report.get('title', ''),
                    'case_report_text': final_text.strip()
                })

        with open(self.config.OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['pmcid', 'case_report_title', 'case_report_text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(processed_data)
        
        return processed_data


def main() -> None:
    """
    Main execution function.
    Handles command-line argument for the input file path.
    """
    config = Config()
    
    # Check for command-line argument for input file path
    if len(sys.argv) > 1:
        config.FILE_PATH = sys.argv[1]
    else:
        config.FILE_PATH = config.DEFAULT_FILE_PATH

    extractor = PMCExtractor(config)
    
    print("=" * 60)
    print("🔬 PMC XML Extraction Pipeline")
    print("=" * 60)
    
    # Step 1: Download
    downloaded_files = extractor.download_tarballs()
    if not downloaded_files and not os.path.exists(config.DOWNLOAD_DIR):
        print("\nExiting pipeline as no files were downloaded and download directory is empty.")
        return
        
    # Step 2: Extract XML
    extractor.extract_xml_from_tar()
    
    # Step 3: Extract Text
    extracted_data = extractor.extract_text_from_xml_folder()
    
    # Step 4: Write to CSV
    print("\n[4/4] Writing to CSV...")
    if extracted_data:
        final_data = extractor.write_to_csv(extracted_data)
        
        print(f"Data saved to: {config.OUTPUT_CSV}")
        print(f"Total files processed: {len(extracted_data)}")
        print(f"Final entries in CSV: {len(final_data)}")
    else:
        print("No parsable data extracted. CSV not created.")
    
    print("\n" + "=" * 60)
    print("Extraction Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
