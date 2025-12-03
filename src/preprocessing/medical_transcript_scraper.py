#!/usr/bin/env python3
"""
Script to extract medical transcription reports from mtsamples.com
Reads URLs from a CSV file with a 'url' column and saves all reports to output CSV

Usage: python extract_mtsample.py --input <input_csv_file> --output <output_csv_file>
Example: python extract_mtsample.py --input mtsamples_urls.csv --output english_medical_transcripts.csv
"""

import requests
from bs4 import BeautifulSoup
import sys
import csv
import re
import argparse


def extract_report_from_url(url):
    """
    Extract medical transcription report content from mtsamples.com URL
    
    Args:
        url: Full URL to the mtsamples.com report page
        
    Returns:
        Dictionary containing report data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Fetch the page
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize report data
        report_data = {
            'url': url,
            'medical_specialty': '',
            'sample_name': '',
            'description': '',
            'keywords': '',
            'report_content': ''
        }
        
        # Extract Medical Specialty and Sample Name
        h1_elem = soup.find('h1', style=lambda x: x and 'font-size: 1.25rem' in x)
        if h1_elem:
            # Get specialty
            specialty_link = h1_elem.find('a', href=lambda x: x and 'browse.asp?type=' in x)
            if specialty_link:
                report_data['medical_specialty'] = specialty_link.get_text(strip=True)
            
            # Get sample name
            full_text = h1_elem.get_text()
            if 'Sample Name:' in full_text:
                sample_name = full_text.split('Sample Name:')[1].strip()
                report_data['sample_name'] = sample_name
        
        # Extract Description
        h2_elem = soup.find('h2', style=lambda x: x and 'font-size: 1rem' in x)
        if h2_elem:
            desc_text = h2_elem.get_text()
            if 'Description:' in desc_text:
                description = desc_text.split('Description:')[1].split('(Medical Transcription')[0].strip()
                report_data['description'] = description
        
        # Extract Keywords
        keywords_div = soup.find('div', class_='mt-5 mb-2')
        if keywords_div:
            keywords_text = keywords_div.get_text()
            if 'Keywords:' in keywords_text:
                keywords = keywords_text.split('Keywords:')[1].strip()
                # Clean up keywords
                keywords = re.sub(r'\s+', ' ', keywords).strip(' ,')
                report_data['keywords'] = keywords
        
        # Extract Report Content
        # Find the main content area
        hilight_div = soup.find('div', class_='hilightBold')
        if hilight_div:
            # Get all text between the header and the "See More Samples" section
            content_html = str(hilight_div)
            
            # Find where actual report starts (after description)
            report_start = content_html.find('<hr />')
            if report_start != -1:
                content_html = content_html[report_start:]
            
            # Find where report ends (before "See More Samples")
            report_end = content_html.find('See More Samples')
            if report_end != -1:
                content_html = content_html[:report_end]
            
            # Parse this section
            content_soup = BeautifulSoup(content_html, 'html.parser')
            
            # Remove script tags, ads, and navigation
            for element in content_soup.find_all(['script', 'ins', 'a', 'div']):
                if element.name == 'a' and 'Go Back to' in element.get_text():
                    element.decompose()
                elif element.name == 'div' and 'alert' in element.get('class', []):
                    element.decompose()
                elif element.name == 'ins':
                    element.decompose()
                elif element.name == 'script':
                    element.decompose()
            
            # Get text content
            text_content = content_soup.get_text(separator='\n')
            # Clean up the text
            lines = text_content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip only unwanted content
                if any(skip in line for skip in [
                    'See More Samples',
                    'Go Back to',
                    'Keywords:',
                    'View this sample in Blog format'
                ]):
                    continue
                cleaned_lines.append(line)
            
            # Add blank line BEFORE lines that start with capital letter and end with :
            standardized_lines = []
            for i, line in enumerate(cleaned_lines):
                # Check if line starts with capital letter and ends with colon
                if line and len(line) > 0 and line[0].isupper() and line.endswith(':'):
                    # Add blank line before header (unless it's the first line or previous line is already blank)
                    if standardized_lines and standardized_lines[-1] != '':
                        standardized_lines.append('')
                standardized_lines.append(line.strip())
            
            report_content = '\n'.join(standardized_lines)
            if '(Medical Transcription Sample Report)' in report_content:
                report_content = report_content.split('(Medical Transcription Sample Report)')[1]
            
            report_content = report_content.strip()
            
            report_data['report_content'] = report_content
        
        return report_data
        
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing content: {e}", file=sys.stderr)
        return None


def print_report(report_data):
    """Pretty print the report data"""
    if not report_data:
        print("No report data to display")
        return
    
    print("\n" + "="*80)
    print("MEDICAL TRANSCRIPTION REPORT")
    print("="*80)
    print(f"\nURL: {report_data['url']}")
    print(f"\nMedical Specialty: {report_data['medical_specialty']}")
    print("\n" + "-"*80)
    print("REPORT CONTENT:")
    print("-"*80)
    print(report_data['report_content'])
    print("\n" + "="*80)


def save_report(report_data, output_file, write_header=False):
    """Append report to a CSV file with three columns"""
    if not report_data:
        print("No report data to save")
        return
    
    mode = 'w' if write_header else 'a'
    
    with open(output_file, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if this is the first report
        if write_header:
            writer.writerow(['medical_transcript', 'report_specialty', 'url'])
        
        # Write data row
        writer.writerow([
            report_data['report_content'],
            report_data['medical_specialty'].lower(),
            report_data['url']
        ])


def main():
    """Main function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract medical transcription reports from mtsamples.com',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_mtsample.py --input urls.csv --output reports.csv
  python extract_mtsample.py -i urls.csv -o reports.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='mtsamples_urls.csv',
        help='Input CSV file containing URLs (default: mtsamples_urls.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='english_medical_transcripts.csv',
        help='Output CSV file for extracted reports (default: english_medical_transcripts.csv)'
    )
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output

    urls = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'url' not in reader.fieldnames:
                print("Error: Input CSV must have a 'url' column", file=sys.stderr)
                sys.exit(1)
            
            for row in reader:
                if row['url'].strip():
                    urls.append(row['url'].strip())
        
        print(f"Found {len(urls)} URLs to process")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process each URL
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(urls)}: {url}")
        print('='*80)
        
        # Extract report
        report_data = extract_report_from_url(url)
        
        if report_data:
            # Save to file (write header only for first report)
            write_header = (i == 1)
            save_report(report_data, output_file, write_header=write_header)
            successful += 1
            print(f"✓ Successfully saved report {i}")
        else:
            print(f"✗ Failed to extract report {i}", file=sys.stderr)
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nAll reports saved to: {output_file}")
    print('='*80)


if __name__ == "__main__":
    main()