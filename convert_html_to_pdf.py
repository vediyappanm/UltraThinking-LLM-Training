"""
HTML to PDF Converter
Converts the ULTRATHINK documentation HTML file to PDF format
"""

from weasyprint import HTML
import os
from pathlib import Path

def convert_html_to_pdf(html_file_path, output_pdf_path=None):
    """
    Convert an HTML file to PDF
    
    Args:
        html_file_path: Path to the input HTML file
        output_pdf_path: Path for the output PDF file (optional)
    """
    # Validate input file
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")
    
    # Generate output path if not provided
    if output_pdf_path is None:
        html_path = Path(html_file_path)
        output_pdf_path = html_path.parent / f"{html_path.stem}.pdf"
    
    print(f"Converting HTML to PDF...")
    print(f"Input:  {html_file_path}")
    print(f"Output: {output_pdf_path}")
    
    # Convert HTML to PDF
    try:
        HTML(filename=html_file_path).write_pdf(output_pdf_path)
        print(f"✓ Conversion successful!")
        print(f"PDF saved to: {output_pdf_path}")
        
        # Print file size
        file_size = os.path.getsize(output_pdf_path) / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f} MB")
        
        return output_pdf_path
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        raise

if __name__ == "__main__":
    # Path to the HTML file
    html_file = r"c:\Users\poorn\OneDrive\Desktop\model building\UltraThinking-LLM-Training-main\UltraThinking-LLM-Training\ultrathink_doc (1).html"
    
    # Convert to PDF
    convert_html_to_pdf(html_file)
