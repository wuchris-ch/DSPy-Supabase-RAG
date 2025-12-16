"""
Download sample PDFs for testing the RAG pipeline.

These are open-access research papers and documents that work well for testing:
- Technical documentation
- Research papers (arXiv)
- Various document layouts (tables, figures, formulas)

Usage:
    python download_samples.py
    python download_samples.py --output ./my_pdfs
"""

import os
import urllib.request
from pathlib import Path

# Sample PDFs for testing
SAMPLE_PDFS = {
    # Docling Technical Report - great for testing (has tables, figures)
    "docling_report.pdf": "https://arxiv.org/pdf/2408.09869",
    
    # Attention Is All You Need - famous transformer paper
    "attention_paper.pdf": "https://arxiv.org/pdf/1706.03762",
    
    # BERT paper - good mix of text and tables
    "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805",
    
    # RAG paper - relevant to what we're building!
    "rag_paper.pdf": "https://arxiv.org/pdf/2005.11401",
    
    # GPT-4 Technical Report - comprehensive doc
    "gpt4_report.pdf": "https://arxiv.org/pdf/2303.08774",
}


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF from URL."""
    try:
        print(f"  Downloading: {output_path.name}...")
        
        # Add headers to avoid 403 errors
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        with urllib.request.urlopen(request) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {output_path.name} ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download {output_path.name}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample PDFs for testing")
    parser.add_argument("--output", "-o", default="./sample_pdfs",
                        help="Output directory for PDFs")
    parser.add_argument("--all", action="store_true",
                        help="Download all samples (default: just 2)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading sample PDFs to: {output_dir.absolute()}")
    print("=" * 50)
    
    # Select which PDFs to download
    if args.all:
        pdfs_to_download = SAMPLE_PDFS
    else:
        # Just download 2 by default for quick testing
        pdfs_to_download = {
            "docling_report.pdf": SAMPLE_PDFS["docling_report.pdf"],
            "rag_paper.pdf": SAMPLE_PDFS["rag_paper.pdf"],
        }
    
    # Download
    success = 0
    for filename, url in pdfs_to_download.items():
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"  ⏭️  {filename} already exists, skipping")
            success += 1
            continue
        
        if download_pdf(url, output_path):
            success += 1
    
    print("=" * 50)
    print(f"✅ Downloaded {success}/{len(pdfs_to_download)} PDFs")
    
    # Show next steps
    print(f"""
🚀 Next Steps:
═════════════

1. Ingest the PDFs:
   python rag_pipeline.py ingest {output_dir}/*.pdf

2. Or ingest one at a time:
   python rag_pipeline.py ingest {output_dir}/docling_report.pdf

3. Query the system:
   python rag_pipeline.py query "What is Docling?"
   python rag_pipeline.py query "How does RAG work?"

4. Interactive mode:
   python rag_pipeline.py interactive

📂 Downloaded files:
""")
    
    for pdf in output_dir.glob("*.pdf"):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"   - {pdf.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

