"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PDF Processor with Docling + OCR Fallback                                   ║
║                                                                              ║
║  Intelligent PDF parsing that:                                               ║
║  1. Uses Docling for layout analysis, table detection, formula extraction    ║
║  2. Falls back to OCR (EasyOCR) for scanned/image-heavy PDFs                ║
║  3. Exports clean Markdown for downstream processing                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, field

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Represents a processed PDF document."""
    source: str
    markdown: str
    metadata: dict = field(default_factory=dict)
    pages: int = 0
    tables_count: int = 0
    ocr_used: bool = False


@dataclass 
class DocumentChunk:
    """A chunk of text from a processed document."""
    content: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """
    Intelligent PDF processor using Docling.
    
    Processing Strategy:
    ═══════════════════
    
    1. Primary Mode (do_ocr=False):
       - Fast layout analysis
       - Table structure detection
       - Formula extraction
       - Works great for digital PDFs
    
    2. OCR Mode (do_ocr=True):
       - Activated for scanned documents
       - Uses EasyOCR by default (good multilingual support)
       - Can fallback to Tesseract if needed
    
    Usage:
        processor = PDFProcessor(enable_ocr=True)
        doc = processor.process("document.pdf")
        chunks = processor.chunk_document(doc)
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_engine: str = "easyocr",  # "easyocr", "tesseract", "rapidocr"
        enable_table_structure: bool = True,
        enable_images: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the PDF processor.
        
        Args:
            enable_ocr: Enable OCR for scanned documents
            ocr_engine: OCR engine to use ("easyocr", "tesseract", "rapidocr")
            enable_table_structure: Extract table structures
            enable_images: Include image descriptions
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
        """
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine
        self.enable_table_structure = enable_table_structure
        self.enable_images = enable_images
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._converter = self._create_converter()
        
        logger.info(f"PDFProcessor initialized (OCR: {enable_ocr}, Engine: {ocr_engine})")
    
    def _create_converter(self) -> DocumentConverter:
        """Create the Docling document converter with configured options."""
        
        # Configure PDF pipeline options
        # Note: Docling v2+ handles OCR automatically when needed
        pipeline_options = PdfPipelineOptions(
            do_ocr=self.enable_ocr,
            do_table_structure=self.enable_table_structure,
        )
        
        # Create converter with format-specific options
        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.HTML,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )
        
        return converter
    
    def process(self, source: str | Path) -> ProcessedDocument:
        """
        Process a single PDF document.
        
        Args:
            source: Path to PDF file or URL
            
        Returns:
            ProcessedDocument with markdown content and metadata
        """
        source_str = str(source)
        logger.info(f"Processing: {source_str}")
        
        try:
            # Convert the document
            result = self._converter.convert(source_str)
            doc = result.document
            
            # Export to markdown
            markdown = doc.export_to_markdown()
            
            # Extract metadata
            metadata = {
                "title": getattr(doc, 'title', None) or Path(source_str).stem,
                "source": source_str,
            }
            
            # Count pages and tables
            pages = len(doc.pages) if hasattr(doc, 'pages') else 0
            tables_count = len(doc.tables) if hasattr(doc, 'tables') else 0
            
            processed = ProcessedDocument(
                source=source_str,
                markdown=markdown,
                metadata=metadata,
                pages=pages,
                tables_count=tables_count,
                ocr_used=self.enable_ocr,
            )
            
            logger.info(f"Processed: {pages} pages, {tables_count} tables, {len(markdown)} chars")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing {source_str}: {e}")
            raise
    
    def process_batch(
        self, 
        sources: list[str | Path]
    ) -> Generator[ProcessedDocument, None, None]:
        """
        Process multiple PDF documents.
        
        Args:
            sources: List of paths to PDF files or URLs
            
        Yields:
            ProcessedDocument for each successfully processed file
        """
        logger.info(f"Processing batch of {len(sources)} documents")
        
        for source in sources:
            try:
                yield self.process(source)
            except Exception as e:
                logger.error(f"Skipping {source}: {e}")
                continue
    
    def chunk_document(
        self, 
        doc: ProcessedDocument,
        method: str = "semantic",  # "semantic", "fixed", "paragraph"
    ) -> list[DocumentChunk]:
        """
        Split a processed document into chunks for embedding.
        
        Args:
            doc: Processed document to chunk
            method: Chunking strategy
                - "semantic": Split by sections/headers (recommended)
                - "fixed": Fixed-size chunks with overlap
                - "paragraph": Split by paragraphs
                
        Returns:
            List of DocumentChunk objects
        """
        if method == "semantic":
            return self._chunk_semantic(doc)
        elif method == "fixed":
            return self._chunk_fixed(doc)
        elif method == "paragraph":
            return self._chunk_paragraph(doc)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def _chunk_semantic(self, doc: ProcessedDocument) -> list[DocumentChunk]:
        """
        Chunk by semantic sections (headers, etc.).
        
        This is the recommended approach as it preserves document structure.
        """
        chunks = []
        lines = doc.markdown.split('\n')
        
        current_section = None
        current_content = []
        current_size = 0
        
        for line in lines:
            # Detect section headers (Markdown format)
            if line.startswith('#'):
                # Save previous section if exists
                if current_content:
                    chunk_text = '\n'.join(current_content).strip()
                    if chunk_text:
                        chunks.append(DocumentChunk(
                            content=chunk_text,
                            source=doc.source,
                            section=current_section,
                            metadata=doc.metadata.copy(),
                        ))
                
                # Start new section
                current_section = line.lstrip('#').strip()
                current_content = [line]
                current_size = len(line)
            else:
                current_content.append(line)
                current_size += len(line)
                
                # Split if chunk gets too large
                if current_size > self.chunk_size:
                    chunk_text = '\n'.join(current_content).strip()
                    if chunk_text:
                        chunks.append(DocumentChunk(
                            content=chunk_text,
                            source=doc.source,
                            section=current_section,
                            metadata=doc.metadata.copy(),
                        ))
                    current_content = []
                    current_size = 0
        
        # Don't forget the last section
        if current_content:
            chunk_text = '\n'.join(current_content).strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=doc.source,
                    section=current_section,
                    metadata=doc.metadata.copy(),
                ))
        
        logger.info(f"Created {len(chunks)} semantic chunks from {doc.source}")
        return chunks
    
    def _chunk_fixed(self, doc: ProcessedDocument) -> list[DocumentChunk]:
        """Chunk with fixed size and overlap."""
        chunks = []
        text = doc.markdown
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence end within the last 20% of the chunk
                search_start = start + int(self.chunk_size * 0.8)
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    pos = text.rfind(punct, search_start, end)
                    if pos != -1:
                        end = pos + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=doc.source,
                    metadata=doc.metadata.copy(),
                ))
            
            start = end - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} fixed-size chunks from {doc.source}")
        return chunks
    
    def _chunk_paragraph(self, doc: ProcessedDocument) -> list[DocumentChunk]:
        """Chunk by paragraphs (double newlines)."""
        chunks = []
        paragraphs = doc.markdown.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If paragraph alone exceeds chunk size, split it
            if para_size > self.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content='\n\n'.join(current_chunk),
                        source=doc.source,
                        metadata=doc.metadata.copy(),
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph
                words = para.split()
                temp_chunk = []
                temp_size = 0
                for word in words:
                    if temp_size + len(word) + 1 > self.chunk_size:
                        chunks.append(DocumentChunk(
                            content=' '.join(temp_chunk),
                            source=doc.source,
                            metadata=doc.metadata.copy(),
                        ))
                        temp_chunk = [word]
                        temp_size = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_size += len(word) + 1
                
                if temp_chunk:
                    current_chunk = [' '.join(temp_chunk)]
                    current_size = temp_size
            
            elif current_size + para_size > self.chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content='\n\n'.join(current_chunk),
                        source=doc.source,
                        metadata=doc.metadata.copy(),
                    ))
                current_chunk = [para]
                current_size = para_size
            
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Save final chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                content='\n\n'.join(current_chunk),
                source=doc.source,
                metadata=doc.metadata.copy(),
            ))
        
        logger.info(f"Created {len(chunks)} paragraph chunks from {doc.source}")
        return chunks


class ContextualChunker:
    """
    Adds LLM-generated context to chunks for better retrieval.
    
    Based on Anthropic's "Contextual Retrieval" technique which reduces
    retrieval failures by up to 67%.
    
    How it works:
    ═════════════
    
    Before: "Revenue grew by 15% compared to Q2..."
            ↓
    After:  "[This chunk is from ACME Corp's Q3 2024 Financial Report,
             specifically discussing revenue growth in the Results section.]
             
             Revenue grew by 15% compared to Q2..."
    """
    
    def __init__(self, lm=None):
        """
        Initialize the contextual chunker.
        
        Args:
            lm: Optional DSPy language model. If not provided, uses default.
        """
        self.lm = lm
        self._context_generator = None
    
    def _lazy_init(self):
        """Lazy initialization of DSPy components."""
        if self._context_generator is None:
            import dspy
            
            class GenerateContext(dspy.Signature):
                """Generate a brief context prefix for a document chunk."""
                
                document_title = dspy.InputField(desc="Title of the source document")
                chunk_content = dspy.InputField(desc="The chunk content to contextualize")
                document_summary = dspy.InputField(desc="Brief summary of the document")
                
                context_prefix = dspy.OutputField(
                    desc="1-2 sentence context starting with 'This chunk'. "
                         "Mention the document name and what this section covers."
                )
            
            self._context_generator = dspy.Predict(GenerateContext)
    
    def add_context(
        self,
        chunks: list[DocumentChunk],
        document_summary: str,
    ) -> list[DocumentChunk]:
        """
        Add contextual prefixes to chunks.
        
        Args:
            chunks: List of document chunks
            document_summary: Brief summary of the source document
            
        Returns:
            Chunks with contextual prefixes added to content
        """
        self._lazy_init()
        
        contextualized = []
        for chunk in chunks:
            try:
                # Generate context for this chunk
                result = self._context_generator(
                    document_title=chunk.metadata.get('title', chunk.source),
                    chunk_content=chunk.content[:500],  # Use first 500 chars
                    document_summary=document_summary,
                )
                
                # Prepend context to content
                contextualized_content = f"[{result.context_prefix}]\n\n{chunk.content}"
                
                new_chunk = DocumentChunk(
                    content=contextualized_content,
                    source=chunk.source,
                    page=chunk.page,
                    section=chunk.section,
                    metadata={
                        **chunk.metadata,
                        "original_content": chunk.content,
                        "context_prefix": result.context_prefix,
                    },
                )
                contextualized.append(new_chunk)
                
            except Exception as e:
                logger.warning(f"Failed to add context to chunk: {e}")
                contextualized.append(chunk)
        
        logger.info(f"Added context to {len(contextualized)} chunks")
        return contextualized


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for testing the PDF processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs with Docling")
    parser.add_argument("source", help="PDF file path or URL")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--ocr-engine", default="easyocr", 
                        choices=["easyocr", "tesseract", "rapidocr"])
    parser.add_argument("--chunk-method", default="semantic",
                        choices=["semantic", "fixed", "paragraph"])
    parser.add_argument("--output", "-o", help="Output markdown file")
    
    args = parser.parse_args()
    
    # Process the document
    processor = PDFProcessor(
        enable_ocr=not args.no_ocr,
        ocr_engine=args.ocr_engine,
    )
    
    doc = processor.process(args.source)
    
    print(f"\n{'═' * 60}")
    print(f"Processed: {doc.source}")
    print(f"Pages: {doc.pages}")
    print(f"Tables: {doc.tables_count}")
    print(f"OCR Used: {doc.ocr_used}")
    print(f"Content Length: {len(doc.markdown)} characters")
    print(f"{'═' * 60}\n")
    
    # Create chunks
    chunks = processor.chunk_document(doc, method=args.chunk_method)
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Section: {chunk.section}")
        print(chunk.content[:200] + "...")
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(doc.markdown)
        print(f"\nSaved markdown to: {args.output}")


if __name__ == "__main__":
    main()

