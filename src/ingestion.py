from pathlib import Path
from typing import List, Tuple
import logging
import re

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """
    Load PDFs and split into overlapping text chunks.

    Uses RecursiveCharacterTextSplitter to preserve semantic structure
    while maintaining chunk overlap for retrieval quality.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}"
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def load_documents(self) -> List[Tuple[str, str]]:
        """
        Load all PDFs and return:
        [
            (document_name, text),
            ...
        ]
        """
        documents: List[Tuple[str, str]] = []

        pdf_files = sorted(self.data_dir.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.data_dir}"
            )

        for pdf_file in pdf_files:
            try:
                text = self._extract_text_from_pdf(pdf_file)

                if text.strip():
                    documents.append(
                        (pdf_file.stem, text)
                    )

                logger.info(
                    "Loaded %s (%d chars)",
                    pdf_file.name,
                    len(text),
                )

            except Exception as exc:
                logger.warning(
                    "Failed to process %s: %s",
                    pdf_file.name,
                    exc,
                )

        return documents

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF and remove sections
        that hurt retrieval quality.
        """
        text_parts: List[str] = []

        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

        text = "\n".join(text_parts)

        # Remove references section
        reference_markers = [
            "\nReferences\n",
            "\nREFERENCES\n",
            "\nBibliography\n",
            "\nBIBLIOGRAPHY\n",
        ]

        for marker in reference_markers:
            if marker in text:
                text = text.split(marker)[0]
                break

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def chunk_documents(
        self,
        documents: List[Tuple[str, str]],
    ) -> List[dict]:
        """
        Convert documents into chunk dictionaries.
        """
        chunks: List[dict] = []

        for doc_name, text in documents:

            split_chunks = self.text_splitter.split_text(text)

            logger.info(
                "%s -> %d chunks",
                doc_name,
                len(split_chunks),
            )

            for chunk_index, chunk_text in enumerate(split_chunks):

                if len(chunk_text.strip()) < 100:
                    continue

                chunks.append(
                    {
                        "id": f"{doc_name}_{chunk_index}",
                        "document": doc_name,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )

        logger.info(
            "Created %d total chunks",
            len(chunks),
        )

        return chunks

    def ingest(self) -> List[dict]:
        """
        Full ingestion pipeline.
        """
        documents = self.load_documents()
        return self.chunk_documents(documents)