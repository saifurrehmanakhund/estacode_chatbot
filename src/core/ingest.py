import base64
import uuid
from io import BytesIO
from pathlib import Path
from pdf2image import convert_from_path
import ollama
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import requests

# Absolute PDF page count mapping from our analysis
CHAPTER_MAP = [
    {"id": 1, "start": 2, "end": 8, "title": "Constitutional Provisions"},
    {"id": 2, "start": 9, "end": 91, "title": "Civil Services Structure"},
    {"id": 3, "start": 92, "end": 305, "title": "Recruitment & Induction"},
    {"id": 4, "start": 306, "end": 457, "title": "Appointments"},
    {"id": 5, "start": 458, "end": 500, "title": "The Revised Leave Rules"},
    {"id": 6, "start": 501, "end": 602, "title": "Transfers, Postings & Rotation"},
    {"id": 7, "start": 603, "end": 724, "title": "Career Planning, Promotion & Training"},
    {"id": 8, "start": 725, "end": 885, "title": "Conduct, Efficiency & Discipline"},
    {"id": 9, "start": 886, "end": 932, "title": "Retirement & Severance"},
    {"id": 10, "start": 933, "end": 958, "title": "Appeals, Petitions & Representations"},
    {"id": 11, "start": 960, "end": 1034, "title": "Index"}
]

def unload_ollama_model(model_name: str):
    """Forcefully unloads a model from GPU VRAM by setting keep_alive to 0."""
    print(f"--- Unloading {model_name} from GPU ---")
    try:
        # We send a blank request with keep_alive=0 to tell Ollama to drop the model
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "keep_alive": 0}
        )
        print(f"Successfully cleared {model_name} from VRAM.")
    except Exception as e:
        print(f"Failed to unload model: {e}")

def get_chapter_info(page_num):
    for ch in CHAPTER_MAP:
        if ch["start"] <= page_num <= ch["end"]:
            return ch
    return {"id": 0, "title": "General"}

def process_page_with_glm(image):
    """Encodes image to base64 and gets Markdown from GLM-OCR."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    response = ollama.generate(
        model='glm-ocr',
        prompt='Document Parsing: Extract all text and tables into Markdown.',
        images=[img_base64]
    )
    return response['response']

def run_phase_1_ingestion():
    base_dir = Path(__file__).resolve().parent.parent.parent
    pdf_path = base_dir / "data" / "estacode_v1.pdf"
    db_path = base_dir / "db" / "chroma_index"

    print("--- Converting PDF Pages to Images ---")
    # 150 DPI is the sweet spot for GLM-OCR speed/accuracy
    pages = convert_from_path(str(pdf_path), dpi=150)
    
    child_documents = []
    
    # Text splitters for Child chunks
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    print("--- Starting Multimodal Ingestion (GLM-OCR) ---")
    # for i, page_image in enumerate(tqdm(pages, desc="Processing Pages")):         # Full run for all pages
    for i, page_image in enumerate(tqdm(pages[:8], desc="Testing Chapter 1")):      # Test with first 8 pages (Chapter 1) for faster iteration
        page_num = i + 1
        chapter = get_chapter_info(page_num)
        
        # 1. Vision-based Markdown extraction
        page_markdown = process_page_with_glm(page_image)
        
        # 2. Create a unique ID for the Parent (the whole page)
        parent_id = str(uuid.uuid4())
        
        # 3. Create Child chunks from the Markdown
        # These will be used for vector search
        sub_docs = child_splitter.split_text(page_markdown)
        
        for sub_text in sub_docs:
            child_documents.append(
                Document(
                    page_content=sub_text,
                    metadata={
                        "parent_id": parent_id, # Link back to parent
                        "full_page_content": page_markdown, # Store parent in metadata for easy retrieval
                        "page": page_num,
                        "chapter_id": chapter["id"],
                        "chapter_title": chapter["title"],
                        "is_index": (chapter["id"] == 11)
                    }
                )
            )

    print(f"--- Indexing {len(child_documents)} Child Chunks in ChromaDB ---")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(
        documents=child_documents,
        embedding=embeddings,
        persist_directory=str(db_path)
    )
    
    unload_ollama_model("glm-ocr")
    unload_ollama_model("nomic-embed-text")
    print("--- Phase 1 Complete: Hierarchical Index Built ---")

if __name__ == "__main__":
    run_phase_1_ingestion()