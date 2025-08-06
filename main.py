import os
import fitz
import requests
import json
import tempfile
import hashlib
import time
import logging
import urllib.parse
from typing import List, Dict, Any
from flask import Flask, request, jsonify, make_response
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from docx import Document as DocxDocument
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== CONFIG ==========
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if not all([GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing one or more environment variables. Please check your .env file.")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
TOP_K_CHUNKS = 3  # Increased from 1 to 3 for better context
COLLECTION_NAME_PREFIX = "bajaj-finsery"
MIN_SCORE_THRESHOLD = 0.3  # Added minimum similarity score threshold

# Enhanced system prompt for better accuracy and rationale
ENHANCED_SYSTEM_PROMPT = """You are a specialized document analysis assistant. Your task is to provide accurate, detailed answers and an explainable rationale based strictly on the provided context.

INSTRUCTIONS:
1. Answer ONLY based on information explicitly stated in the context provided.
2. Be precise about all details, including policy terms, conditions, waiting periods, and coverage limits.
3. Include specific details like percentages, time periods, and monetary amounts when available.
4. If the information is not in the context, state: "This information is not available in the provided documents."
5. Structure your response as a JSON object with two keys: "answer" and "rationale".
6. The "answer" should be the direct response to the user's question.
7. The "rationale" must clearly state which specific parts of the provided context were used to formulate the answer. Cite the source document and page number for each piece of information used.

Context:
{context}

Question: {query}

Provide a comprehensive JSON response based solely on the context above:"""

# ========== INIT ==========
model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
app = Flask(__name__)

# ========== DOCUMENT PROCESSING HELPERS ==========
def download_document(url: str, file_path: str) -> None:
    """Download document from URL to a specified file path."""
    try:
        logger.info(f"Downloading document from: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Document downloaded successfully to: {file_path}")
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise ValueError(f"Failed to download document: {str(e)}")

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with metadata."""
    try:
        doc = fitz.open(file_path)
        pages_data = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_data.append({
                    'text': text.strip(),
                    'page_number': page_num + 1,
                    'source_file': os.path.basename(file_path)
                })
        doc.close()
        return pages_data
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return []

def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX with metadata."""
    try:
        doc = DocxDocument(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return [{
            'text': "\n".join(full_text).strip(),
            'page_number': 1, # DOCX doesn't have native page numbers, so we treat it as a single page
            'source_file': os.path.basename(file_path)
        }]
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return []

def intelligent_chunk_text(text: str, max_words: int = 150, overlap_words: int = 50) -> List[str]:
    """Intelligent text chunking with semantic preservation."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        # Preserve section headers
        if len(para.split()) < 10 and para.endswith(':'):
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [para]
            current_word_count = len(current_chunk)
            continue
            
        para_words = para.split()
        if len(para_words) > max_words:
            sentences = para.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence: continue
                if not sentence.endswith('.'): sentence += '.'
                sentence_words = sentence.split()
                
                if current_word_count + len(sentence_words) > max_words and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep header if present
                    if current_chunk[0].endswith(':'):
                        overlap_text = current_chunk[0] + ' ' + ' '.join(current_chunk[-overlap_words:])
                    else:
                        overlap_text = ' '.join(current_chunk[-overlap_words:])
                    current_chunk = overlap_text.split() + sentence_words
                    current_word_count = len(current_chunk)
                else:
                    current_chunk.extend(sentence_words)
                    current_word_count += len(sentence_words)
        else:
            if current_word_count + len(para_words) > max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                if current_chunk[0].endswith(':'):
                    overlap_text = current_chunk[0] + ' ' + ' '.join(current_chunk[-overlap_words:])
                else:
                    overlap_text = ' '.join(current_chunk[-overlap_words:])
                current_chunk = overlap_text.split() + para_words
                current_word_count = len(current_chunk)
            else:
                current_chunk.extend(para_words)
                current_word_count += len(para_words)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if len(chunk.split()) > 20]

def process_and_index_document(document_url: str) -> str:
    """Process document from URL and index it in vector database."""
    try:
        if not document_url.startswith("http"):
            raise ValueError("A valid document URL is required.")

        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        collection_name = f"{COLLECTION_NAME_PREFIX}_{url_hash}"

        try:
            collections = qdrant_client.get_collections()
            if any(col.name == collection_name for col in collections.collections):
                logger.info(f"Document already indexed in collection: {collection_name}")
                return collection_name
        except Exception:
            pass

        # Parse the URL to get the path and extension without query parameters
        url_path = urllib.parse.urlparse(document_url).path
        file_ext = os.path.splitext(url_path)[1].lower()

        # Use tempfile to create a secure, temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file_path = temp_file.name
            
        download_document(document_url, file_path)
        
        try:
            if file_ext == '.pdf':
                pages_data = extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                pages_data = extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            if not pages_data:
                raise ValueError("No text extracted from document")
            
            all_chunks_with_metadata = []
            for page_data in pages_data:
                chunks = intelligent_chunk_text(page_data['text'])
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks_with_metadata.append({
                        'text': chunk,
                        'page_number': page_data.get('page_number', 'N/A'),
                        'chunk_index': chunk_idx,
                        'source_url': document_url,
                        'source_file': page_data['source_file']
                    })
            
            logger.info(f"Created {len(all_chunks_with_metadata)} chunks")
            
            texts = [chunk['text'] for chunk in all_chunks_with_metadata]
            embeddings = model.encode(texts, show_progress_bar=False)
            
            vectors = []
            for i, (chunk_data, embedding) in enumerate(zip(all_chunks_with_metadata, embeddings)):
                vectors.append(
                    PointStruct(
                        id=i,
                        vector=embedding.tolist(),
                        payload=chunk_data
                    )
                )
            
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
            )
            
            qdrant_client.upsert(collection_name=collection_name, points=vectors)
            logger.info(f"Successfully indexed {len(vectors)} chunks in collection: {collection_name}")
            
            return collection_name
            
        finally:
            try:
                os.unlink(file_path)
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        # Provide a more generic error to the user without leaking local path info
        raise ValueError(f"Failed to process document: {str(e)}")

def search_document(collection_name: str, query: str, limit: int = TOP_K_CHUNKS) -> tuple[str, List[Dict[str, Any]]]:
    """Search for relevant chunks and return context and source metadata."""
    try:
        # Generate embeddings for both the original query and semantic variations
        query_variations = [
            query,
            f"what does the document say about {query}",
            f"find information regarding {query}"
        ]
        embeddings = model.encode(query_variations)
        
        # Search with multiple query variations
        all_results = []
        for embedding in embeddings:
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=embedding.tolist(),
                limit=limit,
                score_threshold=MIN_SCORE_THRESHOLD
            )
            all_results.extend(results)
        
        # Deduplicate and sort by score
        seen_texts = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            text = result.payload["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
        
        if not unique_results:
            return "", []
        
        context_parts = []
        source_documents = []
        
        for result in unique_results[:limit]:
            text = result.payload["text"]
            page_num = result.payload.get("page_number", "N/A")
            source_file = result.payload.get("source_file", "Unknown")
            score = result.score
            
            # Add to context with confidence score
            context_parts.append(f"[{source_file}, Page {page_num}, Confidence: {score:.2f}]\n{text}")
            
            source_documents.append({
                "source_file": source_file,
                "page_number": page_num,
                "snippet": text,
                "confidence": score
            })
        
        return "\n\n".join(context_parts), source_documents
        
    except Exception as e:
        logger.error(f"Error searching document: {e}")
        return "", []

def generate_answer_with_gemini(query: str, context: str) -> str:
    """Generate answer using Gemini API and return a simple string."""
    if not context.strip():
        return "This information is not available in the provided documents."
    
    prompt = ENHANCED_SYSTEM_PROMPT.format(context=context, query=query)
    
    try:
        gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction="Your task is to provide accurate, detailed answers and an explainable rationale based strictly on the provided context. Respond with a JSON object that contains the keys 'answer' and 'rationale'."
        )
        
        response = gemini_model.generate_content(prompt)
        raw_output = response.text.strip()
        
        try:
            # Clean up potential markdown formatting before parsing
            if raw_output.startswith("```json") and raw_output.endswith("```"):
                raw_output = raw_output[7:-3].strip()
            
            llm_response = json.loads(raw_output)
            if "answer" in llm_response:
                return llm_response.get("answer", "No answer found in LLM response.")
            else:
                return "LLM response was missing the 'answer' key."
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM: {raw_output}")
            return "Failed to parse a valid JSON response from the language model."
            
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm currently unable to process your request due to a service issue. Please try again later."

# ========== FLASK ROUTES ==========
@app.route("/")
def root():
    return jsonify({
        "status": "healthy",
        "message": "HackRX LLM Query-Retrieval System (Flask)",
        "version": "1.0.0",
        "endpoints": ["/api/v1/hackrx/run"]
    })

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route("/api/v1/hackrx/run", methods=['POST'])
def run_hackrx():
    """
    Main endpoint for processing documents and answering questions.
    """
    try:
        request_data = request.get_json()
        documents_url = request_data.get('documents')
        questions = request_data.get('questions')
        
        if not questions:
            return make_response(jsonify({"detail": "Questions are required"}), 400)
        
        if not documents_url or not documents_url.startswith("http"):
            return make_response(jsonify({"detail": "A valid document URL is required."}), 400)

        collection_name = process_and_index_document(documents_url)
        
        answers_list = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
            
            try:
                context, _ = search_document(collection_name, question, limit=TOP_K_CHUNKS)
                answer_string = generate_answer_with_gemini(question, context)
                answers_list.append(answer_string)
                
                logger.info(f"Generated answer for question {i+1}")
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers_list.append("I apologize, but I encountered an error while processing this question.")
        
        logger.info(f"Successfully processed all {len(answers_list)} questions")
        return jsonify({"answers": answers_list})
        
    except ValueError as e:
        return make_response(jsonify({"detail": str(e)}), 400)
    except Exception as e:
        logger.error(f"Unexpected error in run_hackrx: {e}")
        return make_response(jsonify({"detail": "Internal server error occurred"}), 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
