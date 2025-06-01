#-------------------------------------------------------------------------------------------------#
#-- rag_core.py: Core RAG Logic --#
#-------------------------------------------------------------------------------------------------#

#-- Standard Library Imports --#
import os
import traceback
import time
import random
import json
from urllib.parse import quote_plus, urljoin, urlparse

#-- Third-Party Library Imports --#
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#-------------------------------------------------------------------------------------------------#
#-- Application Configuration (Moved from v2_app_searxng_discover.py) --#
#-------------------------------------------------------------------------------------------------#
SEARXNG_URL = "http://localhost:8080"
SEARXNG_MAX_RESULTS = 150
MAX_CONTENT_PER_PAGE = 250000
MIN_DELAY_PER_REQUEST_SCRAPE = 0.5
MAX_DELAY_PER_REQUEST_SCRAPE = 1.0
REQUEST_TIMEOUT = 100000

OLLAMA_MODEL_PY = "huihui_ai/qwen3-abliterated:4b" # This might be overridden by shell script
OLLAMA_EMBEDDING_MODEL_PY = "nomic-embed-text" # This might be overridden by shell script

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVER = 200
K_CONTEXT_FOR_LLM = 50

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36',
]
SHARED_CONTEXT_BASE_DIR = os.path.expanduser("~/my_rag_shared_contexts")

#-------------------------------------------------------------------------------------------------#
#-- Initialize Core Components (Moved from v2_app_searxng_discover.py) --#
#-------------------------------------------------------------------------------------------------#
print("Initializing RAG core Langchain components...")
# Note: OLLAMA_MODEL_PY and OLLAMA_EMBEDDING_MODEL_PY can be updated by an external script (e.g. shell script)
# if this module is imported after such configurations are set in os.environ or by direct modification.
# For direct use of this module, these are the defaults.
llm = OllamaLLM(model=OLLAMA_MODEL_PY)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_PY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
print("RAG core Langchain components initialization complete.")

#-------------------------------------------------------------------------------------------------#
#-- Helper for Polite Requests --#
#-------------------------------------------------------------------------------------------------#
def make_polite_request(url, method='GET', data=None, headers=None, stream=False):
    req_headers = {'User-Agent': random.choice(USER_AGENTS)}
    if headers: req_headers.update(headers)
    time.sleep(random.uniform(MIN_DELAY_PER_REQUEST_SCRAPE / 2, MAX_DELAY_PER_REQUEST_SCRAPE / 2))
    try:
        response = requests.request(method, url, headers=req_headers, data=data,
                                    timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=stream)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"  REQUEST_HELPER: Error fetching {url}: {e}")
        return None

#-------------------------------------------------------------------------------------------------#
#-- LLM Query Expansion Function --#
#-------------------------------------------------------------------------------------------------#
def expand_query_with_llm(query, llm_instance=llm): # Default to global llm
    print(f"EXPAND_QUERY: Expanding query '{query}' using LLM.")
    expansion_prompt_template_str = (
        "You are a query optimization assistant. Given the user's web search query: '{original_query}', "
        "suggest exactly two alternative or expanded search queries that are likely to yield more comprehensive or relevant results from a web search engine like SearXNG. "
        "Focus on different angles or aspects of the original query. "
        "Return *only* the two suggested queries, each on a new line. Do not include numbering, labels, or any other explanatory text."
    )
    expansion_prompt = PromptTemplate.from_template(expansion_prompt_template_str)
    expansion_chain = expansion_prompt | llm_instance
    try:
        response_text = expansion_chain.invoke({"original_query": query})
        expanded_queries = [q.strip() for q in response_text.strip().splitlines() if q.strip()]
        print(f"EXPAND_QUERY: LLM suggested: {expanded_queries}")
        return expanded_queries[:2]
    except Exception as e:
        print(f"EXPAND_QUERY: Error during query expansion with LLM: {e}")
        traceback.print_exc()
        return []

#-------------------------------------------------------------------------------------------------#
#-- URL Discovery via Local SearXNG Instance --#
#-------------------------------------------------------------------------------------------------#
def discover_urls_via_searxng(query):
    print(f"SEARXNG_DISCOVERY: Original query '{query}'")
    expanded_queries = expand_query_with_llm(query) # Uses global llm from rag_core
    all_queries_to_search = [query] + expanded_queries
    all_queries_to_search = list(dict.fromkeys(all_queries_to_search))
    print(f"SEARXNG_DISCOVERY: Effective queries for SearXNG: {all_queries_to_search}")
    discovered_urls_info = []
    all_discovered_urls_set = set()
    for current_search_query in all_queries_to_search:
        print(f"SEARXNG_DISCOVERY: Querying SearXNG for '{current_search_query}'")
        params = {'q': current_search_query, 'format': 'json', 'pageno': 1}
        try:
            response = requests.get(f"{SEARXNG_URL.rstrip('/')}/search", params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if "results" in data and data["results"]:
                for res in data["results"][:SEARXNG_MAX_RESULTS]:
                    title, url = res.get('title', 'No Title'), res.get('url')
                    if url and url.startswith(('http://', 'https://')) and url not in all_discovered_urls_set:
                        discovered_urls_info.append({'title': title, 'href': url})
                        all_discovered_urls_set.add(url)
            else:
                print(f"SEARXNG_DISCOVERY: No results for '{current_search_query}'.")
        except requests.exceptions.ConnectionError as e:
            return {"error": f"SearXNG connection error: {e}", "results": []}
        except Exception as e: # Catch other request or JSON errors
            print(f"SEARXNG_DISCOVERY: Error for '{current_search_query}': {e}")
    if not discovered_urls_info: return {"error": "No usable URLs found from SearXNG.", "results": []}
    return {"error": None, "results": discovered_urls_info}

#-------------------------------------------------------------------------------------------------#
#-- Content Fetching --#
#-------------------------------------------------------------------------------------------------#
def fetch_page_content(url):
    response = make_polite_request(url)
    if not response: return None, None
    try:
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type: return None, None
        soup = BeautifulSoup(response.content, 'html.parser')
        title = (soup.find('title').string.strip() if soup.find('title') else "No Title Found")
        text_content = "\n".join([p.get_text(separator=' ', strip=True) for p in soup.find_all('p')])
        if not text_content.strip(): # Fallback for pages without <p> tags
            body = soup.find('body')
            if body:
                for tag in body(["script", "style", "nav", "footer", "header", "aside", "form", "button"]): tag.decompose()
                text_content = body.get_text(separator=' ', strip=True)
        return title, text_content[:MAX_CONTENT_PER_PAGE].strip()
    except Exception as e: print(f"FETCH_PAGE: Error processing {url}: {e}"); return None, None

#-------------------------------------------------------------------------------------------------#
#-- Scraping Orchestrator --#
#-------------------------------------------------------------------------------------------------#
def searxng_discover_and_scrape(query):
    url_discovery = discover_urls_via_searxng(query)
    if url_discovery.get("error"): return url_discovery
    scraped_docs = []
    for item in url_discovery["results"]:
        title, content = fetch_page_content(item['href'])
        if content: scraped_docs.append({"title": title or item['title'], "source": item['href'], "page_content": content})
    if not scraped_docs: return {"error": "Failed to scrape any content.", "results": []}
    return {"error": None, "results": scraped_docs}

#-------------------------------------------------------------------------------------------------#
#-- Document Chunking --#
#-------------------------------------------------------------------------------------------------#
def _chunk_scraped_documents(processed_docs_as_dicts, text_splitter_instance=text_splitter): # Default to global text_splitter
    all_texts, all_metadatas = [], []
    if not processed_docs_as_dicts: return all_texts, all_metadatas
    for i, doc_dict in enumerate(processed_docs_as_dicts):
        page_content = doc_dict.get("page_content", "")
        if not page_content.strip(): continue
        source, title = doc_dict.get("source", f"doc_idx_{i}"), doc_dict.get("title", "Untitled")
        try:
            chunks = text_splitter_instance.split_text(page_content)
            for chunk_content in chunks:
                all_texts.append(chunk_content)
                all_metadatas.append({"source": source, "title": title, "original_doc_index": i})
        except Exception as e: print(f"CHUNK_DOCS: Error chunking {source}: {e}")
    return all_texts, all_metadatas

#-------------------------------------------------------------------------------------------------#
#-- Vector Store Operations --#
#-------------------------------------------------------------------------------------------------#
def get_or_create_vectorstore(processed_docs_as_dicts=None, existing_vectorstore=None, new_texts=None, new_metadatas=None):
    status_detail = ""
    vectorstore_to_return = existing_vectorstore

    if new_texts and new_metadatas: # If texts/metadatas are pre-chunked and passed
        if existing_vectorstore:
            try:
                existing_vectorstore.add_texts(texts=new_texts, metadatas=new_metadatas)
                status_detail = f"Added {len(new_texts)} new chunks to existing KB."
            except Exception as e:
                status_detail = f"Error adding pre-chunked texts to KB: {e}"
                # Decide if we should return None or existing_vectorstore here
        else:
            try:
                vectorstore_to_return = FAISS.from_texts(new_texts, embeddings, metadatas=new_metadatas)
                status_detail = f"Created new KB with {len(new_texts)} pre-chunked texts."
            except Exception as e:
                status_detail = f"Error creating KB from pre-chunked texts: {e}"
                return None, status_detail
    elif processed_docs_as_dicts: # If raw documents are passed
        texts_for_vs, metadatas_for_vs = _chunk_scraped_documents(processed_docs_as_dicts, text_splitter) # Uses global text_splitter
        if not texts_for_vs:
            return existing_vectorstore, "No text chunks generated from documents."

        if existing_vectorstore:
            try:
                existing_vectorstore.add_texts(texts=texts_for_vs, metadatas=metadatas_for_vs)
                status_detail = f"Added {len(processed_docs_as_dicts)} docs ({len(texts_for_vs)} chunks) to existing KB."
            except Exception as e:
                 status_detail = f"Error adding new document chunks to KB: {e}"
        else:
            try:
                vectorstore_to_return = FAISS.from_texts(texts_for_vs, embeddings, metadatas=metadatas_for_vs)
                status_detail = f"Created new KB with {len(processed_docs_as_dicts)} docs ({len(texts_for_vs)} chunks)."
            except Exception as e:
                status_detail = f"Error creating new KB from documents: {e}"
                return None, status_detail

    if not vectorstore_to_return:
        return None, status_detail + " Vector store is unavailable."

    return vectorstore_to_return, status_detail

#-------------------------------------------------------------------------------------------------#
#-- Context Retrieval & LLM Answer Generation --#
#-------------------------------------------------------------------------------------------------#
def retrieve_context(vectorstore, query, k=K_RETRIEVER):
    if not vectorstore: return []
    try: return vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
    except Exception as e: print(f"RETRIEVE_CONTEXT: Error - {e}"); return []

def generate_llm_answer(query, context_docs, history=None, llm_instance=llm): # Default to global llm
    if not context_docs: return {"answer": "Insufficient context to answer the question.", "sources_md": "", "error": "Insufficient context"}

    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs[:K_CONTEXT_FOR_LLM]])

    unique_sources = {} # Using dict to maintain order of first appearance for a source
    for doc in context_docs[:K_CONTEXT_FOR_LLM]:
        source_url = doc.metadata.get('source')
        if source_url and source_url not in unique_sources:
            source_title = doc.metadata.get('title', 'Source')
            safe_title = source_title.replace('[', '(').replace(']', ')')
            unique_sources[source_url] = f"- [{safe_title}]({source_url})"
    sources_md_output = "\n".join(unique_sources.values())

    if not sources_md_output: sources_md_output = "*(No specific web sources identified for this answer from the context.)*"

    prompt_template_str = (
        "You are a helpful AI assistant. Based *only* on the following CONTEXT (snippets from web pages), "
        "answer the user's QUESTION. Consider the CHAT HISTORY for conversational context. "
        "Your answer should be concise and directly address the question. "
        "If the CONTEXT is insufficient or irrelevant, state that clearly. Do not make up information.\n\n"
        "CHAT HISTORY:\n{history_formatted}\n\n"
        "CONTEXT (Web Page Snippets):\n------------------------\n{context}\n------------------------\n\n"
        "Question: {question}\n\nAnswer:"
    )
    prompt = PromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm_instance
    try:
        llm_answer_text = chain.invoke({"context": context_text, "question": query, "history_formatted": "\n".join([f"Human: {h}\nAI: {a}" for h,a in history]) if history else "(No previous conversation)"})
        return {"answer": llm_answer_text.strip(), "sources_md": sources_md_output, "error": None}
    except Exception as e:
        print(f"GENERATE_LLM_ANSWER: Error - {e}")
        return {"answer": f"LLM Error during generation: {e}", "sources_md": sources_md_output, "error": str(e)}

#-------------------------------------------------------------------------------------------------#
#-- Core Save/Load Logic for FAISS & JSON --#
#-------------------------------------------------------------------------------------------------#
def core_save_context(vectorstore, docs_to_save, context_name):
    if not context_name or not context_name.strip():
        return "Error: Context name is empty."
    safe_name = context_name.strip().replace(" ", "_")

    os.makedirs(SHARED_CONTEXT_BASE_DIR, exist_ok=True)
    dynamic_faiss_path = os.path.join(SHARED_CONTEXT_BASE_DIR, f"rag_faiss_index_{safe_name}")
    dynamic_scraped_docs_path = os.path.join(SHARED_CONTEXT_BASE_DIR, f"rag_scraped_docs_{safe_name}.json")

    status = ""
    try:
        if vectorstore:
            vectorstore.save_local(dynamic_faiss_path)
            status += f"FAISS index saved to '{dynamic_faiss_path}'. "
        if docs_to_save: # Ensure docs_to_save is not None or empty before trying to dump
            with open(dynamic_scraped_docs_path, 'w') as f:
                json.dump(docs_to_save, f)
            status += f"Scraped documents saved to '{dynamic_scraped_docs_path}'."

        if not status: # Nothing was saved
            return f"Nothing to save for context '{safe_name}' (Vectorstore or documents missing)."
        return f"Context '{safe_name}' saved successfully. {status}".strip()
    except Exception as e:
        return f"Error saving context '{safe_name}': {e}"

def core_load_context(context_name):
    if not context_name or not context_name.strip():
        return None, [], "Error: Context name is empty."
    safe_name = context_name.strip().replace(" ", "_")

    dynamic_faiss_path = os.path.join(SHARED_CONTEXT_BASE_DIR, f"rag_faiss_index_{safe_name}")
    dynamic_scraped_docs_path = os.path.join(SHARED_CONTEXT_BASE_DIR, f"rag_scraped_docs_{safe_name}.json")

    loaded_vs = None
    loaded_docs = []
    messages = []

    try:
        if os.path.exists(dynamic_faiss_path):
            loaded_vs = FAISS.load_local(dynamic_faiss_path, embeddings, allow_dangerous_deserialization=True) # Uses global embeddings
            messages.append(f"FAISS index '{safe_name}' loaded.")
        else:
            messages.append(f"FAISS index for '{safe_name}' not found at '{dynamic_faiss_path}'.")

        if os.path.exists(dynamic_scraped_docs_path):
            with open(dynamic_scraped_docs_path, 'r') as f:
                loaded_docs = json.load(f)
            messages.append(f"Scraped documents for '{safe_name}' loaded.")
        else:
            messages.append(f"Scraped documents file for '{safe_name}' not found at '{dynamic_scraped_docs_path}'.")

        if not loaded_vs and not loaded_docs:
            return None, [], f"Failed to load context '{safe_name}'. No files found."

        return loaded_vs, loaded_docs, " ".join(messages)
    except Exception as e:
        return None, [], f"Error loading context '{safe_name}': {e}"

# Example of how generate_llm_answer prompt template string looks:
# prompt_template_str = (
# "You are a helpful AI assistant. Based *only* on the following CONTEXT (snippets from web pages), "
# "answer the user's QUESTION. Consider the CHAT HISTORY for conversational context. "
# "Your answer should be concise and directly address the question. "
# "If the CONTEXT is insufficient or irrelevant, state that clearly. Do not make up information.\n\n"
# "CHAT HISTORY:\n{history_formatted}\n\n"
# "CONTEXT (Web Page Snippets):\n------------------------\n{context}\n------------------------\n\n"
# "Question: {question}\n\nAnswer:" )
