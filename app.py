import os
import re
from flask import Flask, render_template, request, jsonify
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from gpt4all import GPT4All
import functools
import concurrent.futures
from cachetools import TTLCache, LRUCache
import time
import random

app = Flask(__name__)

# Set environment variables
os.environ["PINECONE_API_KEY"] = "pcsk_4NC22q_T6VQeA3w4cLszg6TQcFYpjQzUVWdqkHSN3oJbHJfw5QVkFdTLKg49oUYPQAfcTj"
os.environ["PINECONE_API_ENV"] = "us-east-1"

# Initialize embeddings - load once at startup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector stores - load once at startup
vector_store_test = PineconeVectorStore.from_existing_index(
    index_name="test",
    embedding=embeddings
)
vector_store_medicalbot = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

# Initialize LLM with optimized settings
llm = GPT4All(
    model_name="Qwen2-1.5B-Instruct.Q8_0.gguf",
    model_path="/Users/shakil/Downloads/RAG-Chatbot/model/",
    device="gpu",
    n_ctx=2048,  # Increased context window
    n_threads=8,  # Adjust based on your CPU
    n_batch=512   # Batch size for faster inference
)

# Pre-compile frequently used regex patterns
PRICE_PATTERN = re.compile(r'([A-Za-z\s\-,/\(\)]+)\s+(\d+(?:\.\d+)?)\s*(?:BDT|Tk\.?)')
TREATMENT_PRICE_PATTERN = re.compile(r'^\d+\s+([A-Za-z0-9\s\-,/\(\)]+?)\s+BDT\s+(\d+(?:\.\d+)?)', flags=re.I)
SENTENCE_PATTERN = re.compile(r'[.!?]+')
PRICE_FORMAT_PATTERN = re.compile(r'(\d+)(?:\.00)?\s*(?:BDT|Tk\.?)')
SPACE_BEFORE_PRICE_PATTERN = re.compile(r'(?<=\S)(?=\d+\.00 BDT)')
BDT_QUERY_PATTERN = re.compile(r'^\s*(.+?)\s*BDT\s*$', flags=re.I)
GENERAL_PRICE_PATTERN = re.compile(r'BDT\s+(\d+(?:\.\d+)?)')

# Bot identity information
BOT_INFO = {
    "name": "MediAssist",
    "version": "1.1",
    "description": "An advanced healthcare assistant providing accurate medical information and pricing.",
    "capabilities": [
        "Answering medical questions based on verified information",
        "Providing treatment pricing information",
        "Offering general healthcare guidance",
        "Responding with natural conversational patterns"
    ],
    "limitations": [
        "Cannot provide personalized medical diagnosis",
        "Information is limited to the available knowledge database",
        "Not a replacement for professional medical consultation"
    ]
}

# Precomputed bot keywords for faster matching
BOT_KEYWORDS = frozenset([
    "who are you", "what are you", "your name", "about you", "tell me about yourself",
    "what can you do", "your capabilities", "how do you work", "who made you",
    "what is this", "chatbot", "medibot", "assistant", "your purpose", "your function"
])

# Precomputed price keywords for faster matching
PRICE_KEYWORDS = frozenset(['price', 'cost', 'how much', 'fee', 'charge', 'payment', 'rate', 'expense'])

# Cache for storing responses to frequently asked questions (5 minute TTL)
response_cache = TTLCache(maxsize=1000, ttl=300)

# Cache for storing vector search results (10 minute TTL)
vector_search_cache = TTLCache(maxsize=500, ttl=600)

# Cache for storing commonly used treatment information
treatment_cache = LRUCache(maxsize=200)

# Human-like response variations
RESPONSE_VARIATIONS = {
    "uncertainty": [
        "I don't have information about that in my database. I'd recommend consulting with a healthcare professional for personalized advice.",
        "That's beyond the scope of my current knowledge. Please consult with a qualified healthcare provider for specific guidance.",
        "I don't have details on that in my records. It would be best to speak with a healthcare professional for accurate information.",
        "I'm not able to provide information on that topic. I'd suggest reaching out to a medical professional for assistance."
    ],
    "pricing_unknown": [
        "I don't have pricing information about that in my database. Please contact the relevant healthcare facility directly for current pricing.",
        "I don't have that specific pricing information available. For the most up-to-date rates, I'd recommend contacting the healthcare provider directly.",
        "That pricing information isn't in my database. You'll need to check with the healthcare facility for their current rates.",
        "I don't have access to that pricing data. For accurate cost information, please reach out to the healthcare provider directly."
    ],
    "transitions": [
        "Based on the information I have, ",
        "According to my database, ",
        "From what I understand, ",
        "The information I have indicates that ",
        "My records show that ",
    ],
    "acknowledgments": [
        "I see you're asking about ",
        "You're interested in ",
        "Regarding your question about ",
        "Concerning ",
        "About your inquiry on ",
    ]
}

@functools.lru_cache(maxsize=256)
def normalize_treatment_name(name):
    """Normalize treatment name for better matching (cached for performance)"""
    return re.sub(r'[^a-z0-9]', '', name.lower())

@functools.lru_cache(maxsize=512)
def treatments_match(query, treatment):
    """Check if treatment names match (cached for performance)"""
    query_norm = normalize_treatment_name(query)
    treatment_norm = normalize_treatment_name(treatment)
    
    # Direct match
    if query_norm == treatment_norm:
        return True
    
    # Check if one is a substring of the other
    if query_norm in treatment_norm or treatment_norm in query_norm:
        return True
    
    # Check if most words match
    query_words = set(query_norm.split())
    treatment_words = set(treatment_norm.split())
    common_words = query_words.intersection(treatment_words)
    
    if common_words and len(common_words) / max(len(query_words), len(treatment_words)) > 0.6:
        return True
    
    return False

def is_about_bot(question):
    """Detect if the question is asking about the bot itself"""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in BOT_KEYWORDS)

def get_bot_response():
    """Generate a professional response about the bot with human-like variations"""
    variants = [
        f"I'm {BOT_INFO['name']}, a healthcare information assistant designed to provide evidence-based medical information and pricing details. While I offer general healthcare guidance, I'm not a substitute for professional medical advice.",
        
        f"I'm {BOT_INFO['name']}, an AI healthcare assistant that provides medical information and pricing details from verified sources. I can help with general healthcare questions, but remember that I'm not a replacement for consulting with a healthcare professional.",
        
        f"My name is {BOT_INFO['name']}, and I'm here to assist you with healthcare information and pricing inquiries. I draw from a database of verified medical knowledge, though I should mention that my responses aren't a substitute for professional medical consultation."
    ]
    
    return random.choice(variants)

def extract_price_info(docs):
    """Extract price information from document metadata and content"""
    price_data = {}
    for doc in docs:
        # Check metadata if available
        if doc.metadata.get('has_price_info') and 'price_data' in doc.metadata:
            price_data.update(doc.metadata['price_data'])
        # Also scan the content for price patterns
        content = doc.page_content
        price_matches = PRICE_PATTERN.findall(content)
        for treatment, price in price_matches:
            treatment = treatment.strip()
            if treatment and not treatment.isdigit():
                price_data[treatment.lower()] = price  # store keys in lowercase
    return price_data

def extract_price_from_text(text):
    """Extract treatment name and price from a text string."""
    match = TREATMENT_PRICE_PATTERN.search(text)
    if match:
        treatment = match.group(1).strip().lower()
        price = match.group(2).strip()
        return treatment, price
    return None, None

def format_retrieved_context(retrieved_docs):
    """Format documents with metadata and highlighted price information"""
    price_info = extract_price_info(retrieved_docs)
    context_parts = []
    for doc in retrieved_docs:
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        doc_type = f" ({doc.metadata.get('document_type')})" if doc.metadata.get('document_type') else ""
        content_type = f" - {doc.metadata.get('content_type')}" if doc.metadata.get('content_type') else ""
        context_parts.append(f"- {content} (Source: {source}{doc_type}{content_type})")
    if price_info:
        price_entries = [f"- {treatment.title()}: {price} BDT" for treatment, price in price_info.items()]
        context_parts.append("\nPrice Information:\n" + "\n".join(price_entries))
    return "\n\n".join(context_parts)

@functools.lru_cache(maxsize=16)
def create_medical_prompt_template(include_price):
    """Create a template for medical prompts (cached)"""
    price_instruction = "Include price in BDT format if available. " if include_price else ""
    return f"""You are a professional medical assistant named MediAssist. Use ONLY the provided context to answer the question.
If the information is not in the context, say "I don't have information about that in my database."

IMPORTANT: Your response must be 3-5 sentences total. Be concise, direct, and professional but conversational.
Use natural language with slight variations in sentence structure. Avoid repetitive patterns.
Do not use hashtag symbols (#) in your responses.
{price_instruction}

Context:
{{context}}

Question: {{question}}

Response (3-5 sentences only, maintain professional but conversational tone):"""

def create_medical_prompt(context, question, include_price):
    """Generate structured prompt with conditional pricing"""
    template = create_medical_prompt_template(include_price)
    return template.format(context=context, question=question)

@functools.lru_cache(maxsize=1)
def create_price_prompt_template():
    """Create a template for price prompts (cached)"""
    return """You are MediAssist, a professional medical pricing assistant. Use ONLY the provided context to answer the question.
If the pricing information is not in the context, say "I don't have pricing information about that in my database."

IMPORTANT: Your response must be 3-5 sentences total. Be concise, direct, and conversational.
Use natural language with slight variations in sentence structure. Avoid repetitive patterns.
Do not use hashtag symbols (#) in your responses.
When mentioning prices, always use the format: X.00 BDT.

Context:
{context}

Question: {question}

Response (3-5 sentences only, maintain professional but conversational tone):"""

def create_price_prompt(context, question):
    """Create a specialized prompt for price inquiries"""
    template = create_price_prompt_template()
    return template.format(context=context, question=question)

def count_sentences(text):
    """Count the number of sentences in text"""
    sentence_count = len(SENTENCE_PATTERN.findall(text))
    return sentence_count

def ensure_sentence_limit(text, max_sentences=5):
    """Ensure the text has no more than the specified number of sentences"""
    sentences = re.split(r'([.!?])\s+', text)
    complete_sentences = []
    i = 0
    sentence_count = 0
    while i < len(sentences) and sentence_count < max_sentences:
        if i + 1 < len(sentences) and sentences[i+1] in ['.', '!', '?']:
            complete_sentences.append(sentences[i] + sentences[i+1])
            i += 2
            sentence_count += 1
        else:
            if sentences[i] and sentences[i] not in ['.', '!', '?']:
                if not sentences[i].strip().endswith(('.', '!', '?')):
                    complete_sentences.append(sentences[i] + '.')
                else:
                    complete_sentences.append(sentences[i])
                sentence_count += 1
            i += 1
    return ' '.join(complete_sentences)

def format_price_response(response, price_info):
    """Format the response to include price information consistently"""
    response = PRICE_FORMAT_PATTERN.sub(r'\1.00 BDT', response)
    response = SPACE_BEFORE_PRICE_PATTERN.sub(' ', response)
    # Remove any remaining hashtags
    response = response.replace('#', '')
    return response

def add_human_touch(response):
    """Add variations to make responses more human-like"""
    # Remove any hashtags
    response = response.replace('#', '')
    
    # Add transitions or acknowledgments at the beginning (30% chance)
    if random.random() < 0.3 and not any(response.startswith(phrase) for phrase in RESPONSE_VARIATIONS["transitions"] + RESPONSE_VARIATIONS["acknowledgments"]):
        response = random.choice(RESPONSE_VARIATIONS["transitions"]) + response[0].lower() + response[1:]
    
    # Replace standard uncertainty phrases with variations
    for phrase in ["I don't have information about that in my database"]:
        if phrase in response:
            response = response.replace(phrase, random.choice(RESPONSE_VARIATIONS["uncertainty"]))
    
    # Replace standard pricing phrases with variations
    for phrase in ["I don't have pricing information about that in my database"]:
        if phrase in response:
            response = response.replace(phrase, random.choice(RESPONSE_VARIATIONS["pricing_unknown"]))
    
    return response

def process_direct_price_query(treatment_query):
    """Process queries that are direct price lookups"""
    # Check cache first
    cache_key = f"price_{normalize_treatment_name(treatment_query)}"
    if cache_key in treatment_cache:
        return treatment_cache[cache_key]
    
    # Use test index for treatment cost lookup with a higher k value
    search_key = f"search_{normalize_treatment_name(treatment_query)}"
    if search_key in vector_search_cache:
        docs_with_scores = vector_search_cache[search_key]
    else:
        docs_with_scores = vector_store_test.similarity_search_with_score(treatment_query, k=20)
        vector_search_cache[search_key] = docs_with_scores
    
    # First attempt: Find match using our custom matching function
    for doc, score in docs_with_scores:
        treatment, price = extract_price_from_text(doc.page_content)
        if treatment and treatments_match(treatment_query, treatment):
            response = f"The cost for {treatment_query.title()} is {price}.00 BDT. This is based on our current healthcare provider data. Remember that prices may vary slightly depending on the facility and any additional services required."
            treatment_cache[cache_key] = response
            return response
    
    # Second attempt: Extract any treatment name that looks like our query
    potential_matches = []
    for doc, score in docs_with_scores:
        treatment, price = extract_price_from_text(doc.page_content)
        if treatment and treatments_match(treatment_query, treatment):
            potential_matches.append((treatment, price, score))
    
    # If we found potential matches, use the one with highest score
    if potential_matches:
        # Sort by score (lower is better)
        potential_matches.sort(key=lambda x: x[2])
        best_match = potential_matches[0]
        response = f"The cost for {treatment_query.title()} is {best_match[1]}.00 BDT. This pricing information is based on our current database. Please note that actual costs may vary depending on specific circumstances and the healthcare facility."
        treatment_cache[cache_key] = response
        return response
    
    # Third attempt: Look for price information in the text directly
    for doc, score in docs_with_scores:
        if treatment_query.lower() in doc.page_content.lower():
            # Try to extract price using a more general pattern
            price_match = GENERAL_PRICE_PATTERN.search(doc.page_content)
            if price_match:
                price = price_match.group(1)
                response = f"The cost for {treatment_query.title()} is {price}.00 BDT. This information is based on our records, but I'd recommend confirming with the healthcare provider for the most current pricing."
                treatment_cache[cache_key] = response
                return response
    
    response = random.choice(RESPONSE_VARIATIONS["pricing_unknown"])
    treatment_cache[cache_key] = response
    return response

def answer_question(question):
    try:
        # First check if we've seen this exact question before
        question_key = question.strip().lower()
        if question_key in response_cache:
            return response_cache[question_key]
        
        start_time = time.time()
        
        # Check if this is a direct price query
        m = BDT_QUERY_PATTERN.match(question)
        if m:
            treatment_query = m.group(1).strip()
            response = process_direct_price_query(treatment_query)
            # Remove any hashtags
            response = response.replace('#', '')
            # Store in cache
            response_cache[question_key] = response
            return response
        
        # For other queries, use the medicalbot index
        if is_about_bot(question):
            response = get_bot_response()
            response_cache[question_key] = response
            return response
        
        # Detect if query is about pricing
        question_lower = question.lower()
        has_price = any(word in question_lower for word in PRICE_KEYWORDS)
        
        # Adjust search parameters based on query type
        search_k = 5 if has_price else 3
        similarity_threshold = 0.55 if has_price else 0.6
        
        # Enhance search query for price-related questions
        search_query = question
        if has_price and not any(term in search_query.lower() for term in ['price', 'cost']):
            search_query += " price cost"
        
        # Check if we have this search cached
        search_cache_key = f"search_{search_query}"
        if search_cache_key in vector_search_cache:
            docs_with_scores = vector_search_cache[search_cache_key]
        else:
            # Perform vector search
            docs_with_scores = vector_store_medicalbot.similarity_search_with_score(search_query, k=search_k)
            vector_search_cache[search_cache_key] = docs_with_scores
            
        filtered_docs = [doc for doc, score in docs_with_scores if score >= similarity_threshold]
        
        # Handle no results
        if not filtered_docs:
            response = random.choice(RESPONSE_VARIATIONS["uncertainty"])
            response_cache[question_key] = response
            return response
        
        # Extract price information if needed
        price_info = {}
        if has_price:
            price_info = extract_price_info(filtered_docs)
        
        # Format context and create prompt
        formatted_context = format_retrieved_context(filtered_docs)
        if has_price:
            prompt = create_price_prompt(formatted_context, question)
        else:
            prompt = create_medical_prompt(formatted_context, question, has_price)
        
        # Generate response with optimized settings
        with llm.chat_session():
            response = llm.generate(
                prompt, 
                max_tokens=200, 
                temp=0.7,  # Slightly higher temperature for more varied responses
                repeat_penalty=1.1
            )
        
        # Post-process response
        response = ensure_sentence_limit(response.strip(), 5)
        if has_price:
            response = format_price_response(response, price_info)
        
        # Add human-like variations and ensure no hashtags
        response = add_human_touch(response)
        response = response.replace('#', '')
        
        # Cache the response
        response_cache[question_key] = response
        
        # Add thinking time for more natural feel (if response was too quick)
        processing_time = time.time() - start_time
        if processing_time < 0.5:
            time.sleep(min(0.5, 0.5 - processing_time))
            
        return response
    
    except Exception as e:
        # More human-like error responses
        error_responses = [
            "I apologize, but I'm having trouble processing your request right now. Could you try rephrasing your question?",
            "Something went wrong while I was retrieving that information. Can you try asking in a different way?",
            "I seem to be having difficulty with that question. Could you try again or ask something else?",
            "I ran into an issue while processing your question. Could you try again with a more specific query?"
        ]
        return random.choice(error_responses)

# Enhanced thread pool with priority queue
class PriorityThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, max_workers=None):
        super().__init__(max_workers=max_workers)
        self.futures = {}
        
    def submit_with_priority(self, fn, *args, priority=0, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        self.futures[future] = priority
        return future

# Use a request queue to handle concurrent requests better
executor = PriorityThreadPoolExecutor(max_workers=8)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Check if question is about pricing (higher priority)
    has_price = any(word in question.lower() for word in PRICE_KEYWORDS)
    priority = 1 if has_price else 0
    
    # Submit the question to the thread pool with priority
    future = executor.submit_with_priority(answer_question, question, priority=priority)
    answer = future.result()
    
    # Final check to ensure no hashtags in the response
    answer = answer.replace('#', '')
    
    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy", "model": BOT_INFO["name"], "version": BOT_INFO["version"]})

if __name__ == "__main__":
    # Use production WSGI server in production
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)