# chatbot.py
import os
import openai
from vectorstore import search_chunks
from config import config

# Set OpenAI configuration
openai.api_key = config.OPENAI_API_KEY
if config.OPENAI_ORG_ID:
    openai.organization = config.OPENAI_ORG_ID

def ask_question(question, chunks, k=5):
    if not question.strip():
        return "Please enter a valid question.", []
    
    if not chunks:
        return "No transcript loaded.", []
    
    if not config.OPENAI_API_KEY:
        return "OpenAI API key not configured. Please check your environment variables.", []

    try:
        # Search for relevant transcript parts
        relevant_chunks = search_chunks(question, chunks, top_k=k)
        if not relevant_chunks:
            return "No relevant information found in the transcript for your question.", []

        # Build context from relevant chunks
        context_parts = []
        for i, (text, score) in enumerate(relevant_chunks):
            # Clean and truncate text if needed
            clean_text = text.replace('\n', ' ').strip()
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."
            context_parts.append(f"[Source {i+1}, Relevance: {score:.2f}]\n{clean_text}")

        context = "\n\n".join(context_parts)
        answer = generate_answer_with_llm(question, context)
        
        return answer, relevant_chunks

    except Exception as e:
        error_message = f"Error processing your question: {str(e)}"
        print(f"Chatbot error: {e}")  # For debugging
        return error_message, []

def generate_answer_with_llm(question, context):
    system_prompt = """You are an AI assistant specialized in analyzing earnings call transcripts. 

Key Guidelines:
1. ONLY use information from the provided transcript context to answer questions
2. Include specific speaker names, numbers, financial figures, and dates when available
3. Be concise but comprehensive - aim for 2-4 sentences unless more detail is needed
4. If the context doesn't contain relevant information, clearly state that
5. Provide direct quotes when they support your answer
6. Focus on factual information from the earnings call

Answer format:
- Start with a direct answer to the question
- Support with specific details from the transcript
- Include relevant financial metrics, percentages, or targets mentioned
- Mention the speaker's name when providing key insights"""

    user_prompt = f"""
Question: {question}

Earnings Call Transcript Context:
{context}

Please provide a comprehensive answer based solely on the information in the transcript context above.
"""

    try:
        # Use the new OpenAI API format
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Add confidence indicator based on context quality
        if len(context) > 2000:
            confidence = "High"
        elif len(context) > 1000:
            confidence = "Medium" 
        else:
            confidence = "Low"
            
        return f"{answer}\n\n*Confidence Level: {confidence} (based on available context)*"
        
    except openai.APIError as e:
        return f"OpenAI API error: {str(e)}. Please check your API key and try again."
    except openai.RateLimitError as e:
        return "Rate limit exceeded. Please wait a moment and try again."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def get_sample_questions():
    return [
        "What was the revenue growth this quarter?",
        "What are the key challenges mentioned by management?", 
        "Any updates on new product launches or approvals?",
        "What is the company's outlook for next quarter?",
        "Who are the key management personnel speaking?",
        "What were the main financial highlights?",
        "Any discussion about market competition?",
        "What are the company's future strategic plans?",
        "What was the EBITDA margin performance?",
        "Any guidance provided for the full year?",
        "What are the key risks or challenges ahead?",
        "Any updates on regulatory approvals?"
    ]

def format_chat_response(answer, chunks_used):
    if not chunks_used:
        return answer
    
    formatted_response = f"{answer}\n\n**Sources Used:**\n"
    
    for i, (chunk_text, score) in enumerate(chunks_used[:3]):
        preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        formatted_response += f"\n{i+1}. (Relevance: {score:.2f}) {preview}\n"
    
    return formatted_response

def validate_question(question):
    if not question or len(question.strip()) < 5:
        return False, "Question is too short. Please provide a more detailed question."
    
    if len(question) > 500:
        return False, "Question is too long. Please keep it under 500 characters."
    
    # Check for inappropriate content (basic filter)
    inappropriate_words = ["hack", "break", "exploit", "personal", "private"]
    if any(word in question.lower() for word in inappropriate_words):
        return False, "Please ask questions related to the earnings call content."
    
    return True, "Question is valid."