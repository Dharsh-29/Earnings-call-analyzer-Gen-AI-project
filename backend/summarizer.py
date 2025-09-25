# summarizer.py
import os
import openai
import re
import random
from typing import List, Dict
from config import config

# Set up OpenAI client
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

def extract_text_from_chunks(chunks: List[Dict]) -> str:
    return " ".join([c.get("message", "") for c in chunks if c.get("message")])

def generate_topics(chunks: List[Dict], max_topics: int = 5, regenerate: bool = False) -> List[Dict]:
    text = extract_text_from_chunks(chunks)
    if not text:
        return []

    # Limit text to avoid token issues
    text_excerpt = text[:4000]
    
    # Add variety to prompts for regeneration
    focus_angles = [
        "business strategy and financial performance",
        "market dynamics and competitive positioning", 
        "operational excellence and growth initiatives",
        "regulatory environment and risk factors",
        "innovation and future opportunities"
    ]
    
    if regenerate:
        focus = random.choice(focus_angles)
        variety_instruction = f"Focus particularly on {focus}. Provide fresh perspectives and different topic angles than typical earnings call analysis."
    else:
        focus = "key business themes"
        variety_instruction = "Focus on the most important business topics."

    prompt = f"""
You are an expert earnings call analyst. 
Analyze the following transcript section and extract the {max_topics} most important business topics.
{variety_instruction}

For each topic, provide a clear topic name and brief description.

Guidelines:
- Make topic names specific and actionable (not generic)
- Focus on concrete business matters: financials, strategy, operations, market conditions
- Avoid generic topics like "Management Discussion" or "Company Update"
- Include specific numbers, metrics, or initiatives when possible
- Prioritize topics that investors would care about most

Format your response as:
Topic Name: Description

Transcript Section:
{text_excerpt}
"""

    try:
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4 if regenerate else 0.2,  # Higher temperature for variety
            max_tokens=600,
            top_p=0.9
        )
        
        raw_response = response.choices[0].message.content.strip()
        topics = []

        # Parse response - expect "Topic: Description" format
        lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
        
        for line in lines:
            if ':' in line and len(topics) < max_topics:
                parts = line.split(':', 1)
                topic_name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                # Clean up topic name - remove numbering
                topic_name = re.sub(r'^\d+\.\s*', '', topic_name)
                topic_name = re.sub(r'^[\-\*]\s*', '', topic_name)
                
                if topic_name and len(topic_name) > 3:
                    topics.append({
                        "topic": topic_name,
                        "description": description
                    })

        return topics

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        if "insufficient_quota" in str(e).lower() or "billing" in str(e).lower():
            print("⚠️ This looks like a billing/quota issue. Please check your OpenAI account billing.")
        return _get_fallback_topics()
    except Exception as e:
        print(f"Error generating topics: {e}")
        return _get_fallback_topics()

def _get_fallback_topics():
    return [
        {"topic": "Financial Performance", "description": "Revenue, margins and key financial metrics"},
        {"topic": "Business Strategy", "description": "Strategic initiatives and business direction"},
        {"topic": "Market Outlook", "description": "Industry trends and future expectations"},
        {"topic": "Operational Updates", "description": "Business operations and efficiency measures"},
        {"topic": "Risk Factors", "description": "Challenges and risk mitigation strategies"}
    ]

def generate_summary(chunks: List[Dict], topic: str) -> str:
    text = extract_text_from_chunks(chunks)
    if not text:
        return "No content available to summarize."

    # Limit text to avoid token issues  
    text_excerpt = text[:4000]

    prompt = f"""
You are an expert earnings call analyst. 
Create a comprehensive summary of the transcript content related to the topic: "{topic}".

Guidelines:
- Focus only on information directly related to the specified topic
- Include specific numbers, percentages, and financial figures mentioned
- Mention relevant speaker names when providing key insights
- Keep the summary concise but informative (3-5 sentences)
- Use professional business language
- If multiple perspectives are mentioned, include them
- Highlight any forward-looking statements or guidance

Topic: {topic}

Transcript Content:
{text_excerpt}

Provide a detailed summary based solely on the transcript content:
"""

    try:
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        if "insufficient_quota" in str(e).lower() or "billing" in str(e).lower():
            return f"⚠️ Unable to generate summary for {topic}. Please check your OpenAI account billing and ensure you have sufficient credits."
        return f"Unable to generate summary for {topic}. API Error: {str(e)}"
    except Exception as e:
        print(f"Error generating summary for topic '{topic}': {e}")
        return f"Unable to generate summary for {topic}. Please check your OpenAI API configuration."

def test_openai_connection():
    """Test if OpenAI API is working"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API connection successful'"}],
            max_tokens=10
        )
        return True, "Connection successful"
    except openai.APIError as e:
        return False, f"API Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"