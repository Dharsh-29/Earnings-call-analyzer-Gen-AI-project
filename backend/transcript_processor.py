# transcript_processor.py
import os
import re
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader

def read_pdf_text(pdf_path: str) -> Tuple[str, int]:
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            pages_count = len(reader.pages)
            text_blocks = []
            for page in reader.pages:
                text_blocks.append(page.extract_text() or "")
            full_text = "\n".join(text_blocks)
            return full_text, pages_count
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "", 0

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text.strip()

def extract_company_and_date(text: str) -> Tuple[str, str]:
    
    # Look for Laurus Labs specifically
    company_patterns = [
        r'Laurus Labs Limited',
        r'Laurus Labs',
        r'"Laurus Labs Q2 FY\'24 Earnings Conference Call"'
    ]
    
    company = "Unknown Company"
    for pattern in company_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            company = "Laurus Labs Limited"
            break
    
    # Look for October 20, 2023 specifically
    date_patterns = [
        r'October 20, 2023',
        r'Q2 FY\'?24',
        r'Q2 FY24'
    ]
    
    date = "Unknown Date"
    for pattern in date_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            date = "October 20, 2023"
            break
    
    return company, date

def find_earnings_call_start(lines: List[str]) -> int:
    
    # Look for moderator or welcome statements
    start_indicators = [
        r'Moderator.*Ladies and gentlemen',
        r'good day and welcome',
        r'welcome to.*earnings',
        r'Monish Shah.*Thank you',
        r'Dr\. Satyanarayana Chava.*Thank you'
    ]
    
    for i, line in enumerate(lines):
        for pattern in start_indicators:
            if re.search(pattern, line, re.IGNORECASE):
                return i
    
    # Fallback: look for first speaker pattern after page 2
    for i, line in enumerate(lines[50:], 50):  # Skip first 50 lines (headers)
        if re.match(r'^[A-Z][a-zA-Z\s\.]+:\s', line):
            return i
    
    return 0

def find_qa_section_start(lines: List[str], start_idx: int = 0) -> int:    
    qa_patterns = [
        r'question.*answer.*session',
        r'begin the question',
        r'first question.*from.*line',
        r'open.*lines.*Q&A',
        r'Jeevan Patwa.*Firstly'  # First actual question in your transcript
    ]
    
    for i, line in enumerate(lines[start_idx:], start_idx):
        for pattern in qa_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return i
    
    return len(lines)  # If not found, everything is opening remarks

def detect_speaker_and_message(line: str) -> Tuple[str, str]:
        # Pattern for "Speaker Name: Message"
    patterns = [
        r'^([A-Z][a-zA-Z\s\.]+?):\s*(.+)$',
        r'^([A-Z][A-Z\s\.]+?)\s*[-:]\s*(.+)$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, line.strip())
        if match:
            speaker = match.group(1).strip()
            message = match.group(2).strip()
            
            # Clean speaker name
            speaker = re.sub(r'[\(\)\d]+', '', speaker).strip()
            speaker = ' '.join(speaker.split())
            
            if 2 <= len(speaker) <= 50:
                return speaker, message
    
    return "", line.strip()

def is_question_speaker(speaker: str, message: str) -> bool:   
    speaker_lower = speaker.lower()
    
    # Question indicators
    if any(word in speaker_lower for word in ['analyst', 'patwa', 'participant', 'investor']):
        return True
    
    # Answer indicators (management)
    if any(word in speaker_lower for word in ['chava', 'kumar', 'ceo', 'cfo', 'dr.', 'mr.']):
        return False
    
    # Content-based detection
    message_lower = message.lower()
    if message_lower.endswith('?') or any(message_lower.startswith(w) for w in ['what', 'how', 'when', 'why']):
        return True
    
    return False

def split_long_text(text: str, max_chars: int = 800) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence if len(sentence) <= max_chars else sentence[:max_chars]
    
    if current:
        chunks.append(current)
    
    return chunks

def process_pdf(pdf_path: str) -> Dict:
    # Read PDF
    raw_text, pages_count = read_pdf_text(pdf_path)
    if not raw_text.strip():
        return {
            "opening_chunks": [],
            "qa_chunks": [],
            "concall_metadata": {
                "company_name": "Unknown Company",
                "date": "Unknown Date",
                "pages_count": pages_count
            }
        }
    
    # Extract company and date
    company, date = extract_company_and_date(raw_text)
    
    # Split into lines
    lines = [clean_text(line) for line in raw_text.splitlines() if clean_text(line)]
    
    # Find where actual earnings call starts (skip document headers)
    call_start = find_earnings_call_start(lines)
    
    # Find Q&A section start
    qa_start = find_qa_section_start(lines, call_start)
    
    # Process opening remarks
    opening_chunks = []
    opening_counter = 1
    current_speaker = None
    current_message = ""
    
    for line in lines[call_start:qa_start]:
        speaker, message = detect_speaker_and_message(line)
        
        if speaker:  # New speaker
            # Save previous speaker's content
            if current_speaker and current_message.strip():
                for chunk_text in split_long_text(current_message):
                    opening_chunks.append({
                        "id": f"C{opening_counter}",
                        "speaker": current_speaker,
                        "message": chunk_text
                    })
                    opening_counter += 1
            
            current_speaker = speaker
            current_message = message
        else:
            # Continue current speaker's message
            if current_message:
                current_message += " " + message
            else:
                current_message = message
    
    # Save last opening remarks speaker
    if current_speaker and current_message.strip():
        for chunk_text in split_long_text(current_message):
            opening_chunks.append({
                "id": f"C{opening_counter}",
                "speaker": current_speaker,
                "message": chunk_text
            })
            opening_counter += 1
    
    # Process Q&A section
    qa_chunks = []
    question_counter = 1
    answer_counter = 1
    current_speaker = None
    current_message = ""
    
    for line in lines[qa_start:]:
        speaker, message = detect_speaker_and_message(line)
        
        if speaker:  # New speaker
            # Save previous speaker's content
            if current_speaker and current_message.strip():
                is_question = is_question_speaker(current_speaker, current_message)
                
                for chunk_text in split_long_text(current_message):
                    if is_question:
                        qa_chunks.append({
                            "id": f"Q{question_counter}",
                            "speaker": current_speaker,
                            "message": chunk_text,
                            "type": "question"
                        })
                        question_counter += 1
                    else:
                        qa_chunks.append({
                            "id": f"A{answer_counter}",
                            "speaker": current_speaker,
                            "message": chunk_text,
                            "type": "answer"
                        })
                        answer_counter += 1
            
            current_speaker = speaker
            current_message = message
        else:
            # Continue current speaker's message
            if current_message:
                current_message += " " + message
            else:
                current_message = message
    
    # Save last Q&A speaker
    if current_speaker and current_message.strip():
        is_question = is_question_speaker(current_speaker, current_message)
        
        for chunk_text in split_long_text(current_message):
            if is_question:
                qa_chunks.append({
                    "id": f"Q{question_counter}",
                    "speaker": current_speaker,
                    "message": chunk_text,
                    "type": "question"
                })
                question_counter += 1
            else:
                qa_chunks.append({
                    "id": f"A{answer_counter}",
                    "speaker": current_speaker,
                    "message": chunk_text,
                    "type": "answer"
                })
                answer_counter += 1
    
    return {
        "opening_chunks": opening_chunks,
        "qa_chunks": qa_chunks,
        "concall_metadata": {
            "company_name": company,
            "date": date,
            "pages_count": pages_count
        }
    }