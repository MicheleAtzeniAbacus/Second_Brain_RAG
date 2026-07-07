# (c) respects meaning and blows the size
from typing import NamedTuple

class ChunkedDocs(NamedTuple): 
    article: str
    section: str
    text: str

def chunk_structure(title: str, text: str) -> list[ChunkedDocs[str, str, str]]: 

    def flush(str_chunks: list[tuple[str, str, str]], header: str, body_text: list[str], title: str): 
        str_chunks.append(ChunkedDocs(article = title, section = header, text = f"Article: {title} > Section: {header}\n{'\n'.join(body_text)}"))
    
    structured_chunks = []
    current_text = [] 
    current_header = "Intro" 
    for line in text.splitlines():
        if line.startswith("## "):
            if any(current_text): flush(structured_chunks, current_header, current_text, title)
            # all text before the first ## will be with intro header
            current_text = []
            current_header = line.removeprefix("## ")
        else: 
            current_text.append(line)
    
    # final flush
    if any(current_text): flush(structured_chunks, current_header, current_text, title)
    return structured_chunks