import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_markdown_file(file_path):
    """
    Parses a markdown file to extract its front-matter metadata and its content.
    Returns:
        metadata (dict): Key-value pairs from the front-matter.
        content (str): The body text of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    metadata = {}
    content = text
    
    # Check for front-matter (delimited by --- at the start)
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            front_matter_text = parts[1]
            content = parts[2]
            
            # Simple YAML-like parser
            for line in front_matter_text.strip().split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    metadata[k.strip()] = v.strip()
                    
    return metadata, content

def split_into_chunks(filename, metadata, content):
    """
    Splits the content of a markdown document into chunks based on headers.
    Returns:
        list of dicts, where each dict is a chunk:
        {
            "filename": filename,
            "heading": current_heading,
            "metadata": metadata,
            "text": text_content,
            "search_text": header + text_content (for better vector matching)
        }
    """
    chunks = []
    lines = content.split('\n')
    current_heading = "General"
    current_lines = []
    
    for line in lines:
        # Match lines like "# Header" or "## Sub-header"
        match = re.match(r'^(#{1,6})\s+(.*)$', line)
        if match:
            # Save the previous chunk if it has content
            if current_lines:
                text_content = '\n'.join(current_lines).strip()
                if text_content:
                    chunks.append({
                        "filename": filename,
                        "heading": current_heading,
                        "metadata": metadata,
                        "text": text_content,
                        "search_text": f"{current_heading}\n{text_content}"
                    })
            # Start new chunk
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
            
    # Add final chunk
    if current_lines:
        text_content = '\n'.join(current_lines).strip()
        if text_content:
            chunks.append({
                "filename": filename,
                "heading": current_heading,
                "metadata": metadata,
                "text": text_content,
                "search_text": f"{current_heading}\n{text_content}"
            })
            
    return chunks

class KnowledgeRetriever:
    def __init__(self, kb_dir='knowledge-base'):
        self.kb_dir = kb_dir
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.load_and_index()
        
    def load_and_index(self):
        """Loads all markdown files, splits them, and builds the TF-IDF matrix."""
        all_chunks = []
        for file in os.listdir(self.kb_dir):
            if file.endswith('.md'):
                path = os.path.join(self.kb_dir, file)
                metadata, content = parse_markdown_file(path)
                file_chunks = split_into_chunks(file, metadata, content)
                all_chunks.extend(file_chunks)
        
        self.chunks = all_chunks
        
        # Fit TF-IDF vectorizer on search_text (heading + content)
        search_texts = [c['search_text'] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(search_texts)
        
    def retrieve(self, query, top_k=3):
        """
        Retrieves relevant passages, applying metadata penalties for policy precedence.
        """
        # 1. Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])
        
        # 2. Calculate cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 3. Apply Metadata Penalty (Policy Precedence)
        # Check if the query is asking about legacy/past rules
        is_legacy_query = any(word in query.lower() for word in ['legacy', 'old', 'past', 'superseded', '2024', 'before'])
        
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            base_score = similarities[i]
            score = base_score
            
            # Penalize superseded or draft documents unless legacy was asked
            status = chunk['metadata'].get('status', 'active')
            authority = chunk['metadata'].get('policy_authority', 'official')
            
            if not is_legacy_query:
                if status == 'superseded':
                    score *= 0.1 # Strongly penalize legacy
                elif status == 'draft' or authority == 'none':
                    score *= 0.05 # Ignore migration scratchpad or draft policies
            
            scored_chunks.append((score, base_score, chunk))
            
        # 4. Sort and return top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return scored_chunks[:top_k]

# Simple test code to verify retriever works
if __name__ == '__main__':
    retriever = KnowledgeRetriever()
    print(f"Loaded {len(retriever.chunks)} chunks.")
    
    test_query = "What is the return window for a normal customer?"
    print(f"\nQuery: {test_query}")
    results = retriever.retrieve(test_query, top_k=2)
    for score, base_score, chunk in results:
        print(f"\nScore: {score:.4f} (Base: {base_score:.4f})")
        print(f"Source: {chunk['filename']} - {chunk['heading']}")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Snippet: {chunk['text'][:150]}...")
