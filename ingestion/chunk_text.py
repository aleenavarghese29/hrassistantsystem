import os
import re
import json

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "cleaned_texts"
OUTPUT_DIR = BASE_DIR / "chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Remove metadata/disclaimer from text
def remove_metadata(text):
    """
    Remove disclaimers, version info, and other metadata
    that shouldn't be in policy chunks
    """
    metadata_markers = [
        r"\n\s*disclaimer\b",
        r"\n\s*document version\b",
        r"\n\s*last updated\b",
        r"\n\s*effective date\b",
        r"\n\s*approved by\b",
        r"\n\s*revision history\b",
        r"\n\s*contact\b",
        r"\n\s*for\s+(?:any|more)\s+(?:queries|information|details)\b"
    ]
    
    earliest_pos = len(text)
    for marker in metadata_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
    
    policy_text = text[:earliest_pos].strip()
    return policy_text


# 2. FIXED: Extract ALL sections with proper boundaries
def chunk_policy(policy_text, policy_name):
    """
    Split policy into section-based chunks.
    Each chunk contains EXACTLY ONE section with complete content.
    
    FIXES:
    - Captures ALL sections (no skipping)
    - Proper boundary detection (no bleeding)
    - Preserves complete content (no truncation)
    """
    # Match section headings: "1. title" or "2.1 title" or "10.3.2 title"
    section_pattern = re.compile(
        r"^(\d+(?:\.\d+)*\.?)\s+(.+)$",
        re.MULTILINE
    )

    matches = list(section_pattern.finditer(policy_text))
    chunks = []

    if not matches:
        print(f"  WARNING: No sections found in {policy_name}")
        return chunks

    for i, match in enumerate(matches):
        section_start = match.start()
        
        # CRITICAL FIX: Proper boundary detection
        if i + 1 < len(matches):
            next_section_start = matches[i + 1].start()
            
            # Look for last newline before next section
            # But ensure we don't cut off content
            check_start = max(section_start, next_section_start - 200)  # Look back up to 200 chars
            section_end = policy_text.rfind('\n', check_start, next_section_start)
            
            # If no newline found, or it's too close to current section, use next section start
            if section_end == -1 or section_end <= section_start + 10:
                section_end = next_section_start
        else:
            # Last section: goes to end of document
            section_end = len(policy_text)

        section_id = match.group(1).rstrip('.')
        section_title = match.group(2).strip()
        
        # Extract section content with proper boundaries
        section_text = policy_text[section_start:section_end].strip()
        
        # Clean the section text
        section_text = clean_section_text(section_text)
        
        # FIX: Less strict validation - capture all sections
        if not is_valid_section(section_text, section_id, section_title):
            print(f"  WARNING: Skipping potentially invalid section {section_id}: {section_title}")
            continue

        chunks.append({
            "policy_name": policy_name,
            "section_id": section_id,
            "section_title": section_title,
            "text": section_text,
            "chunk_id": f"{policy_name}_{section_id}".replace(" ", "_").replace("__", "_")
        })

    return chunks


# 3. Clean individual section text
def clean_section_text(text):
    """
    Final cleanup for section text:
    - Remove excessive whitespace
    - Remove footer/header noise
    - Remove contact info
    - Preserve structure and content
    """
    # Remove common footer/header noise
    noise_patterns = [
        r"page\s+\d+\s+of\s+\d+",
        r"\bconfidential\b",
        r"\binternal\s+use\s+only\b",
        r"\bproprietary\s+information\b",
        r"\bemail\s*:?\s*[\w\.\-]+@[\w\.\-]+",
        r"\bphone\s*:?\s*[\d\-\(\)\s]+",
        r"\btel\s*:?\s*[\d\-\(\)\s]+",
        r"\bcontact\s*:?\s*[\d\-\(\)\s]+"
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Normalize spaces within lines (but preserve newlines)
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # Normalize excessive newlines (max 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# 4. FIXED: More lenient validation - don't reject valid sections
def is_valid_section(text, section_id, section_title):
    """
    Validate that section has meaningful content.
    
    FIX: Less strict validation to capture all sections.
    Only reject truly empty or malformed sections.
    """
    # Must have the heading plus some content
    heading_length = len(section_id) + len(section_title)
    min_content_length = 20  # At least 20 chars beyond heading
    
    if len(text) < heading_length + min_content_length:
        return False
    
    # Section must have at least 2 lines (heading + content)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return False
    
    # REMOVED: Strict punctuation check - sections can end with colons (list introducers)
    # This was causing valid sections to be rejected
    
    return True


# 5. Process all cleaned files
def process_all_files():
    """
    Main processing pipeline
    """
    all_chunks = []
    file_stats = {}
    
    files = [f for f in os.listdir(CLEANED_DIR) 
             if f.endswith(".txt") and not f.endswith("_metadata.txt")]
    
    if not files:
        print(" No files found to process")
        return []
    
    print(f"Processing {len(files)} policy files...\n")
    
    for file in files:
        file_path = os.path.join(CLEANED_DIR, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Remove metadata before chunking
            policy_text = remove_metadata(raw_text)

            # Normalize policy name (remove extra spaces)
            policy_name = file.replace(".txt", "").replace("_", " ")
            policy_name = re.sub(r'\s+', ' ', policy_name).strip()

            # Create chunks with corrected logic
            chunks = chunk_policy(policy_text, policy_name)
            
            all_chunks.extend(chunks)
            file_stats[file] = len(chunks)

            print(f"{file}")
            print(f"  → {len(chunks)} chunks created")

        except Exception as e:
            print(f"{file}: Error - {str(e)}")
            file_stats[file] = 0

    # Summary
  
    print(f"CHUNKING SUMMARY")
  
    for file, count in file_stats.items():
        print(f"{file}: {count} chunks")
    
    return all_chunks


# 6. Save chunks with metadata
def save_chunks(chunks):
    """
    Save chunks to JSON with metadata
    """
    output_file = os.path.join(OUTPUT_DIR, "policy_chunks.json")

    # Add sequential IDs
    for idx, chunk in enumerate(chunks):
        chunk["global_id"] = idx

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    
    print(f"Total chunks saved: {len(chunks)}")
    print(f"Output file: {output_file}")
 
    
    # Print sample chunks for verification
    if chunks:
        print("\nSample chunk (first):")
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
        
        if len(chunks) > 1:
            print("\nSample chunk (last):")
            print(json.dumps(chunks[-1], indent=2, ensure_ascii=False))


# 7. Main execution
if __name__ == "__main__":
    
    print("RAG Chunk Generator - Section-Aware (FIXED)")

    print()
    
    all_chunks = process_all_files()
    
    if all_chunks:
        save_chunks(all_chunks)
        print("\nChunking complete!")
    else:
        print("\nNo chunks generated. Check your input files.")