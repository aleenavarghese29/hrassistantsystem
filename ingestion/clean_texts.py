import os
import re

INPUT_DIR = r"C:/Users/Aleena/Documents/hr_assistant/cleaned_texts"


# 1. Clean text while preserving critical values (detective work!)
def clean_text(text):
    """
    Preserve: time values, decimals, ranges, percentages, paths
    Remove: unnecessary symbols and excessive whitespace
    """
    text = text.lower()

    # normalize horizontal whitespace but preserve newlines (structure!)
    text = re.sub(r"[ \t]+", " ", text)

    # remove unwanted characters but keep essential punctuation
    # keeping: letters, digits, spaces, newlines, . , : % ( ) / -
    text = re.sub(r"[^\w\s.,:%()/\-\n]", "", text)

    # normalize excessive newlines (max 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# 2. Ensure headings are properly formatted and isolated
def normalize_headings(text):
    """
    Headings MUST:
    - Start with numbers (1., 2.1, 10.3.2)
    - Be on their own line
    - NEVER have bullets
    """
    lines = text.split("\n")
    output = []

    for line in lines:
        stripped = line.strip()
        
        # preserve empty lines for structure
        if not stripped:
            output.append("")
            continue

        # detect section headings: starts with digit pattern
        # pattern matches: "1. intro", "2.1 policy", "10.3.2 subsection"
        if re.match(r"^\d+(\.\d+)*\.?\s+\w", stripped):
            # headings are always on their own line, no bullets
            output.append(stripped)
        else:
            output.append(stripped)

    return "\n".join(output)


# 3. Convert appropriate content to bullet lists
def normalize_lists(text):
    """
    Rules:
    - Lines after ":" become bullets
    - Numbered headings ALWAYS terminate bullet context
    - Empty lines terminate bullet context
    """
    lines = text.split("\n")
    output = []
    in_list_context = False

    for line in lines:
        stripped = line.strip()

        # empty line: break list context and preserve spacing
        if not stripped:
            output.append("")
            in_list_context = False
            continue

        # CRITICAL: numbered heading ALWAYS breaks list and is NEVER bulleted
        if re.match(r"^\d+(\.\d+)*\.?\s+\w", stripped):
            output.append(stripped)
            in_list_context = False
            continue

        # line ending with colon: marks start of list
        if stripped.endswith(":"):
            output.append(stripped)
            in_list_context = True
            continue

        # in list context: add bullet if not already present
        if in_list_context:
            if not stripped.startswith("-"):
                output.append("- " + stripped)
            else:
                output.append(stripped)
        else:
            # regular content line
            output.append(stripped)

    return "\n".join(output)


# 4. Separate policy content from metadata
def split_disclaimer(text):
    """
    Extract metadata (disclaimer, version info, dates) 
    from main policy content for cleaner chunking
    """
    # metadata markers (case insensitive)
    metadata_pattern = r"\n(?=disclaimer|document version|last updated|effective date|approved by|revision history)"
    
    parts = re.split(metadata_pattern, text, maxsplit=1, flags=re.IGNORECASE)

    main_text = parts[0].strip()
    metadata = parts[1].strip() if len(parts) > 1 else ""

    return main_text, metadata


# 5. Process all files in directory
def process_files():
    """
    Main pipeline: clean → normalize headings → create lists → split metadata
    """
    # validate directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory not found: {INPUT_DIR}")
        return

    # get all text files (exclude metadata files)
    files = [f for f in os.listdir(INPUT_DIR) 
             if f.endswith(".txt") and not f.endswith("_metadata.txt")]
    
    if not files:
        print("No .txt files found to process.")
        return

    print(f"Found {len(files)} file(s) to process\n")

    # process each file
    success_count = 0
    for file in files:
        path = os.path.join(INPUT_DIR, file)

        try:
            # read original file
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # apply cleaning pipeline 
            cleaned = clean_text(raw_text)
            structured = normalize_headings(cleaned)
            formatted = normalize_lists(structured)

            # separate policy from metadata
            policy_text, metadata = split_disclaimer(formatted)

            # write cleaned policy back
            with open(path, "w", encoding="utf-8") as f:
                f.write(policy_text)

            # write metadata to separate file if exists
            if metadata:
                meta_path = path.replace(".txt", "_metadata.txt")
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(metadata)
                print(f"{file} → cleaned + metadata extracted")
            else:
                print(f"{file} → cleaned")

            success_count += 1

        except Exception as e:
            print(f"{file} → Error: {str(e)}")

    # summary
    print(f"\nSuccessfully processed {success_count}/{len(files)} files")
    print("Files are now optimized for RAG chunking and embeddings")


if __name__ == "__main__":
    
    process_files()

    print("Processing complete!")
   