import json
import re
import unicodedata

def clean_text(text):
    """ Cleans the input text by removing unwanted characters and normalizing whitespace. """

    # Remove soft hyphens (U+00AD) and other invisible formatting marks
    text = text.replace('\u00ad', '')  # Soft hyphen
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\ufeff', '')  # Zero-width no-break space (BOM)

    # Remove control characters (except newline, tab, carriage return)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')

    # Normalize unicode characters to NFC form
    text = unicodedata.normalize('NFKC', text)

    # Remove emojis
    text = ''.join(char for char in text if unicodedata.category(char) not in ['So', 'Cs'])

    # Normalize em-dash and en-dash to regular hyphen
    text = text.replace('—', '-')  # Em dash
    text = text.replace('–', '-')  # En dash

    # Normalize backticks and curly quotes to straight quotes
    text = text.replace('``', '"')
    text = text.replace('`', "'")
    text = text.replace('“', '"')  # Curly left double quote
    text = text.replace('”', '"')  # Curly right double quote
    text = text.replace('‘', "'")  # Curly left single quote
    text = text.replace('’', "'")  # Curly right single quote

    # Handle double dash: convert -- to - with spaces on both sides
    text = re.sub(r'--+', ' - ', text)  # Double dash or more → single dash with spaces

    # Normalize spaced ellipsis: . . . -> ...
    text = re.sub(r'\.\s+\.\s+\.', '...', text)

    # Normalize ellipsis: handle .. and ... (multiple dots), keep three dots together
    # First convert 2+ dots to a placeholder, then handle spacing, then convert back
    text = re.sub(r'\.{2,}', '<ELLIPSIS>', text)  # Two or more dots → placeholder
    text = re.sub(r'(\S)<ELLIPSIS>', r'\1 <ELLIPSIS>', text)  # Add space before if needed
    text = re.sub(r'<ELLIPSIS>(\S)', r'<ELLIPSIS> \1', text)  # Add space after if needed

    # Collapse spaced exclamation marks and multiple exclamation marks to single !
    text = re.sub(r'(?:!\s+)+!', '!', text)
    text = re.sub(r'!+', '!', text)

    # Collapse multiple question marks to single ?
    text = re.sub(r'\?+', '?', text)

    # Space separate punctuation from words
    # Add space before punctuation if not already there (but not commas between digits)
    text = re.sub(r'(\D)([.!?,:;])', r'\1 \2', text)  # \D = not a digit
    # Add space after punctuation if not already there (and not end of string), but not commas between digits
    text = re.sub(r'([.!?,:;])(?!\d)', r'\1 ', text)  # Negative lookahead for digit after comma
    # Space separate parentheses and brackets from words
    text = re.sub(r'(\S)([()\[\]])', r'\1 \2', text)
    text = re.sub(r'([()\[\]])(\S)', r'\1 \2', text)

    # Space separate quotes when they are not apostrophes inside a word
    text = text.replace('"', ' " ')  # Always separate double quotes
    text = re.sub(r"(?<!\w)'(?!\w)", " ' ", text)
    text = re.sub(r"(?<!\w)'(?=\w)", " ' ", text)
    text = re.sub(r"(?<=\w)'(?!\w)", " ' ", text)

    # Convert ellipsis placeholder back to ...
    text = text.replace('<ELLIPSIS>', '...')

    # Remove excessive whitespace BEFORE handling dashes
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space

    # Handle dashes: convert dash with spaces on either side to <DASH>, leave no-space dashes as is
    text = re.sub(r' - ', ' <DASH> ', text)  # Space on both sides

    # Final whitespace cleanup
    text = re.sub(r'\n\n+', '\n\n', text)  # Excessive newlines → max 2
    text = re.sub(r'\t+', ' ', text)  # Tabs → single space

    text = text.strip()  # Remove leading/trailing whitespace
    return text

def process_file(input_file, output_file):
    """Process JSONL file and write cleaned data to output file."""
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    data['text'] = clean_text(data['text'])
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 10000 == 0:
                        print(f"Processed {count} lines...")
            except json.JSONDecodeError:
                continue
    print(f"Total lines processed: {count}")
    print(f"Cleaned data written to {output_file}")

if __name__ == "__main__":
    # Test samples
    test_sample = [
        "Hello World! This is a test ... haha\n\n\nNew line here.\t\tExtra tabs.",
        "Another example with invisible characters.\n\n\n\nEnd of test.",
        "em-dash -- should be cleaned properly...   with spaces.",
        "Multiple!!! exclamation marks!!!!",
        "Test - with - dashes and-compound-words",
        "Ellipsis. . .test 'and' more..dots",
        "Double dash--test and -- more -- tests",
        "Two dots..here and...three dots....many dots",
        "Spaced dots . . . and spaced ! ! ! and backticks `like this` and ``quote``",
        "Curly quotes: (“hello” and ‘world’ should be straight)",
        "a,b,c and 100,000 should not have spaces around commas, but a,b,c should become a , b , c"
    ]

    print("Testing cleaning function:")
    print("=" * 60)
    for i, sample in enumerate(test_sample, 1):
        print(f"\nTest {i}:")
        print(f"Original: {sample}")
        print(f"Cleaned:  {clean_text(sample)}")

    print("\n" + "=" * 60)
    print("\nProcessing English dataset...")
    process_file("cc100_en.jsonl", "cc100_en_cleaned.jsonl")

    # print("\nProcessing Mongolian dataset...")
    # process_file("cc100_mn.jsonl", "cc100_mn_cleaned.jsonl")