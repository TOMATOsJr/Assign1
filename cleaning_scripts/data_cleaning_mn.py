import json
import re
import unicodedata

def clean_text(text):
    """ Cleans the input text by removing unwanted characters and normalizing whitespace. """

    # Normalize select non-ASCII punctuation to ASCII equivalents
    text = text.translate(str.maketrans({
        "€": "$",
        "£": "$",
        "¥": "$",
        "¢": "$",
        "¤": "$",
        "₹": "$",
        "₤": "$",
        "₦": "$",
        "₰": "$",
        "₱": "$",
        "₵": "$",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "《": '"',
        "》": '"',
        "【": "[",
        "】": "]",
        "。": ".",
        "〜": "~",
    }))

    # Remove soft hyphens (U+00AD) and other invisible formatting marks
    text = text.replace('\u00ad', '')  # Soft hyphen
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\ufeff', '')  # Zero-width no-break space (BOM)

    # Remove control characters (except newline, tab, carriage return)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r ')

    # Normalize unicode characters to NFKC form
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

    # Fold Latin diacritics and map special letters to ASCII
    # NOTE: For Mongolian dataset, we KEEP diacritics and don't normalize them
    # special_map = {
    #     "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ø": "o", "Ø": "O",
    #     "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "", "Þ": "", "ł": "l", "Ł": "L",
    # }
    # text = "".join(special_map.get(ch, ch) for ch in text)
    # text = unicodedata.normalize("NFKD", text)
    # text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # For Mongolian: Keep ONLY Cyrillic, Latin, digits, combining marks for Cyrillic, ASCII punctuation
    # Remove: All other scripts (CJK, Korean, Arabic, Thai, Hindi, etc.), emojis, unwanted symbols
    def is_valid_mongolian_char(ch):
        code_point = ord(ch)
        cat = unicodedata.category(ch)

        # Keep letters only if they are Cyrillic or Latin
        if cat[0] == 'L':  # Letter category
            # Keep Cyrillic (U+0400–U+04FF) - includes Mongolian-specific chars Ө, ү, etc.
            if 0x0400 <= code_point <= 0x04FF:
                return True
            # Keep Latin (A-Z, a-z, U+0041–U+005A, U+0061–U+007A)
            if (0x0041 <= code_point <= 0x005A) or (0x0061 <= code_point <= 0x007A):
                return True
            # Remove ALL other scripts (CJK, Korean, Arabic, Thai, Hindi, Devanagari, etc.)
            return False

        # Keep ASCII digits (0-9, U+0030–U+0039)
        if cat[0] == 'N':
            if 0x0030 <= code_point <= 0x0039:
                return True
            # Remove other digit scripts (Arabic numerals, etc.)
            return False

        # Keep ONLY combining marks that are valid for Cyrillic/Latin
        # Allow: Combining Diacritical Marks (U+0300–U+036F), Cyrillic combining (U+0483–U+0487)
        # Exclude: All other script combining marks (Devanagari, Thai, Khmer, etc.)
        if cat[0] == 'M':
            if (0x0300 <= code_point <= 0x036F) or \
               (0x0483 <= code_point <= 0x0487):
                return True
            # Remove marks from other scripts (Hindi, Thai, Khmer, etc.)
            return False

        # Remove emojis, symbols, and variation selectors
        if cat in ('So', 'Cs'):
            return False
        if 0xFE00 <= code_point <= 0xFE0F:
            return False

        # Keep ASCII punctuation and spaces (U+0000–U+007F)
        # This includes: ! " # $ % & ' ( ) * , - . / : ; < > ? @ [ \ ] ^ _ ` { | } etc.
        if code_point <= 0x007F:
            return True

        # Remove everything else (punctuation/symbols from other scripts)
        return False

    text = "".join(ch for ch in text if is_valid_mongolian_char(ch))

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
    # Test samples - including Mongolian examples
    test_sample = [
        # English examples
        "Hello World! This is a test ... haha\n\n\nNew line here.\t\tExtra tabs.",
        "Another example with invisible characters.\n\n\n\nEnd of test.",
        "em-dash -- should be cleaned properly...   with spaces.",
        "Multiple!!! exclamation marks!!!!",
        "Test - with - dashes and-compound-words",
        "Ellipsis. . .test 'and' more..dots",
        "Double dash--test and -- more -- tests",
        "Two dots..here and...three dots....many dots",
        "Spaced dots . . . and spaced ! ! ! and backticks `like this` and ``quote``",
        "Curly quotes: (\"hello\" and 'world' should be straight)",
        "a,b,c and 100,000 should not have spaces around commas, but a,b,c should become a , b , c",
        # Mongolian examples
        "Бүх төрлийн барилгын заслын ажил чанартай хийнэ - ZARMEDEE.MN",
        "Яг юуг олж харахгүй байна вэ? _ www.baabar.mn _ Шилдэг нийтлэлчдийн клуб",
        "Мөнх Монгол! 123 с Өнгөрөл үүрэг ба ангилал хариу...",
        "Test with emoji 😀 and CJK should be removed from mixed text",
        "Mixed: Эр Latin123 тест",
        " 는니닐님다단담당대"
    ]

    print("Testing cleaning function:")
    print("=" * 60)
    for i, sample in enumerate(test_sample, 1):
        print(f"\nTest {i}:")
        print(f"Original: {sample}")
        print(f"Cleaned:  {clean_text(sample)}")

    print("\n" + "=" * 60)
    print("\nProcessing Mongolian dataset...")
    process_file("cc100_mn.jsonl", "cc100_mn_cleaned.jsonl")