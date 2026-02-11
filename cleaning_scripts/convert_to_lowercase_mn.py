import json
import re

def lowercase_except_tokens(text):
    """Convert text to lowercase while preserving <EOS>, <BOS>, and <DASH> tokens."""

    # Replace special tokens with placeholders to protect them
    text = text.replace('<EOS>', '<PLACEHOLDER_EOS>')
    text = text.replace('<BOS>', '<PLACEHOLDER_BOS>')
    text = text.replace('<DASH>', '<PLACEHOLDER_DASH>')

    # Unicode-aware lowercasing for Cyrillic and Latin
    text = text.casefold()

    # Restore the special tokens in their original form
    text = text.replace('<placeholder_eos>', '<EOS>')
    text = text.replace('<placeholder_bos>', '<BOS>')
    text = text.replace('<placeholder_dash>', '<DASH>')

    return text

def process_file_to_lowercase(input_file, output_file):
    """Process cleaned JSONL file and convert to lowercase, write to new file."""
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    data['text'] = lowercase_except_tokens(data['text'])
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 10000 == 0:
                        print(f"Processed {count} lines...")
            except json.JSONDecodeError:
                continue

    print(f"\nTotal lines processed: {count}")
    print(f"Lowercase cleaned dataset written to {output_file}")

if __name__ == "__main__":
    process_file_to_lowercase('cc100_mn_cleaned2.jsonl', 'cc100_mn_cleaned_final.jsonl')
