import json
import re

def add_eos_tokens(text):
    """Add <EOS> after period before caps and line-end, add <BOS> tags accordingly."""

    # Add <BOS> tags at the start, regardless of leading whitespace
    text = re.sub(r'^(\s*)', r'\1<BOS> <BOS> <BOS> ', text)

    # .) (allow optional spaces between . and )) followed by a capital letter -> add <EOS> and three <BOS>
    text = re.sub(r'(?<!\.)\.(?!\.)\s*\)(\s*)(?=[A-Z])', r'.) <EOS> <BOS> <BOS> <BOS> ', text)

    # .) (allow optional spaces between . and )) at end of line (ignoring trailing whitespace) -> add <EOS>
    text = re.sub(r'(?<!\.)\.(?!\.)\s*\)(\s*$)', r'.) <EOS>\1', text)

    # Period (not ellipsis) followed by a capital letter -> add <EOS> and three <BOS>
    text = re.sub(r'(?<!\.)\.(?!\.)\s*(?=[A-Z])', r'. <EOS> <BOS> <BOS> <BOS> ', text)

    # Period (not ellipsis) at end of line (ignoring trailing whitespace) -> add <EOS>
    text = re.sub(r'(?<!\.)\.(?!\.)\s*$', r'. <EOS>', text)

    return text

def process_file_with_eos(input_file, output_file):
    """Process cleaned JSONL file and add <EOS> tokens, write to new file."""
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    data['text'] = add_eos_tokens(data['text'])
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 10000 == 0:
                        print(f"Processed {count} lines...")
            except json.JSONDecodeError:
                continue

    print(f"\nTotal lines processed: {count}")
    print(f"Data with <EOS> and <BOS> tokens written to {output_file}")

if __name__ == "__main__":
    process_file_with_eos('cc100_en_cleaned.jsonl', 'cc100_en_cleaned2.jsonl')
