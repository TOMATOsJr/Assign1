import json
import re

def add_eos_tokens(text):
    """Add <EOS> token after period or question mark when followed by capital letter."""
    
    # Pattern: period or question mark, followed by space and capital letter
    # Replace with: period/question mark + <EOS> + space + capital letter
    text = re.sub(r'(\. )(?=[A-Z])', r'. <EOS> ', text)
    text = re.sub(r'(\? )(?=[A-Z])', r'? <EOS> ', text)
    
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
    print(f"Data with <EOS> tokens written to {output_file}")

if __name__ == "__main__":
    process_file_with_eos('cc100_en_cleaned.jsonl', 'cc100_en_cleaned2.jsonl')
