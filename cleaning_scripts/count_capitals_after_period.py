import json
import re

def analyze_post_period_characters(input_file):
    """Analyze characters that appear after periods in the cleaned dataset."""

    capital_count = 0
    lowercase_count = 0
    other_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    text = data['text']

                    # Find all characters that come after a period and space
                    # Pattern: period followed by space, then capture the next character
                    matches = re.finditer(r'\?\s(\S)', text)

                    for match in matches:
                        char = match.group(1)

                        if char.isupper():
                            capital_count += 1
                        elif char.islower():
                            lowercase_count += 1
                        else:
                            other_count += 1

                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")

            except json.JSONDecodeError:
                continue

    total = capital_count + lowercase_count + other_count

    print("\n" + "="*60)
    print("ANALYSIS: Characters After Periods in Cleaned English Dataset")
    print("="*60)
    print(f"Capital Letters:  {capital_count:10,} ({100*capital_count/total if total > 0 else 0:.2f}%)")
    print(f"Lowercase Letters: {lowercase_count:10,} ({100*lowercase_count/total if total > 0 else 0:.2f}%)")
    print(f"Other Characters: {other_count:10,} ({100*other_count/total if total > 0 else 0:.2f}%)")
    print("-"*60)
    print(f"Total Occurrences: {total:10,}")
    print("="*60)

if __name__ == "__main__":
    analyze_post_period_characters('cc100_en_cleaned.jsonl')
