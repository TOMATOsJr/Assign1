import json
import re

def analyze_dash_patterns(file_path):
    """
    Analyze different dash usage patterns in the dataset.
    Counts dashes with:
    - Spaces on both sides
    - Space on left only
    - Space on right only
    - No spaces at all
    """

    # Different types of dashes
    dashes = {
        'dash': '-',
        'en_dash': '–',
        'em_dash': '—'
    }

    # Initialize counters
    counts = {
        'both_sides': {dash_type: 0 for dash_type in dashes},
        'left_only': {dash_type: 0 for dash_type in dashes},
        'right_only': {dash_type: 0 for dash_type in dashes},
        'no_spaces': {dash_type: 0 for dash_type in dashes}
    }

    # Read and process the file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')

                # Check each type of dash
                for dash_type, dash_char in dashes.items():
                    # Both sides: space + dash + space
                    counts['both_sides'][dash_type] += len(re.findall(rf' {re.escape(dash_char)} ', text))

                    # Left only: space + dash + non-space (but not another space)
                    counts['left_only'][dash_type] += len(re.findall(rf' {re.escape(dash_char)}(?! )', text))

                    # Right only: non-space + dash + space (but not preceded by space)
                    counts['right_only'][dash_type] += len(re.findall(rf'(?<! ){re.escape(dash_char)} ', text))

                    # No spaces: non-space + dash + non-space
                    counts['no_spaces'][dash_type] += len(re.findall(rf'(?<! ){re.escape(dash_char)}(?! )', text))

            except json.JSONDecodeError:
                continue

    return counts

def print_results(counts):
    """Print the analysis results in a formatted way."""
    print("Dash Usage Pattern Analysis (English Dataset)")
    print("=" * 60)

    categories = [
        ('both_sides', 'Spaces on both sides'),
        ('left_only', 'Space on left only'),
        ('right_only', 'Space on right only'),
        ('no_spaces', 'No spaces')
    ]

    dash_labels = {
        'dash': 'Regular dash (-)',
        'en_dash': 'En dash (–)',
        'em_dash': 'Em dash (—)'
    }

    for category, category_name in categories:
        print(f"\n{category_name}:")
        print("-" * 60)
        for dash_type in ['dash', 'en_dash', 'em_dash']:
            print(f"  {dash_labels[dash_type]:<20}: {counts[category][dash_type]:,}")

if __name__ == "__main__":
    file_path = "cc100_en.jsonl"

    print("Analyzing dash patterns in English dataset...")
    print("This may take a moment...\n")

    counts = analyze_dash_patterns(file_path)
    print_results(counts)
