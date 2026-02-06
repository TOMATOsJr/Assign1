# English Dataset Cleaning Pipeline

## Overview
This directory contains scripts for cleaning the English CC100 dataset. The pipeline involves multiple stages of text normalization, token addition, and case conversion.

## Cleaning Scripts (In Order)

### 1. `data_cleaning.py`
**Role:** Initial text normalization and cleanup
- Removes invisible formatting marks (soft hyphens, zero-width spaces, BOM)
- Removes control characters (except newline, tab, carriage return)
- Normalizes Unicode to NFKC form
- Removes emojis and rare Unicode characters
- Normalizes dashes (em-dash, en-dash → hyphen)
- Normalizes quotes (curly/smart quotes → straight quotes; backticks → straight quotes)
- Converts double dashes (`--`) to spaced single dash (` - `)
- Converts spaced dashes (` - `) to `<DASH>` to differentiate clause break from compound words
- Normalizes ellipsis (`..`, `. . .`, and `....` → `...`) and keeps it spaced as a whole token
- Collapses spaced exclamation sequences (`! ! !` → `!`)
- Spaces out punctuation marks while preserving commas in numbers (e.g., `1,000`)
- Spaces out parentheses and brackets (`()`, `[]`)
- Spaces out quotes when they are not apostrophes inside words
- Cleans excessive whitespace and newlines

**Input:** `cc100_en.jsonl`
**Output:** `cc100_en_cleaned.jsonl` *(First Iteration)*

---

### 2. `add_bos_eos_tokens.py`
**Role:** Add Begin/End-of-Sentence markers
- Adds three `<BOS>` tokens at the start of each line (after any leading whitespace)
- Adds `<EOS>` tokens after periods (`.`) when followed by capital letters, and inserts `<BOS>` tokens after those sentence breaks
- Adds `<EOS>` tokens when a period is the last non-whitespace character in a line
- Applies the same logic for `.)` (allowing optional spaces between `.` and `)`)
- Skips ellipses by only matching single periods

**Input:** `cc100_en_cleaned.jsonl`
**Output:** `cc100_en_cleaned2.jsonl` *(Second Iteration)*

---

### 3. `convert_to_lowercase.py`
**Role:** Final normalization to lowercase
- Converts all text to lowercase for case-insensitive processing
- **Preserves** special tokens: `<EOS>`, `<BOS>`, and `<DASH>` remain uppercase
- Maintains token integrity while normalizing character case

**Input:** `cc100_en_cleaned2.jsonl`
**Output:** `cc100_en_cleaned_final.jsonl` *(Final Clean Dataset)*

---

## Dataset Iterations

### Iteration 1: `cc100_en_cleaned.jsonl`
After running `data_cleaning.py`
- Text is normalized and cleaned
- Punctuation is properly spaced
- Invisible characters and emojis removed
- Maintains original capitalization

### Iteration 2: `cc100_en_cleaned2.jsonl`
After running `add_bos_eos_tokens.py`
- Contains all cleaning from Iteration 1
- Includes `<EOS>` and `<BOS>` tokens marking sentence boundaries
- Maintains original capitalization
- Ready for language model preprocessing

### Final: `cc100_en_cleaned_final.jsonl`
After running `convert_to_lowercase.py`
- Contains all previous cleaning steps
- Fully lowercased for case-insensitive processing
- `<EOS>` and `<DASH>` tokens preserved
- Ready for training/evaluation

---

## Utility Scripts

### `count_capitals_after_period.py`
Analyzes the distribution of characters following periods:
- Counts capital letters after periods
- Counts lowercase letters after periods
- Counts other characters after periods
- Provides percentages for each category

**Usage:** Helps validate the effectiveness of `<EOS>` token placement

### `dash_analysis.py`
Analyzes dash patterns in the dataset specifically for clause breaks or compound words.

---

## Quick Start

Run the cleaning pipeline in order:

```bash
# Step 1: Initial cleaning
python data_cleaning.py

# Step 2: Add BOS/EOS tokens
python add_bos_eos_tokens.py

# Step 3: Convert to lowercase
python convert_to_lowercase.py

# Optional: Analyze capital letters after periods
python count_capitals_after_period.py

# Optional2: Analyzing dashes with and without spaces
python dash_analysis.py
```

The final cleaned dataset will be saved in `cc100_en_cleaned_final.jsonl`

---

## File Format
All datasets are in JSONL format (JSON Lines):
```json
{"text": "the content of each document..."}
{"text": "another document with <EOS> tokens and <DASH> special tokens..."}
```
