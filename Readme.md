## Scripts and Commands

### Top-level scripts

**tokenizer_pipeline.py**
Purpose: Script that runs the tokenizer on cleaned data and also the one that trains the tokenizer when it is bpe. (tokenizes train and val on argument passing)
```powershell
python tokenizer_pipeline.py --tokenizer bpe --train-input cc100_en_train.jsonl --input cc100_en_train.jsonl --output cc100_en_tokens_bpe_train --val-input cc100_en_final_val.jsonl --val-output cc100_en_tokens_bpe_val --jsonl-field text --vocab-size 20000
```

(ignore this script)
**run_language_model.py**
Purpose: Train/evaluate an n-gram LM, generate text, or run autocomplete (but use interactive_autocomplete.py for autocomplete). Was primarily used for training and calculating perplexity scores.
```powershell
python run_language_model.py --train cc100_en_tokens_bpe_train --train-format tokenized --val cc100_en_tokens_bpe_val --val-format tokenized --smoothing kneser_ney
python run_language_model.py --train cc100_en_tokens_whitespace --train-format tokenized --smoothing mle --prompt "the quick brown" --max-tokens 20
```

**interactive_autocomplete.py**
Purpose: Train tokenizer + LM and run interactive generation or top-k suggestions. (main this one for any autocompletion task). Generates auto complete till end of sentence is generated.
```powershell
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer whitespace --smoothing mle --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer whitespace --smoothing kneser_ney --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer whitespace --smoothing witten_bell --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer regex --smoothing mle --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer regex --smoothing kneser_ney --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer regex --smoothing witten_bell --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer bpe --bpe-vocab-size 20000 --smoothing mle --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer bpe --bpe-vocab-size 20000 --smoothing kneser_ney --n 4
python interactive_autocomplete.py --train cc100_en_train.jsonl --train-format jsonl --jsonl-field text --tokenizer bpe --bpe-vocab-size 20000 --smoothing witten_bell --n 4
```

**sweep_kn_discount.py**
Purpose: Sweep Kneser-Ney discount values and report validation perplexity.
```powershell
python sweep_kn_discount.py --train cc100_en_tokens_bpe_train --val cc100_en_tokens_bpe_val --train-format tokenized --val-format tokenized --discounts 0.5,0.6,0.7,0.75,0.8,0.9
```

**run_all_configs.py**
Purpose: Train/evaluate multiple tokenizer x smoothing configs on tokenized files.
```powershell
python run_all_configs.py --base cc100_en_tokens --n 4
```

**count_charset.py**
Purpose: Count unique characters in a text or JSONL file.
```powershell
python count_charset.py --input cc100_en_cleaned_final.jsonl --jsonl-field text
python count_charset.py --input cc100_en_tokens_whitespace
```

**language_models.py**
Purpose: Core n-gram LM implementations and smoothing strategies. No CLI.

**tokenizers.py**
Purpose: Tokenizer implementations (whitespace, regex, BPE). No CLI.

### cleaning_scripts/ (also refer the readme for cleaning in cleaning dir)

**cleaning_scripts/data_cleaning.py**
Purpose: Clean English JSONL text with normalization, punctuation handling, and <DASH> tagging.
```powershell
python cleaning_scripts/data_cleaning.py
```

**cleaning_scripts/data_cleaning_mn.py**
Purpose: Clean Mongolian JSONL text with Cyrillic/Latin filtering and normalization.
```powershell
python cleaning_scripts/data_cleaning_mn.py
```

**cleaning_scripts/add_bos_eos_tokens.py**
Purpose: Add <BOS>/<EOS> tags to English cleaned JSONL.
```powershell
python cleaning_scripts/add_bos_eos_tokens.py
```

**cleaning_scripts/add_bos_eos_tokens_mn.py**
Purpose: Add <BOS>/<EOS> tags to Mongolian cleaned JSONL (Latin/Cyrillic caps).
```powershell
python cleaning_scripts/add_bos_eos_tokens_mn.py
```

**cleaning_scripts/convert_to_lowercase.py**
Purpose: Lowercase English cleaned JSONL while preserving special tokens.
```powershell
python cleaning_scripts/convert_to_lowercase.py
```

**cleaning_scripts/convert_to_lowercase_mn.py**
Purpose: Lowercase Mongolian cleaned JSONL while preserving special tokens.
```powershell
python cleaning_scripts/convert_to_lowercase_mn.py
```

**cleaning_scripts/train_test_val_split.py**
Purpose: Split JSONL into train/val/test (default 60/20/20).
```powershell
python cleaning_scripts/train_test_val_split.py --input cc100_mn_cleaned_final.jsonl --train cc100_mn_train.jsonl --val cc100_mn_val.jsonl --test cc100_mn_test.jsonl --seed 42
```

**cleaning_scripts/dash_analysis.py**
Purpose: Analyze dash usage patterns in JSONL.
```powershell
python cleaning_scripts/dash_analysis.py
```

**cleaning_scripts/count_capitals_after_period.py**
Purpose: Count characters after sentence-ending periods in cleaned English JSONL.
```powershell
python cleaning_scripts/count_capitals_after_period.py
```