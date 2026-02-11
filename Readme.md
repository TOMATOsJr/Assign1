Report
Assumptions
1. assuming that statistically when ever space is used with dash (after normalization of em-dash and en-dash) it most likely as clause break and and no-space as compound word.
2. Learned that , in between numbers and outside need to be handled differently

Tokenizer pipeline usage
1. Whitespace tokenizer on JSONL
```powershell
python tokenizer_pipeline.py --tokenizer whitespace --input cc100_en_final_train.jsonl --output cc100_en_tokens_whitespace --jsonl-field text
```

2. Regex tokenizer on JSONL (when implemented)
```powershell
python tokenizer_pipeline.py --tokenizer regex --input cc100_en_final_train.jsonl --output cc100_en_tokens_regex --jsonl-field text
```

3. BPE training + tokenizing train + val
```powershell
python tokenizer_pipeline.py --tokenizer bpe --train-input cc100_en_final_train.jsonl --input cc100_en_final_train.jsonl --output cc100_en_tokens_bpe_train --val-input cc100_en_final_val.jsonl --val-output cc100_en_tokens_bpe_val --jsonl-field text --vocab-size 20000
```

4. Whitespace tokenizer on plain text
```powershell
python tokenizer_pipeline.py --tokenizer whitespace --input input.txt --output tokens_whitespace.txt
```

Language model usage
1. Train + evaluate perplexity (Kneser-Ney)
```powershell
python run_language_model.py --train cc100_en_tokens_bpe_train --train-format tokenized --val cc100_en_tokens_bpe_val --val-format tokenized --smoothing kneser_ney
```

2. Train + generate from prompt (MLE)
```powershell
python run_language_model.py --train cc100_en_tokens_whitespace --train-format tokenized --smoothing mle --prompt "the quick brown" --max-tokens 20
```

Discount sweep for Kneser-Ney
```powershell
python sweep_kn_discount.py --train cc100_en_tokens_bpe_train --val cc100_en_tokens_bpe_val --train-format tokenized --val-format tokenized --discounts 0.5,0.6,0.7,0.75,0.8,0.9
```

Smoothing result on bpe
1. python run_language_model.py --train cc100_en_tokens_bpe_train --train-format tokenized --val cc100_en_tokens_bpe_val --val-format tokenized --smoothing witten_bell
Loading train: 100%|███████████████████████████████████████████████████████████████████████| 599999/599999 [00:02<00:00, 201944.29lines/s]
Training: 100%|████████████████████████████████████████████████████████████████████████████████| 599999/599999 [02:13<00:00, 4505.27seq/s]
Loading val: 100%|█████████████████████████████████████████████████████████████████████████| 200000/200000 [00:00<00:00, 275737.79lines/s]
Perplexity: 100%|██████████████████████████████████████████████████████████████████████████████| 200000/200000 [01:09<00:00, 2860.83seq/s]
Perplexity: 77.074871

2. python .\sweep_kn_discount.py --train .\cc100_en_tokens_bpe_train --val .\cc100_en_tokens_bpe_val --train-format tokenized --val-format tokenized --discounts 0.5,0.6,0.7,0.75,0.8,0.9
Loading train: 100%|███████████████████████████████████████████████████████████████████████| 599999/599999 [00:02<00:00, 205008.86lines/s]
Loading val: 100%|█████████████████████████████████████████████████████████████████████████| 200000/200000 [00:01<00:00, 198342.02lines/s]
Training counts: 100%|█████████████████████████████████████████████████████████████████████████| 599999/599999 [02:15<00:00, 4427.06seq/s]
OOV scan: 100%|██████████████████████████████████████████████████████████████████████████████| 200000/200000 [00:00<00:00, 252758.98seq/s]

OOV tokens in val: 0/9435808 (0.00%)
Perplexity D=0.5: 100%|████████████████████████████████████████████████████████████████████████| 200000/200000 [01:22<00:00, 2426.49seq/s]
Perplexity D=0.6: 100%|████████████████████████████████████████████████████████████████████████| 200000/200000 [01:30<00:00, 2204.70seq/s]
Perplexity D=0.7: 100%|████████████████████████████████████████████████████████████████████████| 200000/200000 [01:29<00:00, 2228.33seq/s]
Perplexity D=0.75: 100%|███████████████████████████████████████████████████████████████████████| 200000/200000 [02:05<00:00, 1596.58seq/s]
Perplexity D=0.8: 100%|████████████████████████████████████████████████████████████████████████| 200000/200000 [01:22<00:00, 2410.74seq/s]
Perplexity D=0.9: 100%|████████████████████████████████████████████████████████████████████████| 200000/200000 [01:28<00:00, 2268.65seq/s]

Discount        Perplexity
0.5000  73.398002
0.6000  67.679383
0.7000  63.625362
0.7500  62.098541
0.8000  60.884527
0.9000  59.534909

Best: D=0.9000 with perplexity 59.534909