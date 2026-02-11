# Report

## Assumptions
1. Assuming that statistically when ever space is used with dash (after normalization of em-dash and en-dash) it most likely as clause break and and no-space as compound word.
2. Learned that , in between numbers and outside need to be handled differently
3. To reduce sparcity we will only be considering lower case letters.
4. Removed other language scripts during cleaning for the dataset. deemed unnecessary.

## Task 1.3
### WhitespaceTokenizer (text.strip().split())
#### English — good

Input: &lt;BOS&gt; &lt;BOS&gt; &lt;BOS&gt; This is fine . &lt;EOS&gt;<br>
Why good: sentence markers and punctuation already separated by cleaning, so whitespace splitting is enough.<br>
Input: Price is 100 , 000 .<br>
Why good: cleaning keeps commas spaced; whitespace split yields consistent tokens.<br>
Input: Use &lt;DASH&gt; for ranges .<br>
Why good: &lt;DASH&gt; survives as its own token.

#### English — fail

Input: e-mail re-enter co-op<br>
Why fail: no-space hyphens are not split, so compound morphology is stuck as one token.<br>
Input: U.S.A.<br>
Why fail: periods are separate tokens after cleaning; whitespace split cannot distinguish abbreviation.<br>
Input: don't<br>
Why fail: stays as one token; if you want don + ' + t, whitespace can't do it.

#### Mongolian — good

Input: &lt;BOS&gt; &lt;BOS&gt; &lt;BOS&gt; Энэ бол тест . &lt;EOS&gt;<br>
Why good: punctuation spaced; sentence tags preserved.<br>
Input: Монгол Улс<br>
Why good: clean Cyrillic words split cleanly.<br>
Input: 2024 оны<br>
Why good: digits + word separated as intended.

#### Mongolian — fail (agglutinative, suffix-heavy)

Input: Сургуулиудаас<br>
Why fail: suffix chain remains a single token; no subword hint.<br>
Input: Нийслэлийнхэнд<br>
Why fail: complex derivation/possession is not segmented.<br>
Input: Эх орончдоороо<br>
Why fail: long morphologically rich token stays intact.

### RegexTokenizer (specials | letters | numbers | single non-space char)
#### English — good

Input: &lt;BOS&gt; Hello , world ! &lt;EOS&gt;<br>
Why good: specials are matched as whole tokens; punctuation split is consistent.<br>
Input: Version 2.0<br>
Why good: yields Version, 2, ., 0, which is useful for some models.<br>
Input: C++<br>
Why good: yields C, +, +, which is explicit.

#### English — fail

Input: don't<br>
Why fail: apostrophe is a separate token, so it becomes don, ', t (may be undesirable).<br>
Input: 3.14<br>
Why fail: splits into 3, ., 14, losing the decimal as a unit.<br>
Input: New_York<br>
Why fail: underscore is not a letter, so it becomes New, _, York.

#### Mongolian — good

Input: &lt;BOS&gt; Монгол хэл сайхан . &lt;EOS&gt;<br>
Why good: Cyrillic letter runs are preserved as whole words.<br>
Input: 2024 оны<br>
Why good: digits and letters separated cleanly.<br>
Input: Тест !<br>
Why good: punctuation becomes its own token.

#### Mongolian — fail (agglutinative)

Input: Сургуульд<br>
Why fail: whole word is one token; no suffix segmentation.<br>
Input: Хүмүүстэйгээ<br>
Why fail: long suffix chain remains intact.<br>
Input: Өгүүлбэрүүдээс<br>
Why fail: suffixes not separated; subword modeling not helped.

### BPETokenizer (trained on whitespace tokens; subword merges)
#### English — good

Input: internationalization<br>
Why good: BPE can learn common subwords like inter, nation, al, reducing OOV.<br>
Input: unbelievable<br>
Why good: likely splits into meaningful subparts (un, believ, able).<br>
Input: tokenization<br>
Why good: frequent morphemes become reusable pieces.

#### English — fail

Input: qzjxfy<br>
Why fail: rare strings fall back to characters, producing noisy tokens.<br>
Input: C++<br>
Why fail: training is on whitespace tokens; + merges are unlikely, so it becomes C, +, + or worse.<br>
Input: U.S.A.<br>
Why fail: periods split during cleaning, but BPE sees tokens separately and does not reconstruct abbreviations.

#### Mongolian — good (suffix-heavy)

Input: Сургуулиудаас<br>
Why good: BPE can learn common suffixes like -аас, -ууд, -ууд-аас.<br>
Input: Хотуудаар<br>
Why good: subword merges can capture хот, ууд, аар.<br>
Input: Нийслэлийн<br>
Why good: stems + suffixes can emerge as stable pieces.

#### Mongolian — fail

Input: Rare long form with stacked suffixes not seen in training (e.g., Сургуулиудаараа)<br>
Why fail: unseen combinations degrade to character-level pieces.<br>
Input: Mixed Latin + Cyrillic in one token (e.g., MNтест)<br>
Why fail: BPE merges are language-specific; mixed scripts fragment.<br>
Input: New proper names with uncommon character sequences<br>
Why fail: BPE tends to over-fragment into chars when no merges exist.

## Kneser-Ney Discount Tuning and Perplexity
```powershell
python .\sweep_kn_discount.py --train .\cc100_en_tokens_bpe_train --val .\cc100_en_tokens_bpe_val --train-format tokenized --val-format tokenized --discounts 0.5,0.6,0.7,0.75,0.8,0.9
Loading train: 100%|███████████████████████████████████████████████████████████████████████| 599999/599999 [00:02<00:00, 205008.86lines/s]
Loading val: 100%|█████████████████████████████████████████████████████████████████████████| 200000/200000 [00:01<00:00, 198342.02lines/s]
Training counts: 100%|█████████████████████████████████████████████████████████████████████████| 599999/599999 [02:15<00:00, 4427.06seq/s]

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
```

Best: D=0.9000 with perplexity 59.534909

## All models perplexities

| Tokenizer | Smoothing | Test Perplexity |
|-----------|-----------|-----------|
| Whitespace | Kneser-Ney | 130.613497 |
| Whitespace | Witten-Bell | 185.397543 |
| Whitespace | MLE | inf |
| Regex | Kneser-Ney | 106.459900 |
| Regex | Witten-Bell | 147.533849 |
| Regex | MLE | inf |
| BPE | Kneser-Ney | 59.430502 |
| BPE | Witten-Bell | 76.923112 |
| BPE | MLE | inf |


## Autocomplete prompts
1. "i want to"
2. "the government should"
3. "once"
4. "he"
5. "there is"
6. "machine learning is"
7. "this is the first"
8. "in the middle of"
<br>
## Results and Evaluation
Whitespace + Kneser_ney (perplexity = 130.613497)
- i want to
i want to be a part of the world .
- the government should
the government should be minimised .
- once
once you have a great time .
- he
he was a great time .
- there is
there is a great way to get a good deal of time , and the
- machine learning is
machine learning is a great way to get a good deal of time , and the
- this is the first
this is the first time in the past , and i will be visiting again .
- in the middle of
in the middle of the night , and the

Summary: Fluent on common phrases ("i want to be a part of the world"), but backs off to generic templates like "a great way to get a good deal of time" and truncates on "and the".

Whitespace + Witten_bell (perplexity = 185.397543)
- i want to
i want to be a part of the world .
- the government should
the government should not be used as a tool to help you get the idea .
- once
once you have a great time .
- he
he was a man of the match .
- there is
there is a lot of people are hunting round for this information , you can use the following procedure to delete any comments that are submitted without names or other identifiers <DASH> to collect data such as name , address , phone number , and the
- machine learning is
machine learning is the hot sauce of of choice for the color of the dress and the red sea .
- this is the first
this is the first time in a long time .
- in the middle of
in the middle of the night , and the

Summary: Produces fluent short completions ("he was a man of the match"), but can run on with boilerplate and surface artifacts like "<DASH>" and "of of".

Whitespace + MLE (p = inf)
- i want to
i want to be a part of the world .
- the government should
the government should not be used as a tool to help you get the idea .
- once
once you have a great time .
- he
he was a man of the match .
- there is
there is a lot of people are hunting round for this information , you can use the following procedure to view the full press release here .
- machine learning is
machine learning is slowly but surely .
- this is the first
this is the first time in a long time .
- in the middle of
in the middle of the night , and the

Summary: Replays frequent spans ("this is the first time in a long time"), but is brittle on sparse contexts, drifting into boilerplate and stopping mid-clause.

Regex + Witten bell (p = 147.533849)
- i want to
i want to be a part of the world ' s largest online destinations for the home .
- the government should
the government should not regulate the . . .
- once
once you ' ve got a lot of people are hunting round for this information , you can ' t wait to see what you can do to help you get the idea .
- he
he was a man of the match .
- there is
there is a lot of people are hunting round for this information , you can ' t wait to see what you can do to help you get the idea .
- machine learning is
machine learning is the best way to get a good deal of time and effort .
- this is the first
this is the first time in a long time .
- in the middle of
in the middle of the night , and the

Summary: Keeps punctuation-separated fluency, but shows regex splits like "world ' s" and punctuation loops such as ". . .".

Regex + kneser ney (p = 106.459900)
- i want to
i want to be a part of the world ' s largest and most of the time .
- the government should
the government should be minimised .
- once
once you ' ve got a lot of people who are not afraid to mention how they believe .
- he
he was a great time .
- there is
there is a lot of people who are not afraid to mention how they believe .
- machine learning is
machine learning is a great way to get a good deal of time , and i ' m not sure if this is a great way to get a good deal of time , and i ' m not sure if this is a great way to get a good deal of time
- this is the first
this is the first time in the past , and i ' m not sure if this is a great way to get a good deal of time , and i ' m not sure if this is a great way to get a good deal of time , and i ' m not
- in the middle of
in the middle of the night , and the

Summary: Good short completions ("the government should be minimised"), but falls into repetition loops like the repeated "great way to get a good deal of time" chain.

Regex + MLE (p = inf)
- i want to
i want to be a part of the world ' s largest online destinations for the home .
- the government should
the government should not guarantee the accuracy of the information you need to know about the new system .
- once
once you ' ve got a lot of people are hunting round for this information , you can ' t wait to see what you can do to help you get the idea .
- he
he was a man of his word , and the
- there is
there is a lot of people are hunting round for this information , you can ' t wait to see what you can do to help you get the idea .
- machine learning is
machine learning is slowly but surely .
- this is the first
this is the first time in a long time .
- in the middle of
in the middle of the night , and i ' m not sure if this is the first time in a long time .

Summary: Generates long memorized passages, but overfits to single templates and truncates when exact 4-grams end.

BPE + Kneser ney (p = 59.430502)
- i want to
i want to be a part of the world .
- the government should
the government should be mini mi zed .
- once
once you have a great time .
- he
he was a great time .
- there is
there is a lot of people who are not af raid to ask questions and answ ers .
- machine learning is
machine learning is a great way to get a good deal of time , and the
- this is the first
this is the first time in the future .
- in the middle of
in the middle of the night , and i will be visiting again .

Summary: Fluent at the sentence level, but exposes subword splits like "mini mi zed" and "af raid".

BPE + Witten Bell (p = 76.923112)
- i want to
i want to be a part of the world .
- the government should
the government should not guaran tee the accur acy of the information you need to know about the new york times , the same as the one who is in the process of making a living , brea thing , and then the next day .
- once
once you have to do is to find a way to get a good deal of time and effort .
- he
he was a man of the board of direc tors .
- there is
there is a lot of people are hun ting around for this information , you can use the following proc ed ure .
- machine learning is
machine learning is the facili tation of the bi ble .
- this is the first
this is the first time in a long time .
- in the middle of
in the middle of the night , and i will be visiting again .

Summary: Stable short completions, but long outputs show heavy subword fragmentation ("guaran tee", "accur acy") and run-on templates.

BPE + MLE (p = inf)
- i want to
i want to be a part of the world .
- the government should
the government should not make the mista ke of thinking that the arti st 's stu dio , and now i have to say that this writ e- up .
- once
once you have a great time .
- her
her eyes were clo sed .
- he
he was a man of the board of direc tors .
- there is
there is a lot of people are hun ting around for this information , you can use the following proc ed ure .
- machine learning is
machine learning is the hot mom from " a mom ent 's notice .
- this is the first
this is the first time in a long time .
- in the middle of
in the middle of the night , and i will be visiting again .

Summary: Reproduces frequent sequences, but shows the most subword artifacts ("mista ke", "mom ent") and brittle template sticking.

