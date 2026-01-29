import json

with open("cc100_en.jsonl", "rw") as f:
    for line in f:
        data = json.loads(line)
        text = data["text"]
        # Perform cleaning operations on text
        