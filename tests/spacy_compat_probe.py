"""Compare Nesta's published NER weights across spaCy versions.

The evaluation corpus is Green et al.'s CC BY 4.0 skill extraction test set:
https://github.com/acp19tag/skill-extraction-dataset
Run this script in separate Python environments with shared model files.
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import spacy
from huggingface_hub import snapshot_download

from ojd_daps_skills.utils.text_cleaning import clean_text


def prepare_data(path):
    sentences = OrderedDict()
    with open(path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sentences.setdefault(row["sentence_id"], []).append(row)

    texts, gold = [], []
    for rows in sentences.values():
        words, tags = [], []
        entity_id = 0
        previous_type = last_emitted_entity = None
        for row in rows:
            tag = row["tag"]
            if tag == "O":
                current_entity = None
                previous_type = None
            else:
                prefix, label = tag.split("-", 1)
                if prefix == "B" or label != previous_type:
                    entity_id += 1
                current_entity = entity_id
                previous_type = label
            for part in clean_text(row["word"]).split():
                if current_entity is None:
                    part_tag = "O"
                else:
                    prefix = "I" if current_entity == last_emitted_entity else "B"
                    part_tag = f"{prefix}-{label}"
                words.append(part)
                tags.append(part_tag)
                last_emitted_entity = current_entity

        text = " ".join(words)
        assert text and clean_text(text) == text
        positions = []
        position = 0
        active = None
        for word, tag in zip(words, tags):
            end = position + len(word)
            if tag == "O":
                active = None
            else:
                prefix, label = tag.split("-", 1)
                if prefix == "B" or active is None or active[2] != label:
                    active = [position, end, label]
                    positions.append(active)
                else:
                    active[1] = end
            position = end + 1
        texts.append(text)
        gold.append([(start, end, label.upper()) for start, end, label in positions])
    return texts, gold


def predict(nlp, texts):
    return [
        {
            "tokens": len(doc),
            "entities": [
                [ent.start_char, ent.end_char, ent.label_] for ent in doc.ents
            ],
        }
        for doc in nlp.pipe(texts, batch_size=16)
    ]


def score(gold, predictions, label):
    correct = predicted = actual = 0
    for gold_spans, result in zip(gold, predictions):
        expected = {tuple(span) for span in gold_spans if span[2] == label}
        observed = {tuple(span) for span in result["entities"] if span[2] == label}
        correct += len(expected & observed)
        actual += len(expected)
        predicted += len(observed)
    precision = correct / predicted if predicted else 0.0
    recall = correct / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct": correct,
        "predicted": predicted,
        "gold": actual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--resave", type=Path)
    args = parser.parse_args()

    texts, gold = prepare_data(args.data)
    assert len(texts) == 336, "The evaluation set changed."
    model_path = snapshot_download(
        repo_id="nestauk/en_skillner",
        revision="2610a6de4a5764a167c2eceff7cfd0a3e89e7c40",
        cache_dir=args.cache,
        allow_patterns=[
            "config.cfg",
            "meta.json",
            "tokenizer",
            "attribute_ruler/*",
            "lemmatizer/*",
            "ner/*",
            "parser/*",
            "senter/*",
            "tagger/*",
            "tok2vec/*",
            "vocab/*",
        ],
    )
    nlp = spacy.load(model_path)
    results = predict(nlp, texts)
    report = {
        "spacy": spacy.__version__,
        "model_version": nlp.meta.get("version"),
        "texts": texts,
        "predictions": results,
        "skill": score(gold, results, "SKILL"),
        "experience": score(gold, results, "EXPERIENCE"),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("spacy", "model_version", "skill", "experience")
            },
            indent=2,
        ),
        flush=True,
    )

    if args.resave:
        nlp.to_disk(args.resave)
        reloaded = spacy.load(args.resave)
        assert predict(reloaded, texts) == results, (
            "Reloaded 3.8 model changes predictions."
        )
        print(
            "Reserialized model: all 336 predictions identical after reload.",
            flush=True,
        )

    if args.baseline:
        original = json.loads(args.baseline.read_text(encoding="utf-8"))
        assert original["texts"] == texts, "The two runs used different text."
        differences = [
            i
            for i, (a, b) in enumerate(zip(original["predictions"], results))
            if a["entities"] != b["entities"]
        ]
        token_differences = sum(
            a["tokens"] != b["tokens"] for a, b in zip(original["predictions"], results)
        )
        print(
            f"Entity predictions changed in {len(differences)}/336 sentences; "
            f"token counts changed in {token_differences}/336.",
            flush=True,
        )
        for index in differences[:5]:
            print(
                f"Sentence {index + 1}: {texts[index]!r}\n"
                f"  3.7: {original['predictions'][index]['entities']}\n"
                f"  3.8: {results[index]['entities']}",
                flush=True,
            )
        assert not differences, "The spaCy 3.8 model changed entity predictions."
        for label in ("skill", "experience"):
            delta = report[label]["f1"] - original[label]["f1"]
            print(f"{label} F1 change: {delta:+.5f}", flush=True)
            assert delta >= -0.01, f"{label} F1 dropped by more than one point."


if __name__ == "__main__":
    main()
