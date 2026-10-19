# Skills Extractor — maintained fork

This is [kalleknast's fork](https://github.com/kalleknast/ojd_daps_skills) of
[Nesta's Skills Extractor](https://github.com/nestauk/ojd_daps_skills). It extracts
skill phrases from job advertisements and maps them to the **toy**, **ESCO**, or
**Lightcast** taxonomy. Nesta's MIT license and attribution are retained.

## Requirements

Use **Python 3.10, 3.11, or 3.12**, preferably in a fresh virtual environment.
The published Nesta NER model was trained with spaCy 3.7 and requires NumPy 1.x.
Python 3.13/3.14 and NumPy 2 are not supported by this release. Removing these
bounds requires validating or migrating the model, not simply changing pins.

On Windows, Torch is constrained below 2.9 to avoid DLL initialization failures
seen with newer wheels in this stack. Linux and macOS allow newer Torch 2 releases.

CPU inference is supported; a GPU is not required. The first extraction downloads
several hundred MB of model data and needs internet access. Later runs reuse a
user cache. No Hugging Face token is required for the public Nesta models.

## Install this fork

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/kalleknast/ojd_daps_skills.git@dev"
```

The import name remains `ojd_daps_skills`. A plain `pip install ojd-daps-skills`
installs the upstream PyPI package, not this fork. For a reproducible course,
replace `dev` in the Git URL with the tested commit SHA.

### Google Colab

Select **Runtime → Change runtime type → Python 3 → CPU → 2026.07**
(Python 3.12.13), or another available Python 3.10–3.12 runtime.
Then run:

```python
%pip install "git+https://github.com/kalleknast/ojd_daps_skills.git@dev" nervaluate==1.2.1
```

**Restart the session after installation**, then continue with the imports below.
Colab may preload NumPy 2: pip changes its installed files, but already imported
modules remain in memory until restart. Skipping the restart can produce errors
such as `No module named 'numpy.rec'`. Do not rerun the installation cell after
restarting. No manual edits to dependency files or model wheels are needed.

## Extract and map skills

```python
from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor

extractor = SkillsExtractor(taxonomy_name="toy")  # Also "esco" or "lightcast"
job_ads = [
    "The job involves communication skills and maths skills",
    "The job involves Excel skills. You will also need good presentation skills",
    "You will need experience in the IT sector.",
]
for doc in extractor(job_ads):
    print("Entities:", [(entity.text, entity.label_) for entity in doc.ents])
    print("Skill spans:", doc._.skill_spans)
    print("Mapped skills:", doc._.mapped_skills)
```

`extractor(...)` returns one spaCy `Doc` per input text. An empty input list
returns `[]`; documents without skills have empty skill and mapping lists.
Invalid input raises a normal Python exception, rather than exiting the process.

`doc.ents` includes SKILL, EXPERIENCE, and BENEFIT entities. `doc._.skill_spans`
contains the skills after multiskill splitting. Mapping results can contain
`None` when an individual skill does not meet the taxonomy's matching threshold.

## Models, caching, and offline use

The model loader downloads the spaCy pipeline files directly from
[nestauk/en_skillner](https://huggingface.co/nestauk/en_skillner), avoiding its
invalid `any` wheel version and avoiding pip calls inside library code. The
multiskill classifier is downloaded with `huggingface_hub`; `skops` is not needed.
The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

Downloads and generated taxonomy embeddings go into a platform-specific user
cache (normally `~/.cache/ojd_daps_skills` on Linux), never into `site-packages`.
To select another cache, set `OJD_DAPS_SKILLS_CACHE` **before importing** the package:

```bash
export OJD_DAPS_SKILLS_CACHE=/path/to/writable/cache
```

After running the desired taxonomies once, reuse that cache offline:

```bash
export HF_HUB_OFFLINE=1
```

A missing cache still needs an initial online download. If precomputed taxonomy
embeddings are unavailable, the package warns and computes them locally; this
may take several minutes, especially for ESCO. Generated embeddings are saved
atomically for reuse. Corrupt model/JSON files are reported rather than silently
replaced. The pretrained classifier uses joblib/pickle, so custom classifier
repositories must be trusted.

For evaluation, `clean_text` changes some punctuation and CamelCase words, and
spaCy may tokenize differently from the benchmark. Compare gold and predicted
entities on the same text and token boundaries. The training model's entity
labels are distinct from the labels in the mapping taxonomy.

## Development and validation

```bash
python -m pip install -e . pytest ruff build
python -m pytest -m "not integration"
python -m ruff check ojd_daps_skills tests
python -m ruff format --check ojd_daps_skills tests
python -m build
```

The offline regression suite covers invalid/empty input, normalized mapping keys,
flat taxonomies, deterministic deduplication, cache writes, model loading, and
bundled taxonomy data. Integration tests download and run the real models:

```bash
python -m pytest -m integration
# After the first run, verify the three main pipelines without network downloads:
HF_HUB_OFFLINE=1 python -m pytest tests/test_live_models.py
```

CI tests installation and the offline suite across supported Python versions,
builds and installs the wheel, and runs the live model smoke tests on Linux.
The S3 preprocessing helpers are optional: install this fork with the `s3` extra
if you use them. They are not needed for skill extraction or taxonomy mapping.

### Contributor guidelines

The technical and working style guidelines can be found [here](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md).

If contributing, changes will need to be pushed to a new branch in order for our code checks to be triggered.

---

<small><p>This project was made possible via funding from the <a target="_blank" href="https://www.escoe.ac.uk/">Economic Statistics Centre of Excellence</a></p></small>

<small><p>Project template is based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
