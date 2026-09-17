# Copywriting & Tone Transformer

A Python script that takes a raw product description and automatically generates professional marketing copy tailored to a specific platform (**LinkedIn**, **Instagram**, or **Email**) and tone, using Google's Gemini API.

It uses dynamic, platform-specific prompt templates and exposes inference parameters (**temperature**, **top_p**) so you can control how creative or safe the generated copy is.

## Features

- **Platform-aware templates** — LinkedIn, Instagram, and Email each get their own prompt structure (hook style, hashtags, subject lines, ideal length) instead of generic copy.
- **Tone control** — choose from professional, witty, friendly, luxury, bold, playful, urgent, or minimalist.
- **Tunable creativity** — adjust `temperature` and `top_p` per platform, or use sensible built-in defaults.
- **Interactive or CLI mode** — run it with guided prompts, or pass flags for one-shot / scripted generation.
- **Batch mode** — generate copy for all three platforms in a single run with `--all_platforms`.
- **Save output** — optionally write results to a JSON file.

## Requirements

- Python 3.9+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

```bash
git clone https://github.com/hifzamaqbool/copy-transformer.git
cd copy-transformer
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install google-genai python-dotenv
```

## Setup

Create a file named `.env` in the project root (same folder as `copy_transformer.py`) containing:

```
GOOGLE_API_KEY=your-key-here
```

This keeps your key out of your terminal history and out of the code. `.env` is already excluded via `.gitignore`, so it will never be committed to version control.

Verify the key is detected before running the full script:

```bash
python copy_transformer.py --check_key
```

## Usage

### Interactive mode

```bash
python copy_transformer.py
```

You'll be prompted for the product name, description, platform, tone, and inference parameters.

### CLI mode

```bash
python copy_transformer.py \
  --product "EcoBrew Reusable Coffee Pod" \
  --description "A stainless steel reusable coffee pod that cuts single-use waste and fits all major machines." \
  --platform instagram \
  --tone playful \
  --temperature 0.9 \
  --top_p 0.95
```

### Generate for all platforms at once

```bash
python copy_transformer.py \
  --product "EcoBrew Reusable Coffee Pod" \
  --description "A stainless steel reusable coffee pod that cuts single-use waste." \
  --all_platforms \
  --out results.json
```

## CLI Options

| Flag | Description |
|---|---|
| `--product` | Product name |
| `--description` | Raw product description |
| `--platform` | `linkedin`, `instagram`, or `email` |
| `--tone` | `professional`, `witty`, `friendly`, `luxury`, `bold`, `playful`, `urgent`, `minimalist` |
| `--temperature` | Sampling temperature, 0.0–1.0 (creativity) |
| `--top_p` | Nucleus sampling parameter, 0.0–1.0 |
| `--max_tokens` | Max tokens for the generated copy |
| `--all_platforms` | Generate copy for all three platforms in one run |
| `--out` | Path to save output as JSON |
| `--check_key` | Check whether an API key is detected, then exit |

## How it works

1. User input (product, platform, tone) is injected into a platform-specific prompt template.
2. The compiled prompt is sent to the Gemini API along with `temperature` and `top_p` as generation parameters.
3. The model's response is returned as ready-to-publish copy for the chosen platform.

## Notes on inference parameters

- **Temperature** controls randomness — lower values (e.g. 0.5) produce safer, more consistent copy; higher values (e.g. 0.9) produce more varied, unexpected phrasing.
- **Top_p** controls how wide the pool of likely next-words is during generation — used alongside temperature to shape creativity.
- Instagram defaults to higher creativity than LinkedIn/Email, matching the more casual, expressive tone typical of that platform.

## License

This project is provided as-is for personal and educational use.
