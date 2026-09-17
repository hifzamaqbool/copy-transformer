"""
Copywriting & Tone Transformer
================================
Takes a raw product description and automatically generates professional
marketing copy tailored to a target platform (LinkedIn, Instagram, Email)
and a chosen tone, using dynamic prompt templates and tunable inference
parameters (temperature, top_p).

Setup
-----
1. pip install google-genai python-dotenv
2. Get a free API key from Google AI Studio: https://aistudio.google.com/apikey
3. In the SAME FOLDER as this script, create a new file named exactly ".env"
   (no filename before the dot) containing one line:
       GOOGLE_API_KEY=your-key-here
   (no quotes, no spaces around the =). The script will load it automatically.
4. Run:
       python copy_transformer.py

   Alternative (if you don't want a .env file): set the key directly in your
   terminal before running the script:
       Windows (PowerShell):  $env:GOOGLE_API_KEY="your-key-here"
       macOS / Linux:         export GOOGLE_API_KEY="your-key-here"
   Note this only lasts for that terminal session — you'd need to repeat it
   every time you open a new terminal. The .env file above avoids that.

Usage modes
-----------
- Interactive mode (default): just run the script, it will prompt you.
- CLI mode: pass arguments directly, e.g.

    python copy_transformer.py \
        --product "EcoBrew Reusable Coffee Pod" \
        --description "A stainless steel reusable coffee pod that cuts
                        single-use waste and fits all major machines." \
        --platform linkedin \
        --tone professional \
        --temperature 0.7 \
        --top_p 0.9

- Batch mode: generate the same product across ALL platforms at once with
  --all_platforms.
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Missing dependency. Install it with:\n    pip install google-genai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current folder, if present
except ImportError:
    pass  # dotenv is optional; falls back to real environment variables


# ---------------------------------------------------------------------------
# 1. CONFIG: model + default inference parameters
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 600

# Sensible starting points per-platform; the user can still override
# temperature / top_p manually via CLI flags or interactive prompts.
PLATFORM_DEFAULTS = {
    "linkedin": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 500},
    "instagram": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 350},
    "email": {"temperature": 0.5, "top_p": 0.85, "max_tokens": 700},
}

VALID_PLATFORMS = list(PLATFORM_DEFAULTS.keys())
VALID_TONES = [
    "professional", "witty", "friendly", "luxury",
    "bold", "playful", "urgent", "minimalist",
]


# ---------------------------------------------------------------------------
# 2. DYNAMIC PROMPT TEMPLATES
# ---------------------------------------------------------------------------
# Each platform gets its own structural template (format, length, hashtags,
# CTA style, subject lines, etc.) so the model is grounded in real platform
# conventions rather than producing generic copy.

PLATFORM_TEMPLATES = {
    "linkedin": """
You are a senior B2B/B2C marketing copywriter specializing in LinkedIn content.

Write a LinkedIn post for the following product.

Product name: {product_name}
Raw description: {description}
Desired tone: {tone}

Requirements:
- Hook in the first line (LinkedIn cuts off after ~2 lines, so make it count).
- 3-5 short paragraphs or a scannable structure with line breaks.
- Focus on value, credibility, and professional relevance (career, business
  impact, industry trend, or problem/solution framing).
- End with a clear call-to-action and 3-5 relevant hashtags.
- Do not use excessive emojis (max 1-2, optional).
- Target length: 80-150 words.

Return ONLY the final post text, ready to publish. No preamble, no notes.
""".strip(),

    "instagram": """
You are a social media copywriter specializing in Instagram captions.

Write an Instagram caption for the following product.

Product name: {product_name}
Raw description: {description}
Desired tone: {tone}

Requirements:
- Punchy, scroll-stopping opening line.
- Conversational, visual, and emotionally engaging language.
- Use line breaks for readability.
- Include 2-4 well-placed emojis that fit the tone.
- End with a strong call-to-action.
- Include 5-10 relevant hashtags at the end (mix of broad + niche).
- Target length: 40-100 words (excluding hashtags).

Return ONLY the final caption text, ready to publish. No preamble, no notes.
""".strip(),

    "email": """
You are an email marketing copywriter.

Write a marketing email for the following product.

Product name: {product_name}
Raw description: {description}
Desired tone: {tone}

Requirements:
- Provide a compelling subject line on the first line, formatted as:
  "Subject: <subject line>"
- Then a blank line, then the email body.
- Body should have a short greeting/hook, 2-3 short paragraphs highlighting
  benefits (not just features), and a single clear call-to-action button/link
  placeholder like [Shop Now] or [Learn More].
- Keep paragraphs short (2-3 sentences max) for scannability.
- Target length: 120-200 words for the body.

Return ONLY the subject line and email body, ready to send. No preamble, no notes.
""".strip(),
}


# ---------------------------------------------------------------------------
# 3. REQUEST OBJECT
# ---------------------------------------------------------------------------

@dataclass
class CopyRequest:
    product_name: str
    description: str
    platform: str
    tone: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        self.platform = self.platform.lower().strip()
        self.tone = self.tone.lower().strip()

        if self.platform not in VALID_PLATFORMS:
            raise ValueError(
                f"Invalid platform '{self.platform}'. "
                f"Choose from: {', '.join(VALID_PLATFORMS)}"
            )

        defaults = PLATFORM_DEFAULTS[self.platform]
        if self.temperature is None:
            self.temperature = defaults["temperature"]
        if self.top_p is None:
            self.top_p = defaults["top_p"]
        if self.max_tokens is None:
            self.max_tokens = defaults["max_tokens"]

        # Clamp values to safe ranges
        self.temperature = max(0.0, min(1.0, float(self.temperature)))
        self.top_p = max(0.0, min(1.0, float(self.top_p)))
        self.max_tokens = int(self.max_tokens)

    def build_prompt(self) -> str:
        """Compile the dynamic template for the selected platform."""
        template = PLATFORM_TEMPLATES[self.platform]
        return template.format(
            product_name=self.product_name,
            description=self.description,
            tone=self.tone,
        )


# ---------------------------------------------------------------------------
# 4. GENERATION ENGINE
# ---------------------------------------------------------------------------

class CopyGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_NAME):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set the GOOGLE_API_KEY environment "
                "variable in your terminal before running this script "
                "(see the Setup notes at the top of this file). Get a free "
                "key at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate(self, request: CopyRequest) -> str:
        prompt = request.build_prompt()

        config = genai_types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        return response.text.strip()


# ---------------------------------------------------------------------------
# 5. CLI / INTERACTIVE INTERFACE
# ---------------------------------------------------------------------------

def prompt_choice(label: str, choices: list, default: str = None) -> str:
    choices_display = "/".join(choices)
    default_hint = f" [{default}]" if default else ""
    while True:
        val = input(f"{label} ({choices_display}){default_hint}: ").strip().lower()
        if not val and default:
            return default
        if val in choices:
            return val
        print(f"  -> Please choose one of: {choices_display}")


def prompt_float(label: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    while True:
        raw = input(f"{label} [{default}] ({lo}-{hi}): ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  -> Must be between {lo} and {hi}")
        except ValueError:
            print("  -> Enter a valid number")


def run_interactive() -> CopyRequest:
    print("=== Copywriting & Tone Transformer ===\n")
    product_name = input("Product name: ").strip()
    print("Raw product description (paste text, then press Enter):")
    description = input("> ").strip()

    platform = prompt_choice("Platform", VALID_PLATFORMS, default="linkedin")
    tone = prompt_choice("Tone", VALID_TONES, default="professional")

    print("\n-- Inference parameters (press Enter to accept platform defaults) --")
    defaults = PLATFORM_DEFAULTS[platform]
    temperature = prompt_float("Temperature (creativity)", defaults["temperature"])
    top_p = prompt_float("Top_p (nucleus sampling)", defaults["top_p"])

    return CopyRequest(
        product_name=product_name,
        description=description,
        platform=platform,
        tone=tone,
        temperature=temperature,
        top_p=top_p,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate platform-tailored marketing copy from a raw product description."
    )
    parser.add_argument("--product", type=str, help="Product name")
    parser.add_argument("--description", type=str, help="Raw product description")
    parser.add_argument(
        "--platform", type=str, choices=VALID_PLATFORMS,
        help="Target platform"
    )
    parser.add_argument(
        "--tone", type=str, choices=VALID_TONES, default="professional",
        help="Desired tone (default: professional)"
    )
    parser.add_argument("--temperature", type=float, default=None,
                         help="Sampling temperature 0.0-1.0 (controls creativity)")
    parser.add_argument("--top_p", type=float, default=None,
                         help="Nucleus sampling parameter 0.0-1.0")
    parser.add_argument("--max_tokens", type=int, default=None,
                         help="Max tokens for the generated copy")
    parser.add_argument("--all_platforms", action="store_true",
                         help="Generate copy for LinkedIn, Instagram, and Email in one run")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to save output as JSON")
    return parser


def main():
    parser = build_arg_parser()
    parser.add_argument("--check_key", action="store_true",
                         help="Just check whether an API key is detected, then exit")
    args = parser.parse_args()

    if args.check_key:
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            masked = key[:6] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"Key detected: {masked} (length {len(key)})")
        else:
            print("No GOOGLE_API_KEY detected in this process's environment.")
            print("Checked: real env vars and a .env file in the current folder.")
        sys.exit(0)

    try:
        generator = CopyGenerator()
    except EnvironmentError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = {}

    # --- CLI mode ---
    if args.product and args.description:
        platforms_to_run = VALID_PLATFORMS if args.all_platforms else [args.platform or "linkedin"]

        for platform in platforms_to_run:
            request = CopyRequest(
                product_name=args.product,
                description=args.description,
                platform=platform,
                tone=args.tone,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            print(f"\nGenerating {platform.upper()} copy "
                  f"(temperature={request.temperature}, top_p={request.top_p})...\n")
            copy_text = generator.generate(request)
            print("-" * 60)
            print(copy_text)
            print("-" * 60)
            results[platform] = copy_text

    # --- Interactive mode ---
    else:
        request = run_interactive()
        print(f"\nGenerating {request.platform.upper()} copy "
              f"(temperature={request.temperature}, top_p={request.top_p})...\n")
        copy_text = generator.generate(request)
        print("=" * 60)
        print(copy_text)
        print("=" * 60)
        results[request.platform] = copy_text

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved output to {args.out}")


if __name__ == "__main__":
    main()