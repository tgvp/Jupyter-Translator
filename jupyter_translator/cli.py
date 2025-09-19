#!/usr/bin/env python3
"""
Translate all Portuguese text in a Jupyter Notebook (.ipynb) to English,
keeping code cells intact. Translates Markdown (excluding fenced code blocks).
(Outputs are NOT translated by default.)

Usage:
  python translate_notebook.py input.ipynb [-o OUTPUT.ipynb]
  # optional: python translate_notebook.py input.ipynb --translate-outputs

Requires internet access (for the translation API) and:
  pip install -r requirements.txt   # tqdm, deep-translator, nbformat
"""

import argparse
import re
from pathlib import Path

from tqdm import tqdm
import nbformat
from deep_translator import GoogleTranslator

# --- Regexes to preserve structure in Markdown ---
FENCE_REGEX = re.compile(r"(```.*?```)", flags=re.DOTALL)                # fenced code blocks
INLINE_CODE_REGEX = re.compile(r"(`[^`]*`)", flags=re.DOTALL)            # inline `code`
MATH_BLOCK_REGEX = re.compile(r"(\${1,2}.*?\${1,2})", flags=re.DOTALL)   # $...$ or $$...$$


def fix_markdown_spacing(text: str) -> str:
    """
    Clean Markdown spans in Markdown cells:
    - Normalize Unicode/zero-width spaces.
    - Remove spaces INSIDE **bold**, __bold__, *italic*, _italic_, `inline`.
    - Ensure one space OUTSIDE inline code when glued to words.
    """
    if not text:
        return text

    # Normalize invisible spaces
    text = (text
            .replace("\u00A0", " ")  # NBSP
            .replace("\u202F", " ")  # narrow NBSP
            .replace("\u2009", " ").replace("\u200A", " ")
            .replace("\u2002", " ").replace("\u2003", " ").replace("\u2004", " ")
            .replace("\u2005", " ").replace("\u2006", " ").replace("\u2007", " ")
            .replace("\u2008", " ")
            .replace("\u2060", "")   # word joiner
            .replace("\u200B", "")   # zero-width space
            .replace("\u200C", "")   # ZWNJ
            .replace("\u200D", "")   # ZWJ
            .replace("\uFEFF", ""))  # BOM

    # Rebuild spans removing internal spaces
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"**{m.group(1).strip()}**", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__",   lambda m: f"__{m.group(1).strip()}__",   text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
                  lambda m: f"*{m.group(1).strip()}*", text, flags=re.DOTALL)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
                  lambda m: f"_{m.group(1).strip()}_", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`",     lambda m: f"`{m.group(1).strip()}`",     text, flags=re.DOTALL)

    # <<< NEW: add spacing around inline code when glued >>>
    text = re.sub(r"(?<=\w)(`[^`]+`)", r" \1", text)   # space before if glued
    text = re.sub(r"(`[^`]+`)(?=\w)", r"\1 ", text)    # space after if glued

    # Collapse multiple mid-line spaces (preserve Markdown two-spaces at EOL)
    text = re.sub(r"(?<!\n) {2,}(?!\n)", " ", text)

    return text

def _apply_glossary(text: str) -> str:
    """
    Domain normalizations applied to Markdown segments (non-code):
    - "data set(s)" -> "dataset(s)"
    - "numpy" -> "NumPy"
    - "pandas" -> "Pandas"
    - "dataframe(s)" or "data frame(s)" -> "DataFrame(s)"
    - "pre -processing" -> "preprocessing" (or change to "pre-processing" if preferred)
    """
    # data set(s) -> dataset(s)
    text = re.sub(r"\bdata\s*sets\b", "datasets", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdata\s*set\b", "dataset", text, flags=re.IGNORECASE)

    # dataframe(s) / data frame(s) -> DataFrame(s)
    text = re.sub(r"\bdata\s*frames\b", "DataFrames", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdata\s*frame\b", "DataFrame", text, flags=re.IGNORECASE)

    # numpy -> NumPy
    text = re.sub(r"\bnumpy\b", "NumPy", text, flags=re.IGNORECASE)

    # pandas -> Pandas
    text = re.sub(r"\bpandas\b", "Pandas", text, flags=re.IGNORECASE)

    # statistical "fashion" -> "mode"
    text = re.sub(r"\bfashion\b", "mode", text, flags=re.IGNORECASE)

    # pre -processing -> preprocessing
    text = re.sub(r"\bpre\s*-\s*processing\b", "preprocessing", text, flags=re.IGNORECASE)
    # If you prefer the hyphenated form, use the line below instead:
    # text = re.sub(r"\bpre\s*-\s*processing\b", "pre-processing", text, flags=re.IGNORECASE)

    return text


def _translate_segment(seg: str, translator: GoogleTranslator, markdown: bool) -> str:
    """
    Translate a plain segment. For Markdown segments, if the translator returns
    an unchanged string (common with numbered lists/headings), fall back to a
    line-by-line translation. Then apply Markdown spacing fixes and glossary.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().lower()

    # 1) Try normal translation
    try:
        t = translator.translate(seg)
        if not isinstance(t, str) or t == "":
            t = seg
    except Exception:
        t = seg

    # 2) Markdown fallback: if nothing changed, try line-by-line
    if markdown and _norm(t) == _norm(seg):
        try:
            parts = re.split(r"(\n)", seg)  # keep newlines
            out_parts = []
            for p in parts:
                if p == "\n":
                    out_parts.append(p)
                else:
                    try:
                        tp = translator.translate(p)
                        if not isinstance(tp, str) or tp == "":
                            tp = p
                    except Exception:
                        tp = p
                    out_parts.append(tp)
            t = "".join(out_parts)
        except Exception:
            # if anything goes wrong, keep original
            t = seg

    # 3) Post-process only for Markdown
    if markdown:
        t = fix_markdown_spacing(t)  # **pandas** (never ** pandas **)
        t = _apply_glossary(t)       # datasets, NumPy, Pandas, DataFrame, preprocessing
    return t


def translate_text(
    text: str,
    translator: GoogleTranslator,
    *,
    markdown: bool,  # True for Markdown cells; False for other contexts
) -> str:
    """
    Translation with safeguards.
    - If markdown=True:
        - Preserve ```fenced blocks```, `inline code`, and $...$/$$...$$.
        - Fix spacing around Markdown delimiters.
    - If markdown=False:
        - Preserve only LaTeX blocks ($...$ or $$...$$) and fenced (for safety),
          but do NOT treat inline code specially.
    """
    if not text or text.strip() == "":
        return text

    out = []

    # 1) Split by fenced blocks (always preserved)
    fence_parts = FENCE_REGEX.split(text)
    for i, fence_part in enumerate(fence_parts):
        if i % 2 == 1 and fence_part.startswith("```"):
            out.append(fence_part)
            continue

        if markdown:
            # 2) In Markdown, preserve inline code `...`
            inline_parts = INLINE_CODE_REGEX.split(fence_part)
            for k, inline_part in enumerate(inline_parts):
                if k % 2 == 1 and inline_part.startswith("`"):
                    out.append(inline_part)
                    continue

                # 3) Preserve LaTeX
                math_parts = MATH_BLOCK_REGEX.split(inline_part)
                for j, math_part in enumerate(math_parts):
                    if j % 2 == 1 and (math_part.startswith("$") or math_part.startswith("$$")):
                        out.append(math_part)
                    else:
                        out.append(_translate_segment(math_part, translator, markdown=True))
        else:
            # Outside Markdown, do not treat inline code specially
            math_parts = MATH_BLOCK_REGEX.split(fence_part)
            for j, math_part in enumerate(math_parts):
                if j % 2 == 1 and (math_part.startswith("$") or math_part.startswith("$$")):
                    out.append(math_part)
                else:
                    out.append(_translate_segment(math_part, translator, markdown=False))

    result = "".join(out)
    if markdown:
        # final pass to fix spacing around preserved inline code/backticks
        result = fix_markdown_spacing(result)
    return result

def translate_code_comments(source: str, translator: GoogleTranslator) -> str:
    """
    Translate only inline comments in code cells (lines starting with or containing '#').
    """
    lines = source.splitlines()
    new_lines = []
    for line in lines:
        if "#" in line:
            prefix, comment = line.split("#", 1)
            try:
                translated_comment = translator.translate(comment.strip())
                if not isinstance(translated_comment, str) or translated_comment == "":
                    translated_comment = comment.strip()
            except Exception:
                translated_comment = comment.strip()

            # apply domain glossary to comments (e.g., fashion -> mode)
            translated_comment = _apply_glossary(translated_comment)

            new_lines.append(f"{prefix}# {translated_comment}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)



def translate_notebook(
    input_path: Path,
    output_path: Path,
    source_language: str = "pt",
    target_language: str = "en",
    translate_outputs: bool = False,   # <- by default DO NOT translate outputs
):
    translator = GoogleTranslator(source=source_language, target=target_language)

    nb = nbformat.read(input_path, as_version=4)

    for cell in tqdm(nb.cells, desc="Translating notebook cells"):
        try:
            if cell.cell_type == "markdown":
                cell.source = translate_text(cell.source, translator, markdown=True)

            elif cell.cell_type == "raw":
                # Raw is not necessarily Markdown; keep it simple
                cell.source = translate_text(cell.source, translator, markdown=False)

            elif cell.cell_type == "code":
                # Translate only inline comments inside the source code
                if cell.source:
                    cell.source = translate_code_comments(cell.source, translator)

                if translate_outputs and "outputs" in cell and cell.outputs:
                    # Only process outputs if --translate-outputs is enabled
                    for out in tqdm(cell.outputs, desc="   Translating code outputs", leave=False):
                        try:
                            if out.get("output_type") == "stream" and "text" in out:
                                out["text"] = translate_text(out["text"], translator, markdown=False)

                            elif out.get("output_type") in {"execute_result", "display_data"}:
                                data = out.get("data", {})
                                if "text/plain" in data and isinstance(data["text/plain"], str):
                                    data["text/plain"] = translate_text(data["text/plain"], translator, markdown=False)
                                if "text" in data and isinstance(data["text"], str):
                                    data["text"] = translate_text(data["text"], translator, markdown=False)
                        except Exception:
                            continue

        except Exception:
            continue

    nbformat.write(nb, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Translate Jupyter notebook text between languages."
    )
    parser.add_argument("input", type=str, help="Path to input .ipynb")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Path to output .ipynb (default: <input>-<target>.ipynb)",
    )
    parser.add_argument("--source", type=str, default="pt", help="Source language code (default: pt)")
    parser.add_argument("--target", type=str, default="en", help="Target language code (default: en)")
    parser.add_argument(
        "--translate-outputs",
        action="store_true",
        help="Also translate textual cell outputs (OFF by default).",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else in_path.with_name(in_path.stem + f"-{args.target}.ipynb")
    )

    translate_notebook(
        in_path,
        out_path,
        source_language=args.source,
        target_language=args.target,
        translate_outputs=args.translate_outputs,
    )

    print(f"Translated notebook saved to: {out_path}")


if __name__ == "__main__":
    main()