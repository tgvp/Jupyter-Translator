# Jupyter-Translator
A Jupyter Notebook translator that uses Deep Translator and Google API to translate markdown cells without messing with the code formatting.

Translate the textual content of Jupyter notebooks while **preserving code cells, LaTeX, and formatting**.

- ✅ Default translation **PT → EN** (configurable via `--source/--target`)
- ✅ Keeps code cells intact but also translates inline Python comments
- ✅ Fixes Markdown spacing (**bold**/*italic*/``inline``) and preserves LaTeX (`$...$`, `$$...$$`)
- ✅ Domain glossary
- ✅ Optional: translate textual outputs with `--translate-outputs`

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Alternatively, you can pip install using the `.toml`

1. Git Clone the repository
2. `cd` into the project directory and pip install it
```bash
pip install -e .
```

## Usage

### CLI
```bash
jupyter-translator input.ipynb -o output.ipynb
# translate outputs too (OFF by default)
jupyter-translator input.ipynb --translate-outputs
# other languages
jupyter-translator input.ipynb --source pt --target es
```

### As a module
```bash
python -m jupyter_translator input.ipynb -o output.ipynb
```

## Notes
- Fenced code blocks inside Markdown (``` ... ```) are not translated.
- LaTeX math segments ($...$ or $$...$$) are preserved as-is.
- Only textual contents are translated; code is left intact. You can enable output translation with `--translate-outputs`.
- Requires internet access (uses GoogleTranslator via deep-translator).

## License
This project is licensed under the [MIT License](LICENSE).
