# Training Data

Drop UTF-8 text files here (e.g. `.txt`, `.md`, `.tex`).

The Train tab can point at this folder (just `training_data`) and the server will
recursively load all supported files under it.

Notes:
- This folder is ignored by git (so you can sync large corpora locally).
- The trainer loads `*.txt/*.md/*.text/*.tex/*.rst` recursively.
- Hidden folders (like `training_data/.sources/`) are ignored, which is useful for keeping dataset clones without accidentally training on their metadata.

Helpers (optional):
- `scripts/fetch_tinystories.sh`: pulls `TinyStoriesV2-GPT4-valid.txt` into this folder via Git LFS.
- `scripts/build_assets_corpus.sh`: extracts `assets/*.pdf` into `training_data/assets_papers.txt`.
