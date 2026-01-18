# Training Data

Drop UTF-8 text files here (e.g. `.txt`, `.md`, `.tex`).

The Train tab can point at this folder (just `training_data`) and the server will
recursively load all supported files under it.

Notes:
- This folder is ignored by git (so you can sync large corpora locally).
- The trainer loads `*.txt/*.md/*.text/*.tex/*.rst` recursively.
