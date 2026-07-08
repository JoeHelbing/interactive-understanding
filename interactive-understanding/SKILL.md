---
name: interactive-understanding
description: Use when a terminal agent should discuss a PDF paper by building a Docling text-and-image context pack
---

# Interactive Understanding

Use this skill when the user wants to read, discuss, or ask questions about a
PDF paper, report, slide deck, or other visually rich document.

This is a direct-use workflow for the current terminal agent. Build a local
Docling context pack, load the extracted text and generated image sheets into
model context, and answer in the current conversation. Do not delegate to another
agent or build a separate orchestration layer.

Implementation: `scripts/` is a small uv project. Run commands from that
directory.

## Context Pack Contents

`iu-context-pack` accepts a local PDF path, `file://` URL, `http://` URL, or
`https://` URL. It:

1. reads or downloads the PDF bytes
2. validates the bytes start with `%PDF-`
3. writes validated bytes as `source.pdf`
4. runs Docling with referenced image export
5. converts Docling page/artifact images to WebP
6. crops Docling bboxes for formulas, code blocks, pictures, and tables
7. builds crop contact sheets
8. builds page overview sheets
9. writes 130 DPI full-page image copies
10. writes `text.md` with reading-order text and visual links
11. writes `manifest.json` and `source.json` for navigation

By default, generated crop and page sheets target native Codex CLI high-detail
image limits: max 2048px on either side, 32px patches, and 2500 patches per
sheet. The script reduces the number of crops/pages on each sheet before
reducing thumbnail fidelity, so more sheets are preferred over hidden
model-side downscaling.

Important files:

| Path | Purpose |
|------|---------|
| `text.md` | Reading-order text with visual links |
| `crop-sheets/*.webp` | Formula, code, picture, and table sheets |
| `page-sheets/*.webp` | Page overview sheets |
| `page-images-130dpi/*.webp` | Individual full-page images at 130 DPI |
| `visual-crops/*.webp` | Individual visual crops |
| `manifest.json` | Relative paths and sheet coordinates |
| `source.pdf` | Validated PDF used by Docling |

## Rules

- Use `iu-context-pack` as the entrypoint.
- Load `text.md` before answering broad content questions.
- Load page sheets and crop sheets before making claims about layout, figures,
  equations, tables, code, or diagrams.
- Treat text as extracted evidence and sheets/crops as visual evidence. If they
  disagree, inspect the relevant sheet or crop.
- Cite inspected pages, figure labels, table labels, crop labels, or note that
  only text was inspected.
- Pi terminal agents cannot render LaTeX math. Explain equations with
  Markdown/plain-text math: ASCII inline forms, code blocks, bullet derivations,
  or tables. Refer to formula crops/page labels for exact notation when needed.
- Create persistent discussion records as soon as the pack is built, before the
  first substantive paper answer.
- Load those records during initial paper setup so context resets can recover
  them.
- After every paper question, append the question and answer to the Q&A record.

## Persistent Discussion Records

Maintain three records per paper:

1. Q&A record - every user paper question and answer, with inspected evidence or
   uncertainty.
2. Further reading record - papers, books, sections, concepts, links, or search
   queries to pursue later.
3. Working notes record - experiments, follow-up questions, hypotheses,
   implementation ideas, critiques, and non-Q&A notes.

Storage priority:

1. User-named storage target wins. For one Markdown file, keep three top-level
   sections. For a directory, create `q-and-a.md`, `further-reading.md`, and
   `working-notes.md`.
2. If Basic Memory works (`bm status --project main --local --json`), create or
   update three Basic Memory notes with `bm tool ... --project main --local`.
3. Otherwise create pack-local files under
   `<pack-output-dir>/discussion-notes/` and tell the user the record location.

Use the PDF metadata title when available; otherwise use the PDF filename or URL
stem. Use the context-pack output directory basename as `<paper-slug>` and keep
it stable.

For Basic Memory records:

- Always `search-notes` before `write-note` to avoid duplicates.
- Reuse exact-title matches.
- If multiple plausible non-exact matches appear, ask which record to reuse or
  create a new dated set.
- Read/load the exact existing or newly created records before answering the
  first substantive question.

## Workflow

Check help:

```bash
uv run iu-context-pack --help
```

### 1. Build the context pack

Remote PDF:

```bash
uv run iu-context-pack \
  'https://example.com/paper.pdf' \
  --output-dir /tmp/iu-paper-context/example-paper \
  --json
```

Local file:

```bash
uv run iu-context-pack \
  /inputs/paper.pdf \
  --output-dir /tmp/iu-paper-context/paper \
  --json
```

Keep Docling defaults unless the user asks otherwise: OCR enabled, OCR engine
`auto`, no forced full-page OCR over existing PDF text, and
`--docling-device auto`.

### 2. Inspect the manifest

```bash
jq '.text, .visual_count, [.crop_sheets[].sheet], [.page_sheets[].sheet]' \
  /tmp/iu-paper-context/example-paper/manifest.json
```

Use this to choose which sheets to load first.

### 3. Create and load discussion records

Resolve record storage before the first substantive answer.

Basic Memory lookup:

```bash
bm tool search-notes "Paper Q&A: <paper title>" --project main --local
bm tool search-notes "Paper Further Reading: <paper title>" --project main --local
bm tool search-notes "Paper Working Notes: <paper title>" --project main --local
```

Create missing records with Project Context frontmatter:

```bash
bm tool write-note --title "Paper Q&A: <paper title>" \
  --folder "projects/paper-reading" \
  --project main --local <<'EOF'
---
title: "Paper Q&A: <paper title>"
type: note
schema: Project Context
created_from_cwd: /path/to/current/repo
git_project: <owner>/<repo>
git_root: /path/to/current/repo
tags:
  - paper-reading
  - q-and-a
---
# Paper Q&A: <paper title>

## Source

- Context pack: /tmp/iu-paper-context/<paper-slug>
- Source PDF: /tmp/iu-paper-context/<paper-slug>/source.pdf

## Q&A
EOF
```

Create matching `Paper Further Reading: <paper title>` and
`Paper Working Notes: <paper title>` notes with `further-reading` or
`working-notes` tags. Then read/load all three exact note locations.

Pack-local fallback:

```bash
mkdir -p /tmp/iu-paper-context/<paper-slug>/discussion-notes
cat > /tmp/iu-paper-context/<paper-slug>/discussion-notes/q-and-a.md <<'EOF'
# Paper Q&A: <paper title>

## Source

- Context pack: /tmp/iu-paper-context/<paper-slug>
- Source PDF: /tmp/iu-paper-context/<paper-slug>/source.pdf

## Q&A
EOF
cat > /tmp/iu-paper-context/<paper-slug>/discussion-notes/further-reading.md <<'EOF'
# Paper Further Reading: <paper title>

## Further Reading
EOF
cat > /tmp/iu-paper-context/<paper-slug>/discussion-notes/working-notes.md <<'EOF'
# Paper Working Notes: <paper title>

## Working Notes
EOF
```

Read/load the three files and tell the user the pack-local `discussion-notes/`
directory is the ongoing record.

### 4. Load text, records, and sheets

Use file-read tools on the text, discussion records, and relevant sheets:

```text
/tmp/iu-paper-context/example-paper/text.md
/tmp/iu-paper-context/example-paper/discussion-notes/q-and-a.md
/tmp/iu-paper-context/example-paper/discussion-notes/further-reading.md
/tmp/iu-paper-context/example-paper/discussion-notes/working-notes.md
/tmp/iu-paper-context/example-paper/page-sheets/page-sheet-001.webp
/tmp/iu-paper-context/example-paper/crop-sheets/crop-sheet-001.webp
```

If a sheet is not detailed enough for a graph or visual detail, load the matching
`visual-crops/*.webp` file.

### 5. Discuss and append records

Answer in the active conversation using loaded text, images, and prior records.
After each paper question, append to Q&A immediately:

```markdown
### YYYY-MM-DD HH:MM - <short question title>

**Question:** <user's question>

**Answer:** <assistant's answer>

**Evidence inspected:** <pages, figures, tables, crop labels, or "text only">
```

Append recommended sources/background to the further-reading record. Append
experiments, critiques, hypotheses, and non-Q&A follow-ups to working notes. If
an append fails, say so and provide the intended record path.

## Budget Strategy

- Always: load `text.md`, all page sheets, and all crop sheets when sheet
  count is small.
- Long papers: load `text.md`, skim page sheets for structure, then load crop
  sheets relevant to the question.
- Dense visual papers: prefer crop sheets and individual `visual-crops/*.webp`
  for exact figure/table/equation detail.
- Garbled Docling text: rely more on page sheets/crops and state extraction
  uncertainty.
