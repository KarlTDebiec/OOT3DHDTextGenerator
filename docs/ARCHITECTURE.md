# Architecture

## Purpose

OOT3DHDTextGenerator builds high-resolution replacement text textures for The Legend of Zelda: Ocarina of Time 3D. The package watches or processes texture dumps, identifies the game's generated glyph tiles, maps those tiles to characters, and renders replacement images at a higher scale.

## Package Layout

* `oot3dhdtextgenerator.apps`: Flask-backed tools for inspecting and assigning character images.
  * `char_assigner`: interactive workflow for assigning unrecognized glyph tiles and reviewing OCR predictions.
  * `char_inspector`: inspection workflow for browsing known character data.
* `oot3dhdtextgenerator.cli`: command-line entry points. CLI modules should stay thin: parse arguments, validate paths, format templated defaults, and delegate work to app or utility classes.
* `oot3dhdtextgenerator.common`: shared infrastructure for argument parsing, CLI base classes, CSV helpers, exceptions, filesystem operations, logging, subprocesses, testing, and validation.
* `oot3dhdtextgenerator.core`: domain objects and ML primitives, including assignment datasets, training datasets, base64 image conversion, and the OCR model.
* `oot3dhdtextgenerator.data`: bundled package data, including character lists, assignment CSVs, generated datasets, and model checkpoints.
* `oot3dhdtextgenerator.image`: image processors that transform dumped texture images into high-resolution replacements.
* `oot3dhdtextgenerator.utilities`: reusable non-interactive workflows, such as model training and training dataset generation.
* `test`: pytest tests organized to mirror the package hierarchy where practical.

## Data Flow

1. Texture images from Citra or fixture data are represented as alpha-channel arrays split into fixed 16 by 16 glyph cells.
2. `AssignmentDataset` records known glyph bytes in `assigned.csv` and unknown glyph bytes in `unassigned.csv`.
3. Training utilities generate image arrays and specifications for model training from assignment data and bundled character frequency data.
4. `Model` predicts likely characters for unknown glyphs, with optional prior weighting from character frequencies.
5. The Flask assignment app presents unknown and assigned glyphs for human review, then writes updated assignment CSVs.
6. `OOT3DHDTextProcessor` looks up every glyph in a texture, renders the corresponding text with PIL, and returns a scaled RGBA replacement image.

## Dependency Boundaries

* CLI modules may depend on app, core, data, image, utility, and common modules, but should not contain domain-heavy logic.
* App route modules should stay focused on request handling and presentation; durable state changes should go through app or core classes.
* Core modules should avoid depending on Flask or CLI concerns.
* Image processors may depend on core datasets and validation helpers, but should keep filesystem side effects explicit.
* Common modules should remain broadly reusable and avoid importing from domain-specific packages such as `apps`, `core`, `image`, or `utilities`.
* Tests may use helpers from `oot3dhdtextgenerator.common.testing`, but production modules should not depend on test code.

## Data and Artifacts

* Treat bundled CSVs, `.npy` arrays, and `.pth` checkpoints as versioned artifacts. Avoid rewriting them incidentally while changing code.
* Keep generated debug artifacts outside package data unless they are intentionally added as fixtures.
* When changing data formats, update readers, writers, tests, and docs together so existing artifacts remain understandable.

## Operational Notes

* Prefer deterministic file writes for datasets and CSVs. Write through a temporary file and replace the target where practical.
* Validate user-provided paths at the boundary using `oot3dhdtextgenerator.common.validation`.
* Preserve CUDA, MPS, and CPU fallback behavior when changing model-loading or inference paths.
* Keep Flask apps usable from the CLI without requiring package-level global mutable state.
