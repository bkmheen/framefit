# framefit

**Version:** 0.1.0

Detect a document or presentation slide inside a photo, correct its perspective,
and crop it to a clean full-frame image.

Point it at a photo of a projected slide, a whiteboard, or a paper document taken
at an angle — framefit finds the bright rectangle, removes the keystone/perspective
distortion, and saves the flattened, full-frame result.

## Status

Early scaffolding. Design and implementation in progress.

## Planned pipeline

1. **Detect** the slide/screen quadrilateral in the image.
2. **Correct** perspective (dewarp the keystone).
3. **Crop** to the slide bounds — full frame, edges trimmed.
4. **Save** the flattened image.

## Project records

- [CHANGELOG.md](CHANGELOG.md) — versioned change history (`major.minor.patch`).
- [DEVLOG.md](DEVLOG.md) — chronological engineering notes and decisions.

If any record file grows too large it will be split by topic (e.g.
`CHANGELOG_<subtitle>.md`) and linked from here.

## License

TBD (intended to be an OSI-approved license before public release).
