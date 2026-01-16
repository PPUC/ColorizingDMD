# Codex Notes: PPUC-Serum-Colorizer Editor + libserum

## Scope
- This file documents development context and non-obvious legacy-format behavior.
- It must not change cROM/cROMc data formats.
- The editor must remain a drop-in replacement for the legacy app.
- The new editor targets Serum V2 only; libserum still supports V1 for other clients.
- The editor UX/workflow documentation lives in `handbook.md`.

## Data format constraints (legacy compatibility)
- Do not alter the binary layout of cROM/cROMc files.
- All editor changes must map to existing fields in the legacy format.

## Key serum relationships (from libserum)
- Dynamic masks are per-frame pixel maps:
  - `dynamasks[frame][pixel]` (SD) and `dynamasks_extra[frame][pixel]` (HD).
  - Each pixel stores a dynacouche index (0..31) or 255 for none.
- Dynamic colors are per-frame sets:
  - `dyna4cols_v2[frame][dynacouche * nocolors + originalPixel]` (SD)
  - `dyna4cols_v2_extra[frame][...]` (HD)
- Sprite dynamic colors are per-sprite:
  - `dynaspritemasks[sprite][pixel]` selects a dynacouche index.
  - `dynasprite4cols[sprite][dynacouche * nocolors + originalPixel]`.
- Sprite detection:
  - Frame detection zones: `framespriteBB[frame][slot*4..]`.
  - Per-sprite detection areas: `spritedetareas[sprite][area*4..]`.
  - Matching + placement logic is in libserum `Check_Spritesv2`.
- Backgrounds:
  - `backgroundIDs[frame][0]` selects background.
  - `backgroundmask[frame][pixel]` masks background visibility.
  - Applied before dynamic masks; sprites applied after.

## Rendering order (v2, runtime behavior)
1) Background (if original pixel is 0 and mask allows).
2) Dynamic mask replacement (dyna4cols).
3) Color rotations (if enabled).
4) Sprites overlay (spritecolored / dynasprite4cols).

## Editor goals
- WYSIWYG: render canvas + preview using libserum logic.
- Avoid data duplication for large projects.
- Preview row only renders the visible subset of frames.

## Editor mask behavior
- Frame uses exactly one comparison mask and one dynamic mask assignment.
- Mask editing happens on the original (orange) frame.
- Background mask is independent from mask/dynamic mask toggles.

## HD/SD rules
- HD is 256x64 (SD is 128x32).
- HD previews are double size vs SD previews.
- When HD is absent, treat all-black HD backgrounds as non-existent.
- HD background masks should be double-scaled when generated from SD.

## Open integration task
- Add libserum editor API (non-owning view) to render frames + sprite matches.
- Keep libserum free of new dependencies (no Qt/OpenCV).
