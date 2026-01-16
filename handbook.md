# PPUC-Serum-Colorizer Handbook

This handbook captures editor behaviors and workflows that differ from the legacy app. It is meant to be updated as new features land.

## Navigation and Selection

- Use the back/forward buttons (`<`/`>`) on each canvas to jump through recently viewed frames, sprites, images, and backgrounds.
- The preview row supports multi-selection with native shortcuts (Cmd/Ctrl and Shift).
- The last selected frame in a multi-selection is the active canvas frame; clicking within the selection switches the active frame.
- Use the preview-row filter button to show only the selected frames.

## Preview Row

- Frames are shown stacked: colorized on top, original (orange) below, and HD below that if available.
- Use the filter button to show only frames that use the currently selected mask, dynamic mask, background, or sprite.
- Use the HD filter button to show only frames that have HD data.
- Use the mask overlay toggle to show masks on the original preview frames.
- Use the rotate toggle to preview color rotations in the preview row.

## Canvas Controls

- **Fit to View** fits the current frame/sprite/background to the canvas.
- **Grid toggle** shows/hides the pixel grid.
- **Original toggle** shows/hides the original (orange) reference under the colorized frame.
- **Mask/Dynamic buttons** select the mask mode for the original reference (only one active).
- **Background Mask** is independent and edits the background mask on the colorized frame.
- **Background toggle** shows/hides the background layer while editing frames.
- **HD toggle** switches between SD and HD for frames, sprites, and backgrounds.

## Drawing Tools

- Tools: point, line, rectangle, filled rectangle, circle, filled circle, ellipse, filled ellipse, magic fill, color picker.
- Right-click erases for pixel edits; Shift erases when editing masks/dynamic masks.
- ESC cancels the current drawing preview or palette action.

## Frames and Multi-Frame Editing

- Drawing actions on frames apply to all currently selected frames.
- Mask edits apply to the assigned mask, which is shared across all frames using it.
- Dynamic mask edits apply to the current dynamic set for all selected frames.
- Background assignment and background mask edits apply to all selected frames.

## Masks (Comparison Masks)

- 64 shared masks; each frame references exactly one.
- Edit masks on the **original** (orange) frame.
- Reorder masks with `Up`/`Down` without breaking frame assignments.

## Dynamic Masks

- 32 dynamic masks per frame; each pixel stores a dynacouche index.
- Edit dynamic masks on the **original** (orange) frame.
- Dynamic masks use per-frame dynamic color sets (2-bit or 4-bit).
- The dynamic mask list shows a mask preview plus a mini color strip.

## Backgrounds

- Assign backgrounds per frame (drag & drop or inspector).
- Each frame has a background mask drawn on the **colorized** frame.
- Backgrounds can be edited on the Background canvas with the same drawing tools.

## Sprites

- Sprite canvas shows colorized sprite on top and the original reference below.
- Sprite detection areas (4) are edited on the original (bottom) sprite.
- Drag & drop sprites onto a selected frame zone to assign.
- Sprite overlays are drawn on top of frames and cannot be painted over.

## Sprite Zones (Frame Detection Zones)

- **Create a zone:** In Components → Sprite Zones, click `Add Zone`. A new zone is created for the current frame and shown on the frame canvas.
- **Select a zone:** Click a zone in the list to show its bounding box on the frame canvas.
- **Edit a zone:** Enable Zones on the frame canvas, then draw a rectangle on the *original* (bottom) frame. This updates the selected zone.
- **Assign a sprite to a zone:** With Zones enabled, drag a sprite from Components → Sprites onto the original frame on the canvas. The sprite is added to the selected zone.
- **Reorder sprites in a zone:** Use `Up`/`Down` in the Sprite Zones tab to reorder sprites assigned to the selected zone.
- **Remove a sprite from a zone:** Select the sprite in the zone list and click `Remove`.
- **Remove a zone:** Select the zone and click `Remove Zone`. This clears all sprite assignments for that zone.

## HD Sprites

- Use the HD toggle on the sprite canvas to switch between SD and HD sprites.
- `Create HD` in the HD tab will generate an HD sprite for the selected sprite when none exists.

## HD Frames and Backgrounds

- HD frames are 256x64; SD frames are 128x32.
- Use the HD tab to create or delete HD frames/backgrounds/sprites.
- Upscaling modes include nearest/scale2x/bilinear/bicubic with brightness correction.
- HD previews are double-size in Components and preview row.
- An all-black HD background is treated as “missing.”

## Colors, Palettes, and Rotations

- Full palettes: 256 sets × 64 RGB565 colors.
- Reduced palettes: 64 sets of 4 or 16 colors (2-bit/4-bit ROMs).
- Dynamic color sets: 32 per frame, used by dynamic masks.
- Rotation sets: editable in the Colors tab; use `Set Slot` to assign, and the gradient tool for ranges.
- Selecting a color sets the active draw color; the color picker updates the active draw color.

## Undo/Redo and Settings

- Undo/redo stacks are separated for frames, masks, backgrounds, sprites, and palettes.
- Undo/redo applies to all selected frames where relevant.
- Settings dialog allows adjusting undo and navigation history sizes.
