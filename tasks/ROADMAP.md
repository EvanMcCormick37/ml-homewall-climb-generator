## Development Roadmap

### Feature: `image_edges` — Image-to-Wall Coordinate Calibration

**Goal:** Add `image_edges: [left, right, bottom, top]` (ft) to `LayoutMetadata`. This field defines
what wall-coordinate ft values correspond to each edge of the layout photo, allowing the WallCanvas
to correctly overlay holds/sizes even when the photo is not perfectly cropped to the wall boundary.

---

#### Task 1 — Types & API [ STATUS: COMPLETED ]
- Add `image_edges: number[]` to `LayoutMetadata` in `src/types/layout.ts`
- Add `LayoutUpdate` interface to `src/types/layout.ts`
- Add `updateLayout(layoutId, data: LayoutUpdate)` to `src/api/layouts.ts` (calls `PUT /layouts/{id}`)
- Export `LayoutUpdate` from `src/types/index.ts` if needed

#### Task 2 — WallCanvas coordinate transform [ STATUS: COMPLETED ]
- Add optional `imageEdges?: [number, number, number, number]` prop to `WallCanvasProps`
- Update `toPixelCoords` to use imageEdges (fallback to `[0, wallW, 0, wallH]`)
- Update `activeSize` overlay normalization to use imageEdges
- Update `renderExportImage` in `sharing.ts` to accept and use `imageEdges`

#### Task 3 — Image Alignment UI in sizes.tsx [ STATUS: COMPLETED ]
- Add `ImageAlignmentOverlay` component (draggable wall-boundary box on the image)
- Add "Image Alignment" section to the right-side panel in `sizes.tsx`
- Initialize local `imageEdges` state from `layout.metadata.image_edges` (fallback to `[0,W,0,H]`)
- Add 4 numeric inputs (L/R/B/T) bound to imageEdges state
- Add "Save" button that calls `updateLayout(layoutId, { image_edges: imageEdges })`
- Show success/error feedback

#### Task 4 — Wire up imageEdges in set.tsx and view.tsx [ STATUS: COMPLETED ]
- In `set.tsx`: read `layout.metadata.image_edges`, pass to `WallCanvas` and `renderExportImage`
- In `view.tsx`: read `layout.metadata.image_edges`, pass to `WallCanvas`

---

#### Organization instructions:

After reading this Roadmap, take the first task shown here, memorize it, mark it as STATUS: WORKING, and create a more detailed version of it in CURRENT_TASK.md, breaking it up into more detailed sub-tasks. Then, go through the sub-tasks in CURRENT_TASK.md and work through them one-by-one (marking them as INCOMPLETE, WORKING, and COMPLETED in the same way you're doing here). Once all of the subtasks in CURRENT_TASK.md are completed, empty out the CURRENT_TASK.md tasklist, come back here, and mark the task as COMPLETED. Then start the next task on the list, following the same procedure.

#### Dealing with bugs:

If you run into trouble or come across a bug while performing this update, write down the bug in BUGS.md. If the bug is something which must be fixed for work to continue, then do your best to fix it, and delete it from BUGS.md when you have fixed it. However, if it isn't necessary to fix it immediately, then just move on to the next task. I will look through the bugs later and work on them.

#### What to do if you're unsure of what to do:

This is an autonomous task, and I expect it may take awhile and involve a few nuanced or tricky decisions. Use your best judgement whenever possible. If you must ask for clarification, then ask for clarification. Otherwise, go to TRICKY_DECISIONS.md and write down the question/decision you are having trouble with and your thoughts on the situation. Then, if you feel confident enough to make a call, write down the decision you've made and continue with the work. If you can't make the decision, but you feel like you can continue to do useful work on other parts of this roadmap, then write down the decision, and move on to other work.

#### One more thing

Don't commit these changes to git yet. just leave them as is for now.
