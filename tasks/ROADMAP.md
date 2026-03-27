## Organization instructions:

After reading this Roadmap, take the first task shown here, memorize it, mark it as STATUS: WORKING, and create a more detailed version of it in CURRENT_TASK.md, breaking it up into more detailed sub-tasks. Then, go through the sub-tasks in CURRENT_TASK.md and work through them one-by-one (marking them as INCOMPLETE, WORKING, and COMPLETED in the same way you're doing here). Once all of the subtasks in CURRENT_TASK.md are completed, empty out the CURRENT_TASK.md tasklist, come back here, and mark the task as COMPLETED. Then start the next task on the list, following the same procedure.

## Dealing with bugs:

If you run into trouble or come across a bug while performing this update, write down the bug in BUGS.md. If the bug is something which must be fixed for work to continue, then do your best to fix it, and delete it from BUGS.md when you have fixed it. However, if it isn't necessary to fix it immediately, then just move on to the next task. I will look through the bugs later and work on them.

## What to do if you're unsure of what to do:

This is an autonomous task, and I expect it may take awhile and involve a few nuanced or tricky decisions. Use your best judgement whenever possible. If you must ask for clarification, then ask for clarification. Otherwise, go to TRICKY_DECISIONS.md and write down the question/decision you are having trouble with and your thoughts on the situation. Then, if you feel confident enough to make a call, write down the decision you've made and continue with the work. If you can't make the decision, but you feel like you can continue to do useful work on other parts of this roadmap, then write down the decision, and move on to other work.

## One more thing

Don't commit these changes to git yet. Just leave them as is for now.

---

## Refactor: Scale generate endpoint to 10+ concurrent users

**Goal:** Replace the singleton `ClimbDDPMGenerator` with a `GeneratorPool` so up to N requests can generate climbs concurrently without blocking each other. FastAPI already runs sync handlers in a thread pool; the pool gives each thread its own model instance. No new dependencies required.

### Tasks

- [x] **Task 1** — Add `GENERATOR_POOL_SIZE` config setting  STATUS: COMPLETED
- [x] **Task 2** — Implement `GeneratorPool` in `generation_utils.py`  STATUS: COMPLETED
- [x] **Task 3** — Update `utils/__init__.py` exports  STATUS: COMPLETED
- [x] **Task 4** — Update `generation_service.py` to use pool  STATUS: COMPLETED
- [x] **Task 5** — Update `layout_service.py` manifold refresh calls  STATUS: COMPLETED
