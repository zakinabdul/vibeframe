# Paper Design Improvement Plan

## Current Active Shape

- Backend entrypoint: `backend/main.py`
- Active generation pipeline: `backend/agents.py`
- Paper transport: `backend/app/paper_mcp.py`
- Frontend control panel: `frontend/app/page.tsx`
- Main live routes used by the UI: `/generate`, `/refine`, `/canvas/current`, `/stream`, `/paper/open`, `/session/reset`

## What To Keep

- Keep the active multi-agent flow in `backend/agents.py`.
- Keep the SSE stream contract because the frontend depends on it for live feedback.
- Keep `backend/app/paper_mcp.py` protocol handling intact; it already handles SSE and session retries correctly.

## What Is Legacy Or Compatibility Only

- `backend/app/orchestrator.py` and `backend/app/groq_designer.py` are legacy compatibility paths for `/api/design`.
- The empty `frontend/components/` folder is safe to remove if no shared components are introduced.
- The styling in `frontend/app/page.module.css` and `frontend/app/globals.css` contains older visual layers that can be simplified further.

## Improvement Phases

1. Tighten design prompting.
   - Make the designer output feel more editorial and less template-like.
   - Keep the critic strict enough to reject generic layouts.
   - Preserve the current JSON contract so the backend stays stable.

2. Reduce UI clutter in the control panel.
   - Remove stale styles and dead visual states.
   - Keep only the classes that the current page uses.
   - Make the shell feel calmer and more intentional.

3. Decide whether the legacy path should stay.
   - If `/api/design` is still needed, leave the compatibility code in place.
   - If not, delete the legacy orchestrator path in a separate cleanup pass.

4. Improve validation.
   - Add or restore frontend linting so visual refactors stay safe.
   - Keep backend unit tests focused on Paper MCP transport and orchestration behavior.

## Recommended Next Work

- Redesign the Paper canvas HTML output prompts first.
- Then simplify the frontend styling system.
- Only remove legacy backend files after confirming no callers still depend on `/api/design`.