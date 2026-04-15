# Vibeframe Move-to-D Drive Plan

## Goal
Open the project from D drive in a new VS Code window and run backend + frontend successfully.

## 1. Open Project in New VS Code Window
1. In current VS Code, open Command Palette.
2. Run: File: New Window.
3. In new window, open folder:
   - /mnt/d/DevHub/Projects/vibeframe (if using WSL workspace)
   - or D:\DevHub\Projects\vibeframe (if using native Windows workspace)
4. Confirm you can see: backend, frontend, README.md, plan.md.

## 2. Pick Runtime Mode (Recommended: Windows Native)
1. Recommended for Paper MCP localhost reliability:
   - Open project as native Windows folder (D:\...)
   - Run backend with Windows Python.
2. Alternative:
   - Keep running in WSL with mirrored networking enabled.

## 3. Backend Setup
1. Go to backend folder.
2. Create virtual environment.
3. Install dependencies from requirements.txt.
4. Ensure .env exists and has:
   - GROQ_API_KEY
   - GROQ_MODEL=llama-3.3-70b-versatile
   - PAPER_MCP_URL=http://127.0.0.1:29979/mcp
   - PAPER_MCP_TIMEOUT_SECONDS=20
   - APP_ENV=development
   - APP_PORT=8000
5. Start backend:
   - uvicorn main:app --reload --host 0.0.0.0 --port 8000

## 4. Frontend Setup
1. Go to frontend folder.
2. Run npm install.
3. Ensure .env.local exists with:
   - NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
4. Start frontend:
   - npm run dev

## 5. Dependency Checks
1. Open browser:
   - http://127.0.0.1:8000/health
   - http://127.0.0.1:8000/health/dependencies
2. Expected:
   - health returns status ok
   - dependencies shows Paper MCP reachable and Groq configured

## 6. Paper MCP Verification
1. Keep Paper Desktop open with a file loaded.
2. If running in WSL and still failing, enable mirrored mode:
   - Add to Windows user .wslconfig:
     [wsl2]
     networkingMode=mirrored
   - Run wsl --shutdown
   - Reopen WSL and restart services
3. Retry /health/dependencies.

## 7. End-to-End Test (Task 1 Definition of Done)
1. Open frontend at http://127.0.0.1:3000.
2. Submit a text prompt for UI generation.
3. Confirm API returns tool trace.
4. Confirm visible elements are created on Paper canvas.

## 8. Security Cleanup
1. Rotate Groq API key if it was shared in chat/screenshots.
2. Keep backend/.env out of git.

## 9. Optional Cleanup After Verification
1. Once D-drive setup is confirmed, archive or remove old source folder:
   - /home/zakinabdul/devhub/projects/vibeframe
2. Do this only after a successful end-to-end run from D drive.
