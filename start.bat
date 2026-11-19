@echo off
echo ========================================================
echo   Starting VocalAlign AI (Backend + React Frontend)
echo ========================================================

start "VocalAlign Backend (FastAPI)" cmd /k ".venv\Scripts\python.exe server.py"
start "VocalAlign Frontend (React + Vite)" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting up!
echo Frontend UI : http://localhost:5173
echo Backend API: http://localhost:8000/docs
echo ========================================================
