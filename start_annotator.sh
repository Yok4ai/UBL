#!/bin/bash

# Start Grounding DINO Annotator
# This script starts both the backend and frontend servers

echo "Starting Grounding DINO Annotator..."

# Check if we're in the correct directory
if [ ! -d "annotator_backend" ] || [ ! -d "annotator" ]; then
    echo "Error: Please run this script from /home/mkultra/Documents/UBL/"
    exit 1
fi

# Start backend in background
echo "Starting backend server..."
cd annotator_backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting frontend server..."
cd annotator
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "Servers started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
