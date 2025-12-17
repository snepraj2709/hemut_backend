"""
WebSocket Test Client for Q&A Dashboard

This script tests the WebSocket endpoint for real-time question updates.
Run this alongside the FastAPI server to verify WebSocket functionality.
"""

import asyncio
import websockets
import json
from datetime import datetime

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")
            
            # Wait for initial data
            print("\n📥 Waiting for initial data...")
            initial_message = await websocket.recv()
            initial_data = json.loads(initial_message)
            
            if initial_data.get("type") == "initial_data":
                questions = initial_data.get("data", [])
                print(f"✅ Received {len(questions)} questions")
                for q in questions[:3]:  # Show first 3
                    print(f"   - Q{q['question_id']}: {q['message'][:50]}...")
            
            # Send ping
            print("\n🏓 Sending ping...")
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Listen for messages for 60 seconds
            print("\n👂 Listening for real-time updates (60 seconds)...")
            print("   Try submitting questions or answers in another window!\n")
            
            timeout = 60
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    msg_type = data.get("type")
                    
                    if msg_type == "pong":
                        print(f"🏓 Pong received at {data.get('timestamp')}")
                    
                    elif msg_type == "new_question":
                        q = data.get("data", {})
                        print(f"🆕 NEW QUESTION: Q{q.get('question_id')} - {q.get('message')[:50]}...")
                    
                    elif msg_type == "answer_added":
                        q = data.get("data", {})
                        print(f"💬 ANSWER ADDED to Q{q.get('question_id')}")
                    
                    elif msg_type == "question_status_changed":
                        q = data.get("data", {})
                        print(f"🔄 STATUS CHANGED: Q{q.get('question_id')} -> {q.get('status')}")
                    
                    else:
                        print(f"📨 Received: {msg_type}")
                
                except asyncio.TimeoutError:
                    # Send periodic ping
                    await websocket.send(json.dumps({"type": "ping"}))
                    continue
            
            print("\n✅ Test completed successfully!")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket Test Client for Q&A Dashboard")
    print("=" * 60)
    print("\nMake sure the FastAPI server is running on http://localhost:8000")
    print("You can start it with: python main.py\n")
    
    asyncio.run(test_websocket())
