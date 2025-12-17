# main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict
from datetime import datetime
import jwt
import bcrypt
import asyncio
import httpx
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import desc
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import from database.py
from database import engine, SessionLocal, init_db, create_default_admin, User, Question, Answer, get_db

# Load environment variables
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://webhook.site/mock-endpoint")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, Optional[int]] = {}  # WebSocket -> user_id mapping

    async def connect(self, websocket: WebSocket, user_id: Optional[int] = None):
        await websocket.accept()
        self.active_connections[websocket] = user_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections.keys():
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Pydantic Models
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

    @validator('username')
    def username_validator(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v.strip()

    @validator('password')
    def password_validator(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class QuestionSubmit(BaseModel):
    message: str

    @validator('message')
    def message_validator(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be blank')
        return v.strip()

class AnswerSubmit(BaseModel):
    message: str

    @validator('message')
    def message_validator(cls, v):
        if not v.strip():
            raise ValueError('Answer cannot be blank')
        return v.strip()

class AnswerResponse(BaseModel):
    answer_id: int
    user_id: Optional[int]
    message: str
    timestamp: Optional[str]

class QuestionResponse(BaseModel):
    question_id: int
    user_id: Optional[int]
    message: str
    status: str
    timestamp: Optional[str]
    answers: List[AnswerResponse] = []

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    is_admin: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and default admin
    print("Initializing database...")
    init_db()
    create_default_admin()
    print("✅ FastAPI server started successfully")
    yield
    # Shutdown
    print("🛑 FastAPI server shutting down")

# Initialize FastAPI
app = FastAPI(
    title="Q&A Dashboard API",
    description="Real-time Q&A platform with WebSocket support (PostgreSQL/SQLite)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middlewareresolve cors 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://qna-dashboard.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Helper Functions
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # hashed_password from DB is string, needs bytes for bcrypt
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict) -> str:
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token_data = decode_access_token(credentials.credentials)
    user_id = token_data.get("user_id")
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

# RAG/Langchain Mock Integration
async def generate_ai_suggestion(message: str) -> str:
    """Get AI-powered answer suggestion using Langchain and OpenAI"""
    
    if not OPENAI_API_KEY:
        # Fallback to mock if no API key
        print("⚠️ OPENAI_API_KEY not found. Using mock response.")
        await asyncio.sleep(1)
        return "AI Suggestion (Mock): Configure OPENAI_API_KEY for real responses. Try checking the documentation."

    try:
        chat = ChatOpenAI(temperature=0.7, api_key=OPENAI_API_KEY)
        messages = [
            SystemMessage(content="You are a helpful expert assistant for a Q&A dashboard. Provide a concise, helpful answer to the user's question."),
            HumanMessage(content=message)
        ]
        response = await chat.ainvoke(messages)
        return response.content
    except Exception as e:
        print(f"❌ OpenAI API Error: {str(e)}")
        return "Sorry, I couldn't generate a suggestion at this time."

# Webhook Functions
async def trigger_webhook(event_type: str, data: dict):
    """Send webhook notification for various events"""
    if not WEBHOOK_URL or WEBHOOK_URL == "https://webhook.site/mock-endpoint":
        print(f"⚠️ Webhook URL not configured, skipping webhook for {event_type}")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "event": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            response = await client.post(WEBHOOK_URL, json=payload, timeout=5.0)
            print(f"✅ Webhook sent to {WEBHOOK_URL} [{event_type}]: {response.status_code}")
    except Exception as e:
        print(f"❌ Webhook failed [{event_type}]: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Q&A Dashboard API (Persistent)",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/auth/*",
            "questions": "/api/questions/*",
            "websocket": "/ws"
        }
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        # Check DB connection
        users_count = db.query(User).count()
        questions_count = db.query(Question).count()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "users_count": users_count,
            "questions_count": questions_count,
            "database": "connected"
        }
    except Exception as e:
         return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Authentication Endpoints

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this username or email already exists")
    
    # Create new user
    hashed_password = hash_password(user_data.password) # Returns bytes
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password=hashed_password.decode('utf-8'), # Store as string
        is_admin=False
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Generate token
    token = create_access_token({"user_id": new_user.user_id})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(
            user_id=new_user.user_id,
            username=new_user.username,
            email=new_user.email,
            is_admin=new_user.is_admin
        )
    }

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    # Find user
    user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user or not verify_password(credentials.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate token
    token = create_access_token({"user_id": user.user_id})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            is_admin=user.is_admin
        )
    }

@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin
    )

# Question Endpoints

@app.post("/api/questions", response_model=QuestionResponse)
async def submit_question(
    question_data: QuestionSubmit,
    db: Session = Depends(get_db)
):
    """Submit a new question (guests allowed - user is Optional)"""
    user_id = None
    # If using the "Depends(get_current_user) if False" pattern, it meant it was disabled.
    # So we'll just set user_id to None.
    
    new_question = Question(
        user_id=user_id,
        message=question_data.message,
        status="Pending",
        timestamp=datetime.utcnow()
    )
    db.add(new_question)
    db.commit()
    db.refresh(new_question)
    
    # Broadcast to WebSocket clients
    question_data = new_question.to_dict()
    await manager.broadcast({
        "type": "new_question",
        "data": question_data
    })
    
    # Trigger webhook for new question
    asyncio.create_task(trigger_webhook("question_created", question_data))
    
    return question_data

@app.get("/api/questions", response_model=List[QuestionResponse])
async def get_questions(db: Session = Depends(get_db)):
    """Get all questions sorted by escalation status and timestamp"""
    # Sorting: Escalated first (we can use a custom ordering or just Python sort after fetching if dataset is small, or SQL)
    # SQL: ORDER BY CASE WHEN status='Escalated' THEN 0 ELSE 1 END, timestamp DESC
    questions = db.query(Question).all()
    
    # Python sorting to match complex logic (Escalated first)
    questions.sort(key=lambda q: (
        0 if q.status == "Escalated" else 1,
        q.timestamp.timestamp() if q.timestamp else 0 # Sort descending? Original was -timestamp
    ), reverse=False) 
    # Wait, original: key=(priority, -timestamp). 
    # Escalated=0, others=1. Low number first.
    # -timestamp means larger timestamp (newer) has smaller number -> sorts first.
    # So we want Escalated First, Newest First.
    
    questions.sort(key=lambda q: (
        0 if q.status == "Escalated" else 1,
        -q.timestamp.timestamp() if q.timestamp else 0
    ))

    return [q.to_dict(include_answers=True) for q in questions]

@app.get("/api/questions/{question_id}", response_model=QuestionResponse)
async def get_question(question_id: int, db: Session = Depends(get_db)):
    """Get a specific question by ID"""
    question = db.query(Question).filter(Question.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question.to_dict(include_answers=True)

@app.post("/api/questions/{question_id}/answers")
async def add_answer(
    question_id: int,
    answer_data: AnswerSubmit,
    db: Session = Depends(get_db)
    # user: Optional[User] ... (Simlar to question, assuming anonymous allowed or fixed later)
):
    """Add an answer to a question"""
    question = db.query(Question).filter(Question.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    new_answer = Answer(
        question_id=question_id,
        user_id=None, # Anonymous for now
        message=answer_data.message,
        timestamp=datetime.utcnow()
    )
    db.add(new_answer)
    db.commit()
    db.refresh(new_answer)
    
    # Broadcast update
    question_data = question.to_dict(include_answers=True)
    await manager.broadcast({
        "type": "answer_added",
        "data": question_data
    })
    
    # Trigger webhook for new answer
    asyncio.create_task(trigger_webhook("answer_created", {
        "question_id": question_id,
        "answer": new_answer.to_dict(),
        "question": question_data
    }))
    
    return {"message": "Answer added successfully", "answer": new_answer.to_dict()}

@app.put("/api/questions/{question_id}/mark-answered")
async def mark_answered(
    question_id: int, 
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Mark a question as answered (admin only)"""
    question = db.query(Question).filter(Question.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    question.status = "Answered"
    db.commit()
    db.refresh(question)
    
    question_data = question.to_dict(include_answers=True)
    
    # Trigger webhook
    asyncio.create_task(trigger_webhook("question_answered", question_data))
    
    # Broadcast update
    await manager.broadcast({
        "type": "question_status_changed",
        "data": question_data
    })
    
    return {"message": "Question marked as answered", "question": question_data}

@app.put("/api/questions/{question_id}/escalate")
async def escalate_question(
    question_id: int, 
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Escalate a question (admin only)"""
    question = db.query(Question).filter(Question.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    question.status = "Escalated"
    db.commit()
    db.refresh(question)
    
    question_data = question.to_dict(include_answers=True)
    
    # Trigger webhook
    asyncio.create_task(trigger_webhook("question_escalated", question_data))
    
    # Broadcast update
    await manager.broadcast({
        "type": "question_status_changed",
        "data": question_data
    })
    
    return {"message": "Question escalated", "question": question_data}

@app.post("/api/questions/{question_id}/ai-suggest")
async def get_ai_suggestion(question_id: int, db: Session = Depends(get_db)):
    """Get AI-powered answer suggestion using RAG/Langchain"""
    question = db.query(Question).filter(Question.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    suggestion = await generate_ai_suggestion(question.message)
    
    return {
        "question_id": question_id,
        "suggestion": suggestion,
        "timestamp": datetime.now().isoformat()
    }

# WebSocket Endpoint

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket endpoint for real-time updates
    
    Usage:
    - Connect to ws://localhost:8000/ws or ws://localhost:8000/ws?token=YOUR_JWT_TOKEN
    - Optionally send authentication via first message: {"type": "auth", "token": "YOUR_JWT_TOKEN"}
    - Receive real-time updates for questions, answers, and status changes
    - Send ping messages to keep connection alive: {"type": "ping"}
    """
    user_id = None
    
    # Try to authenticate from query parameter
    if token:
        try:
            token_data = decode_access_token(token)
            user_id = token_data.get("user_id")
        except:
            pass
    
    await manager.connect(websocket, user_id)
    
    # Get database session
    db = SessionLocal()
    
    try:
        # Send initial data - all questions
        questions = db.query(Question).all()
        questions.sort(key=lambda q: (
            0 if q.status == "Escalated" else 1,
            -q.timestamp.timestamp() if q.timestamp else 0
        ))
        
        await manager.send_personal_message({
            "type": "initial_data",
            "data": [q.to_dict(include_answers=True) for q in questions]
        }, websocket)
        
        # Keep connection alive and handle client messages
        while True:
            message = await websocket.receive_json()
            
            # Handle authentication message
            if message.get("type") == "auth":
                auth_token = message.get("token")
                if auth_token:
                    try:
                        token_data = decode_access_token(auth_token)
                        user_id = token_data.get("user_id")
                        manager.active_connections[websocket] = user_id
                        await manager.send_personal_message({
                            "type": "auth_success",
                            "user_id": user_id
                        }, websocket)
                    except:
                        await manager.send_personal_message({
                            "type": "auth_error",
                            "message": "Invalid token"
                        }, websocket)
            
            # Handle ping/pong for keepalive
            elif message.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            
            # Handle refresh request
            elif message.get("type") == "refresh":
                questions = db.query(Question).all()
                questions.sort(key=lambda q: (
                    0 if q.status == "Escalated" else 1,
                    -q.timestamp.timestamp() if q.timestamp else 0
                ))
                await manager.send_personal_message({
                    "type": "refresh_data",
                    "data": [q.to_dict(include_answers=True) for q in questions]
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )