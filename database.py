# database.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./qa_dashboard.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

# Models

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    questions = relationship("Question", back_populates="user")
    answers = relationship("Answer", back_populates="user")

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Question(Base):
    __tablename__ = "questions"

    question_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    message = Column(Text, nullable=False)
    status = Column(String(20), default="Pending", index=True)  # Pending, Escalated, Answered
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="questions")
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")

    def to_dict(self, include_answers=True):
        result = {
            "question_id": self.question_id,
            "user_id": self.user_id,
            "message": self.message,
            "status": self.status,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
        if include_answers:
            result["answers"] = [answer.to_dict() for answer in self.answers]
        return result


class Answer(Base):
    __tablename__ = "answers"

    answer_id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.question_id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    question = relationship("Question", back_populates="answers")
    user = relationship("User", back_populates="answers")

    def to_dict(self):
        return {
            "answer_id": self.answer_id,
            "question_id": self.question_id,
            "user_id": self.user_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class AISuggestion(Base):
    __tablename__ = "ai_suggestions"

    suggestion_id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.question_id"), nullable=False)
    suggestion = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "suggestion_id": self.suggestion_id,
            "question_id": self.question_id,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize default admin user
def create_default_admin():
    import bcrypt
    db = SessionLocal()
    try:
        # Check if admin exists
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            hashed_password = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            admin = User(
                username="admin",
                email="admin@example.com",
                password=hashed_password.decode('utf-8'),
                is_admin=True
            )
            db.add(admin)
            db.commit()
            print("✅ Default admin user created: username=admin, password=admin123")
        else:
            print("ℹ️  Admin user already exists")
    finally:
        db.close()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    create_default_admin()
    print("Database setup complete!")