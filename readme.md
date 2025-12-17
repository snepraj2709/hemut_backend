# Q&A Dashboard - FastAPI Backend

A production-ready FastAPI backend for a real-time Q&A platform. This application features a robust RESTful API, WebSocket support for real-time updates, secure authentication, and **AI-powered answer suggestions** using OpenAI.

## 🌟 Key Features

### Core
- ✅ **Database**: PostgreSQL (via Neon DB) for production, SQLite for local dev.
- ✅ **AI Suggestions**: Integrated OpenAI (via LangChain) to answer questions automatically.
- ✅ **Real-time**: WebSockets for instant updates on questions and answers.
- ✅ **Authentication**: Secure JWT-based auth with bcrypt password hashing.
- ✅ **Admin Control**: Role-based access for escalating and marking questions.

### Tech Stack
- **Framework**: FastAPI
- **Database**: PostgreSQL / SQLite (SQLAlchemy ORM)
- **AI/ML**: LangChain + OpenAI `gpt-3.5-turbo`
- **Deployment**: Docker & Docker Compose (AWS Ready)

---

## 🚀 Quick Start (Local)

### 1. Setup
```bash
# Clone the repo
git clone <repo-url>
cd hemut_backend

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:

```properties
SECRET_KEY=your-secure-secret-key
# Leave commented for local SQLite, or add your Neon DB URL:
# DATABASE_URL=postgresql://user:pass@ep-host.aws.neon.tech/neondb
OPENAI_API_KEY=sk-your-openai-key
WEBHOOK_URL=https://webhook.site/your-id
```

### 3. Run Application
The app will automatically initialize the database on first run.

```bash
uvicorn main:app --reload
```
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Docker Deployment (AWS EC2)

The project is containerized and ready for cloud deployment.

### 1. Build & Run
```bash
# Ensure you have your .env file created first!
docker-compose up -d --build
```

### 2. Deployment Guide
See the detailed **[Deployment Guide](deployment_guide.md)** for step-by-step instructions on deploying this to an AWS EC2 instance.

---

## 📡 API Endpoints

### 🤖 AI
- **Get Suggestion**: `POST /api/questions/{id}/ai-suggest` (Requires OPENAI_API_KEY)

### ❓ Questions
- **List All**: `GET /api/questions`
- **Submit**: `POST /api/questions`
- **Escalate**: `PUT /api/questions/{id}/escalate` (Admin)

### 💬 Answers
- **Add Answer**: `POST /api/questions/{id}/answers`

### 👤 Auth
- **Register**: `POST /api/auth/register`
- **Login**: `POST /api/auth/login`

---

## 📝 Default Admin
- **Username**: `admin`
- **Password**: `admin123`
*(Created automatically on startup)*

## 🧪 Testing
```bash
# Run syntax check
python -c "import main; print('Syntax OK')"
```