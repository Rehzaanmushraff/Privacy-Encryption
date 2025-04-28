from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from . import models, database
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# CORS so Streamlit frontend can access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class MessageCreate(BaseModel):
    original_message: str
    encrypted_message: str

@app.post("/messages/")
def create_message(message: MessageCreate, db: Session = Depends(get_db)):
    db_message = models.Message(
        original_message=message.original_message,
        encrypted_message=message.encrypted_message
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return {"message": "Saved successfully!"}
