# sustainsc/config.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Streamlit Cloud: solo /tmp es escribible
DEFAULT_DB_URL = "sqlite:////tmp/sustainsc.db"
SQLALCHEMY_DATABASE_URL = os.getenv("SUSTAINSC_DB_URL", DEFAULT_DB_URL)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()
