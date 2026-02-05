import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DEFAULT_DB_URL = "sqlite:////tmp/sustainsc.db"
SQLALCHEMY_DATABASE_URL = os.getenv("SUSTAINSC_DB_URL", DEFAULT_DB_URL)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
