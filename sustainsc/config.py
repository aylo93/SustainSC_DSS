# sustainsc/config.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# En Streamlit Cloud el FS del repo puede ser read-only.
# /tmp sí es escribible.
DEFAULT_DB = "sqlite:////tmp/sustainsc.db"

SQLALCHEMY_DATABASE_URL = os.getenv("SUSTAINSC_DB_URL", DEFAULT_DB)

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
