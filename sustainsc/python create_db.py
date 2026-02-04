# create_db.py

from sustainsc.config import engine, Base
from sustainsc import models  # noqa: F401  (import to register models with Base)

if __name__ == "__main__":
    print("Creating database schema...")
    Base.metadata.create_all(bind=engine)
    print("Done.")
