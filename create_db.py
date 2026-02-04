# create_db.py

from sustainsc.config import engine, Base
import sustainsc.models  # IMPORTANTE: registra todas las clases ORM

def main():
    print("Creating database schema...")
    Base.metadata.create_all(bind=engine)
    print("Done.")

if __name__ == "__main__":
    main()
