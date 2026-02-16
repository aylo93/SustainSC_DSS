# create_db.py
from sustainsc.config import Base, engine
import sustainsc.models  # registra todas las tablas

def main():
    Base.metadata.create_all(bind=engine)
    print("DB schema ensured (tables created if missing).")

if __name__ == "__main__":
    main()
