# create_db.py
from sustainsc.config import engine
from sustainsc.models import Base

def main():
    Base.metadata.create_all(bind=engine)
    print("DB schema ensured (tables created if missing).")

if __name__ == "__main__":
    main()
