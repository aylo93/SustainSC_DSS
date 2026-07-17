# create_db.py
import argparse

from sustainsc.config import Base, engine
import sustainsc.models  # Register every table with SQLAlchemy.


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or reset the SustainSC DSS database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all application data and recreate an empty database schema.",
    )
    args = parser.parse_args()

    if args.reset:
        Base.metadata.drop_all(bind=engine)
        print("Existing application data deleted.")

    Base.metadata.create_all(bind=engine)
    print("Empty database schema ready." if args.reset else "Database schema ready.")

if __name__ == "__main__":
    main()
