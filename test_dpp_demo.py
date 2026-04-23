from __future__ import annotations

from pathlib import Path

from sustainsc.config import SessionLocal
from sustainsc.dpp_service import build_dpp_passport, dpp_passport_to_json


def main():
    session = SessionLocal()
    try:
        passport = build_dpp_passport(session, "BATCH_DEMO_001")
    finally:
        session.close()

    json_text = dpp_passport_to_json(passport)
    print(json_text)

    out_path = Path("dpp_demo_batch_001.json")
    out_path.write_text(json_text, encoding="utf-8")
    print(f"\nPassport saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()