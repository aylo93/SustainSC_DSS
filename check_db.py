import sqlite3
from pathlib import Path

db_path = Path("sustainsc.db")
print("DB exists:", db_path.exists())
print("DB path :", db_path.resolve())

con = sqlite3.connect(str(db_path))
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = cur.fetchall()
print("Tables:", tables)

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sc_scenario';")
print("Has sc_scenario?:", cur.fetchone())

con.close()
