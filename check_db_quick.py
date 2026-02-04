import sqlite3
con = sqlite3.connect("sustainsc.db")
cur = con.cursor()

cur.execute("SELECT id, code FROM sc_scenario ORDER BY id")
print("SCENARIOS:", cur.fetchall())

cur.execute("SELECT scenario_id, COUNT(*) FROM sc_measurement GROUP BY scenario_id ORDER BY scenario_id")
print("MEASUREMENTS BY scenario_id:", cur.fetchall())

cur.execute("""
SELECT s.code, COUNT(*)
FROM sc_measurement m
JOIN sc_scenario s ON s.id = m.scenario_id
GROUP BY s.code
ORDER BY s.code
""")
print("MEASUREMENTS BY scenario_code:", cur.fetchall())

con.close()
