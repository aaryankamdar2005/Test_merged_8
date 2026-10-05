import sqlite3
con=sqlite3.connect('data/cognitrack.db')
res = con.execute('SELECT COUNT(*) FROM tracking_data WHERE component_type="multimodal_tracking" AND facial_sample_count > 0').fetchall()
print(res)
