import sqlite3
con=sqlite3.connect('data/cognitrack.db')
print(con.execute('SELECT facial_face_detected_ratio, facial_sample_count FROM tracking_data WHERE component_type="multimodal_tracking" ORDER BY id DESC LIMIT 5').fetchall())
