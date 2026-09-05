import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.storage import create_storage
import pandas as pd
import io

storage = create_storage('')
output = io.BytesIO()

tables_to_export = [
    "participants",
    "sessions",
    "answers",
    "multimodal_tracking",
    "eye_tracking",
    "facial_expression",
    "question_tracking"
]

try:
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for table in tables_to_export:
            print(f"Exporting {table}...")
            records = storage.list_records(table, 100_000)
            if records:
                df = pd.DataFrame(records)
                sheet_name = table[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name=table[:31], index=False)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
