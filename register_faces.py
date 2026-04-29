"""
register_faces.py — CLI utility for the Smart Attendance System
================================================================
Usage:
    python register_faces.py register   # add a person via webcam
    python register_faces.py list       # list all registered people
    python register_faces.py report     # print today's attendance

No web server required. Run this standalone to manage the face database.
"""

import cv2, sys, os
from pathlib import Path

KNOWN_FACES_DIR = Path("known_faces")
ATTENDANCE_DIR  = Path("attendance_logs")
KNOWN_FACES_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)


def register():
    name = input("Enter full name: ").strip()
    if not name:
        print("Name cannot be empty."); return

    cap = cv2.VideoCapture(0)
    print("📷 Camera open. Press SPACE to capture, Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        cv2.imshow("Register — Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            safe = name.replace(" ","_").lower()
            path = KNOWN_FACES_DIR / f"{safe}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"✅ Saved face for '{name}' → {path}")
            break
        elif key == ord('q'):
            print("Cancelled.")
            break
    cap.release()
    cv2.destroyAllWindows()


def list_people():
    imgs = list(KNOWN_FACES_DIR.glob("*.[jp][pn]g"))
    if not imgs:
        print("No people registered yet."); return
    print(f"\n{'#':<5} {'Name':<30} {'File'}")
    print("─" * 60)
    for i, p in enumerate(imgs, 1):
        name = p.stem.replace("_", " ").title()
        print(f"{i:<5} {name:<30} {p.name}")
    print(f"\nTotal: {len(imgs)}")


def report():
    from datetime import date
    import pandas as pd
    csv_path = ATTENDANCE_DIR / f"attendance_{date.today().isoformat()}.csv"
    if not csv_path.exists():
        print("No attendance recorded today."); return
    df = pd.read_csv(str(csv_path))
    print(f"\nAttendance for {date.today()} ({len(df)} records)")
    print("─" * 50)
    print(df.to_string(index=False))


CMDS = {"register": register, "list": list_people, "report": report}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in CMDS:
        CMDS[cmd]()
    else:
        print("Usage: python register_faces.py [register | list | report]")