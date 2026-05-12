import os
os.environ['TCL_LIBRARY'] = r'C:\Users\Steamlink\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Steamlink\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle
import csv

# ==============================
# 📂 CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(BASE_DIR, "debug_faces")  # 👈 aangepast!
DB_PATH = os.path.join(BASE_DIR, "face_db.pkl")
CSV_PATH = os.path.join(BASE_DIR, "foto_overzicht.csv")

CATEGORIES = ["gezichten", "eten", "natuur", "stad", "water", "overig"]

# ==============================
# 🧠 DATA LADEN
# ==============================
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

image_files = [f for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

current_index = 0

csv_data = {}

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data[row["bestand"]] = row

# ==============================
# 🖼️ IMAGE LOAD
# ==============================
def load_image():
    global current_index

    if not image_files:
        return

    file = image_files[current_index]
    path = os.path.join(IMAGE_FOLDER, file)

    img = Image.open(path)
    img.thumbnail((800, 600))

    tk_img = ImageTk.PhotoImage(img)

    panel.config(image=tk_img)
    panel.image = tk_img

    label_file.config(text=file)

    load_metadata(file)


# ==============================
# 📊 LOAD METADATA
# ==============================
def load_metadata(file):
    if file not in csv_data:
        return

    row = csv_data[file]

    # categorie
    for i in range(1, 5):
        key = f"categorie_{i}"
        if key in row and row[key]:
            category_var.set(row[key])
            break

    # persoon
    for i in range(1, 5):
        key = f"persoon_{i}"
        if key in row and row[key]:
            person_var.set(row[key])
            break


# ==============================
# 💾 OPSLAAN
# ==============================
def save_changes():
    file = image_files[current_index]

    row = csv_data.get(file, {"bestand": file})

    row["categorie_1"] = category_var.get()
    row["persoon_1"] = person_var.get()

    csv_data[file] = row

    write_csv()

    messagebox.showinfo("Opslaan", "Wijzigingen opgeslagen!")


def write_csv():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["bestand", "persoon_1", "categorie_1"]
        writer.writerow(header)

        for file, row in csv_data.items():
            writer.writerow([
                file,
                row.get("persoon_1", ""),
                row.get("categorie_1", "")
            ])


# ==============================
# 🔁 NAVIGATIE
# ==============================
def next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)
    load_image()


def prev_image():
    global current_index
    current_index = (current_index - 1) % len(image_files)
    load_image()


# ==============================
# 👥 SAMENVOEGEN
# ==============================
def merge_persons():
    p1 = merge_var_1.get()
    p2 = merge_var_2.get()

    if p1 == p2:
        messagebox.showwarning("Fout", "Kies 2 verschillende personen")
        return

    if p1 not in face_db or p2 not in face_db:
        messagebox.showerror("Fout", "Persoon niet gevonden")
        return

    if not messagebox.askyesno("Bevestigen", f"{p2} samenvoegen met {p1}?"):
        return

    face_db[p1]["embeddings"].extend(face_db[p2]["embeddings"])

    import numpy as np
    face_db[p1]["mean"] = np.mean(face_db[p1]["embeddings"], axis=0)

    del face_db[p2]

    with open(DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

    update_person_dropdowns()

    messagebox.showinfo("Succes", "Personen samengevoegd!")


# ==============================
# 🔄 DROPDOWNS
# ==============================
def update_person_dropdowns():
    persons = list(face_db.keys())

    person_menu["menu"].delete(0, "end")
    merge_menu_1["menu"].delete(0, "end")
    merge_menu_2["menu"].delete(0, "end")

    for p in persons:
        person_menu["menu"].add_command(label=p, command=lambda v=p: person_var.set(v))
        merge_menu_1["menu"].add_command(label=p, command=lambda v=p: merge_var_1.set(v))
        merge_menu_2["menu"].add_command(label=p, command=lambda v=p: merge_var_2.set(v))

    if persons:
        person_var.set(persons[0])
        merge_var_1.set(persons[0])
        merge_var_2.set(persons[-1])


# ==============================
# 🖥️ GUI
# ==============================
root = tk.Tk()
root.title("Foto Manager AI")

# afbeelding
panel = tk.Label(root)
panel.pack()

label_file = tk.Label(root, text="")
label_file.pack()

# ==============================
# 🎛️ CONTROLS FRAME
# ==============================
controls_frame = tk.Frame(root)
controls_frame.pack(pady=10)

# categorie
category_var = tk.StringVar()
category_menu = tk.OptionMenu(controls_frame, category_var, *CATEGORIES)
category_menu.grid(row=0, column=0, padx=5)

# persoon
person_var = tk.StringVar()
person_menu = tk.OptionMenu(controls_frame, person_var, "")
person_menu.grid(row=0, column=1, padx=5)

# knoppen naast elkaar
tk.Button(controls_frame, text="⬅ Vorige", command=prev_image).grid(row=0, column=2, padx=5)
tk.Button(controls_frame, text="➡ Volgende", command=next_image).grid(row=0, column=3, padx=5)
tk.Button(controls_frame, text="💾 Opslaan", command=save_changes).grid(row=0, column=4, padx=5)

# ==============================
# 🔗 SAMENVOEGEN FRAME
# ==============================
merge_frame = tk.Frame(root)
merge_frame.pack(pady=10)

tk.Label(merge_frame, text="Samenvoegen:").grid(row=0, column=0, padx=5)

merge_var_1 = tk.StringVar()
merge_var_2 = tk.StringVar()

merge_menu_1 = tk.OptionMenu(merge_frame, merge_var_1, "")
merge_menu_1.grid(row=0, column=1, padx=5)

merge_menu_2 = tk.OptionMenu(merge_frame, merge_var_2, "")
merge_menu_2.grid(row=0, column=2, padx=5)

tk.Button(merge_frame, text="🔗 Samenvoegen", command=merge_persons)\
    .grid(row=0, column=3, padx=10)

# ==============================
# 🚀 START
# ==============================
update_person_dropdowns()
load_image()

root.mainloop()