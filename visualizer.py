import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / "data/model_dataset.csv"
OVERLAYS = {"None": "image_file", "SAM": "sam_image", "Depth": "depth_image"}
IMAGE_SIZE = (760, 650)


class Visualizer:
    def __init__(self, window):
        self.window = window
        self.data = pd.read_csv(CSV_FILE)
        if self.data.empty:
            raise ValueError(f"No rows found in {CSV_FILE}")
        self.index = 0
        self.photo = None
        self.columns = [
            column for column in self.data
            if pd.api.types.is_numeric_dtype(self.data[column]) or "timestamp" in column
        ]

        window.title("PhenoCam Model Visualizer")
        window.geometry("1450x820")
        self.make_controls()
        self.make_content()
        window.bind("<Left>", lambda _: self.move(-1))
        window.bind("<Right>", lambda _: self.move(1))
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def make_controls(self):
        controls = ttk.Frame(self.window, padding=8)
        controls.pack(fill="x")
        self.previous = ttk.Button(controls, text="◀ Previous", command=lambda: self.move(-1))
        self.previous.pack(side="left")
        self.next = ttk.Button(controls, text="Next ▶", command=lambda: self.move(1))
        self.next.pack(side="left", padx=6)
        self.position = ttk.Label(controls)
        self.position.pack(side="left", padx=12)
        ttk.Label(controls, text="Overlay:").pack(side="left", padx=(20, 4))
        self.overlay = tk.StringVar(value="None")
        box = ttk.Combobox(controls, textvariable=self.overlay, values=list(OVERLAYS), width=10, state="readonly")
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda _: self.show_image())

    def make_content(self):
        panes = ttk.Panedwindow(self.window, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        image_frame = ttk.Frame(panes)
        graph_frame = ttk.Frame(panes)
        panes.add(image_frame, weight=3)
        panes.add(graph_frame, weight=2)

        self.image_label = ttk.Label(image_frame, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self.details = ttk.Label(image_frame, anchor="center", justify="center")
        self.details.pack(fill="x", pady=6)

        selectors = ttk.Frame(graph_frame)
        selectors.pack(fill="x", pady=(4, 8))
        self.x_name = tk.StringVar(value="sam_river_coverage")
        self.y_name = tk.StringVar(value="discharge")
        ttk.Label(selectors, text="X:").pack(side="left", padx=(4, 2))
        ttk.Button(selectors, text="<", width=2, command=lambda: self.move_x(-1)).pack(side="left")
        x_box = ttk.Combobox(selectors, textvariable=self.x_name, values=self.columns, width=24)
        x_box.pack(side="left", padx=2)
        ttk.Button(selectors, text=">", width=2, command=lambda: self.move_x(1)).pack(side="left", padx=(0, 8))
        ttk.Label(selectors, text="Y:").pack(side="left", padx=(4, 2))
        y_box = ttk.Combobox(selectors, textvariable=self.y_name, values=self.columns, width=24)
        y_box.pack(side="left")
        for box in (x_box, y_box):
            box.bind("<<ComboboxSelected>>", lambda _: self.show_graph())
            box.bind("<Return>", lambda _: self.show_graph())

        self.figure = Figure(figsize=(6, 6), layout="constrained")
        self.axes = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def path_for(self, column):
        path = Path(str(self.data.iloc[self.index][column]))
        return path if path.is_absolute() else ROOT / path

    def show_image(self):
        path = self.path_for(OVERLAYS[self.overlay.get()])
        if not path.exists():
            self.image_label.configure(image="", text=f"Missing image:\n{path}")
            self.photo = None
            return
        with Image.open(path) as image:
            image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image.copy())
        self.image_label.configure(image=self.photo, text="")

    def plot_values(self, column):
        values = self.data[column]
        return pd.to_datetime(values, errors="coerce") if "timestamp" in column else pd.to_numeric(values, errors="coerce")

    def show_graph(self):
        x_column, y_column = self.x_name.get(), self.y_name.get()
        if x_column not in self.columns or y_column not in self.columns:
            messagebox.showerror("Invalid column", "Choose two numeric or timestamp columns.")
            return
        x, y = self.plot_values(x_column), self.plot_values(y_column)
        valid = x.notna() & y.notna()
        self.axes.clear()
        self.axes.scatter(x[valid], y[valid], color="#2878b5", s=45)
        if valid.iloc[self.index]:
            self.axes.scatter([x.iloc[self.index]], [y.iloc[self.index]], color="red", s=100, edgecolor="white")
        self.axes.set(xlabel=x_column, ylabel=y_column, title=f"{y_column} vs. {x_column}")
        self.axes.grid(alpha=.25)
        self.canvas.draw_idle()

    def refresh(self):
        row = self.data.iloc[self.index]
        self.position.configure(text=f"Image {self.index + 1} of {len(self.data)}")
        self.previous.configure(state="disabled" if self.index == 0 else "normal")
        self.next.configure(state="disabled" if self.index == len(self.data) - 1 else "normal")
        self.details.configure(text=f"{row['image_timestamp']}   |   Discharge: {row['discharge']}")
        self.show_image()
        self.show_graph()

    def move(self, amount):
        new_index = min(max(self.index + amount, 0), len(self.data) - 1)
        if new_index != self.index:
            self.index = new_index
            self.refresh()

    def move_x(self, amount):
        try:
            index = self.columns.index(self.x_name.get())
        except ValueError:
            index = 0
        self.x_name.set(self.columns[min(max(index + amount, 0), len(self.columns) - 1)])
        self.show_graph()

    def close(self):
        self.figure.clear()
        self.window.quit()
        self.window.destroy()
        os._exit(0)


def main():
    window = tk.Tk()
    Visualizer(window)
    window.mainloop()


if __name__ == "__main__":
    main()
