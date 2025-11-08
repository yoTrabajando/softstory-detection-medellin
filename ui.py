import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os, json, datetime
from dataset_trch import augmentation
from model_trch import model_highlight
import torch

# Colors
DARK_BG = "#0f172a"
BLUE_BG = "#1e3a8a"
TEXT = "#f1f5f9"

class AnnotationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Tool")
        self.root.configure(bg=DARK_BG)
        self.root.state("zoomed")

        # Language state for instructions
        self.lang = "EN"

        # Start screen
        self.start_frame = tk.Frame(root, bg=BLUE_BG)
        self.start_frame.pack(fill="both", expand=True)

        # Left panel with general instructions
        left_frame = tk.Frame(self.start_frame, bg=BLUE_BG)
        left_frame.pack(side="left", expand=True, fill="both", padx=20, pady=20)

        lbl = tk.Label(left_frame, text="Select a PNG or JPEG image",
                       bg=BLUE_BG, fg=TEXT, font=("Arial", 16))
        lbl.pack(pady=20)

        btn = tk.Button(left_frame, text="⬆📂 Load Image",
                        command=self.load_image,
                        bg="#2563eb", fg="white",
                        font=("Arial", 18, "bold"),
                        relief="flat", padx=40, pady=20)
        btn.pack(pady=10)

        # Instructions (toggle language)
        self.instructions_en = """
🖊️ Polygon → click multiple points and press ENTER to close.
▭ / ◯ Rectangle or circle → click and drag.
✏️ Edit → drag yellow vertices.
❌ Delete → click on any shape.
↩ Undo → removes last point or shape.
💾 Save → exports image and JSON.
🌍 GeoRef → open georeference window.
"""
        self.instructions_es = """
🖊️ Polígono → clic en varios puntos y ENTER para cerrar.
▭ / ◯ Rectángulo o círculo → clic y arrastra.
✏️ Editar → arrastra vértices amarillos.
❌ Borrar → clic sobre cualquier figura.
↩ Undo → elimina último punto o figura.
💾 Guardar → exporta imagen y JSON.
🌍 GeoRef → abre ventana de datos.
"""
        self.instr_label_left = tk.Label(left_frame, text=self.instructions_en,
                                         bg=BLUE_BG, fg=TEXT, font=("Arial", 11),
                                         justify="left", anchor="w")
        self.instr_label_left.pack(padx=10, pady=20, fill="x")

        # Language toggle button
        tk.Button(left_frame, text="🌐", command=self.toggle_language,
                  bg="gray", fg="white", font=("Arial", 12, "bold"),
                  width=4, height=2).pack(pady=5)

        # Right panel with polygon rules
        right_frame = tk.Frame(self.start_frame, bg=BLUE_BG)
        right_frame.pack(side="right", fill="y", padx=20, pady=20)

        self.rules_en = """
📐 Drawing Instructions:

1️⃣ First draw the openings of the building 
   (doors and windows) using polygons.

2️⃣ Once all openings are drawn, 
   proceed to draw the balcony polygon.

⚠️ Important:
The balcony polygon must surround the window polygons, 
never be drawn on top of them.
"""
        self.rules_es = """
📐 Instrucciones de dibujo:

1️⃣ Dibuja primero las aberturas de la vivienda 
   (puertas y ventanas) usando polígonos.

2️⃣ Una vez dibujadas todas las aberturas, 
   procede a dibujar el polígono del balcón.

⚠️ Importante:
El polígono del balcón debe rodear los polígonos 
de las ventanas, nunca dibujarse por encima.
"""
        self.instr_label_right = tk.Label(right_frame, text=self.rules_en,
                                          bg=BLUE_BG, fg="white", font=("Arial", 11, "bold"),
                                          justify="left", anchor="w")
        self.instr_label_right.pack(padx=10, pady=10, fill="x")

        # Language toggle button for right panel
        tk.Button(right_frame, text="🌐", command=self.toggle_language,
                  bg="gray", fg="white", font=("Arial", 12, "bold"),
                  width=4, height=2).pack(pady=5)

        # Footer text
        tk.Label(right_frame, text="UNIVERSIDAD EIA",
                 bg=BLUE_BG, fg="white",
                 font=("Times New Roman", 20, "bold italic")).pack(side="bottom", pady=15)

        # Internal variables
        self.name = None
        self.image = None
        self.original_image = None
        self.tk_image = None
        self.tool = None
        self.current_points = []
        self.start_point = None
        self.shapes = []
        self.opening_count = 0
        self.balcony_count = 0
        self.preview_line = None
        self.image_dir = None
        self.editing_index = None
        self.hover_vertex = None
        self.dragging_vertex = None
        self.image_offset = (0, 0)
        self.image_scale = 1.0

        # Neural network variables
        self.image_trch = None
        self.highlight_trch = None
        self.model = model_highlight(
            channels_in = 5,
            out_conv = 1024*5*5,
            hidden_layers = [128, 64],
            probability = 0.087849927
            )
        self.model.load_state_dict(torch.load("saved_models/best_model.pth"))
        self.model.eval()

    # Toggle instructions language
    def toggle_language(self):
        if self.lang == "EN":
            self.lang = "ES"
            self.instr_label_left.config(text=self.instructions_es)
            self.instr_label_right.config(text=self.rules_es)
        else:
            self.lang = "EN"
            self.instr_label_left.config(text=self.instructions_en)
            self.instr_label_right.config(text=self.rules_en)

    # Utility: convert canvas points to image coordinates
    def canvas_to_image_points(self, pts):
        ix0, iy0 = self.image_offset
        s = self.image_scale if self.image_scale != 0 else 1.0
        return [((x - ix0) / s, (y - iy0) / s) for (x, y) in pts]

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg")]
        )
        if not path:
            return
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.image = Image.open(path).convert("RGBA")
        self.original_image = self.image.copy()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_dir = os.path.dirname(path)
        self.start_frame.destroy()
        self.build_main_ui()

    def build_main_ui(self):
        # Left toolbar
        toolbar = tk.Frame(self.root, bg=BLUE_BG, width=90)
        toolbar.pack(side="left", fill="y")

        self.add_toolbar_button(toolbar, "🖊️", "Polygon", lambda: self.set_tool("polygon"))
        self.add_toolbar_button(toolbar, "▭", "Rectangle", lambda: self.set_tool("rectangle"))
        self.add_toolbar_button(toolbar, "◯", "Circle", lambda: self.set_tool("ellipse"))
        self.add_toolbar_button(toolbar, "✏️", "Edit", lambda: self.set_tool("edit"))
        self.add_toolbar_button(toolbar, "❌", "Delete", lambda: self.set_tool("delete"))
        self.add_toolbar_button(toolbar, "↩", "Undo", self.undo)
        self.add_toolbar_button(toolbar, "💾", "Save", self.save_all)
        self.add_toolbar_button(toolbar, "🌍", "GeoRef", lambda: GeoReferenceWindow(self.root, self))
        self.add_toolbar_button(toolbar, "🏠", "Detect", lambda: self.neural_network_prediction())

        # Central canvas
        self.canvas = tk.Canvas(self.root, bg=DARK_BG, highlightthickness=0)
        self.canvas.pack(side="left", expand=True, fill="both", padx=5, pady=5)

        self.render_fit_image()
        self.canvas.bind("<Configure>", lambda e: self.render_fit_image())

        # Events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)
        self.root.bind("<Return>", lambda e: self.close_polygon())

        # Right panel
        right_panel = tk.Frame(self.root, bg=BLUE_BG, width=160)
        right_panel.pack(side="right", fill="y")

        tk.Label(right_panel, text="Geo", bg=BLUE_BG, fg=TEXT, font=("Arial", 9, "bold")).pack(pady=2)
        self.geo_text = tk.Text(right_panel, height=10, bg=DARK_BG, fg=TEXT, font=("Arial", 9))
        self.geo_text.pack(fill="x", padx=2)

        tk.Label(right_panel, text="Polygons", bg=BLUE_BG, fg=TEXT, font=("Arial", 9, "bold")).pack(pady=2)
        # shorter polygon list (half height)
        self.poly_list = tk.Listbox(right_panel, bg=DARK_BG, fg=TEXT, height=5, font=("Arial", 9))
        self.poly_list.pack(fill="x", padx=2, pady=2)

        # New Soft Story Detection window
        tk.Label(right_panel, text="Soft Story Detection", bg=BLUE_BG, fg="yellow",
                 font=("Arial", 10, "bold")).pack(pady=4)
        self.soft_text = tk.Listbox(right_panel, height=6, bg=DARK_BG, fg="orange", font=("Arial", 9))
        self.soft_text.pack(fill="x", padx=2, pady=2)

        tk.Label(right_panel, text="Status", bg=BLUE_BG, fg=TEXT, font=("Arial", 9, "bold")).pack(pady=2)
        self.status_label = tk.Label(right_panel, text="Ready.", bg=DARK_BG, fg="lime", font=("Arial", 9))
        self.status_label.pack(fill="x", padx=2, pady=2)

        self.save_msg = tk.Label(right_panel, text="", bg=BLUE_BG, fg="yellow", font=("Arial", 9, "bold"))
        self.save_msg.pack(pady=4)

        self.set_tool("polygon")

    def render_fit_image(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1 or self.original_image is None:
            return
        iw, ih = self.original_image.width, self.original_image.height
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = self.original_image.resize((nw, nh))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)
        self.image_offset = (x, y)
        self.image_scale = scale

    def add_toolbar_button(self, parent, icon, text, command):
        btn = tk.Button(parent, text=f"{icon}\n{text}", command=command,
                        bg=BLUE_BG, fg=TEXT, relief="flat",
                        font=("Arial", 9), width=8, height=3)
        btn.pack(pady=4, fill="x")

    def set_tool(self, tool):
        self.tool = tool
        cursor_map = {"polygon": "cross", "rectangle": "tcross", "ellipse": "circle",
                      "edit": "hand2", "delete": "X_cursor"}
        self.canvas.config(cursor=cursor_map.get(tool, "arrow"))
        self.set_status(f"Tool: {tool}")

    def on_click(self, event):
        if self.tool == "edit" and self.hover_vertex:
            self.dragging_vertex = self.hover_vertex
        elif self.tool == "polygon":
            self.current_points.append((event.x, event.y))
            r = 3
            self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r,
                                    fill="yellow", outline="")
            if len(self.current_points) > 1:
                x0, y0 = self.current_points[-2]
                self.canvas.create_line(x0, y0, event.x, event.y, fill="cyan")
        elif self.tool in ("rectangle", "ellipse"):
            self.start_point = (event.x, event.y)
            self.canvas.delete("preview")
        elif self.tool == "delete":
            self.delete_polygon_at(event.x, event.y)

    def on_motion(self, event):
        if self.tool == "polygon" and self.current_points:
            self.canvas.delete("preview_line")
            x0, y0 = self.current_points[-1]
            self.preview_line = self.canvas.create_line(x0, y0, event.x, event.y,
                                                        fill="orange", dash=(4, 2),
                                                        tags="preview_line")
        elif self.tool == "edit":
            self.canvas.delete("highlight")
            self.hover_vertex = None
            for idx, shape in enumerate(self.shapes):
                if shape["shape"] == "polygon":
                    for vi, (vx, vy) in enumerate(shape["points"]):
                        if abs(event.x - vx) < 6 and abs(event.y - vy) < 6:
                            r = 6
                            self.canvas.create_oval(vx-r, vy-r, vx+r, vy+r,
                                                    fill="blue", outline="white",
                                                    tags="highlight")
                            self.hover_vertex = (idx, vi)
                            return
            self.hover_vertex = None

    def on_drag(self, event):
        if self.tool == "edit" and self.dragging_vertex:
            idx, vi = self.dragging_vertex
            self.shapes[idx]["points"][vi] = (event.x, event.y)
            self.redraw()
        elif self.tool in ("rectangle", "ellipse") and self.start_point:
            self.canvas.delete("preview")
            x0, y0 = self.start_point
            if self.tool == "rectangle":
                self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                             outline="orange", width=2, tags="preview")
            elif self.tool == "ellipse":
                self.canvas.create_oval(x0, y0, event.x, event.y,
                                        outline="orange", width=2, tags="preview")

    def on_release(self, event):
        if self.tool == "edit" and self.dragging_vertex:
            self.dragging_vertex = None
            self.hover_vertex = None
        elif self.tool in ("rectangle", "ellipse") and self.start_point:
            pts = [self.start_point, (event.x, event.y)]
            self.start_point = None
            self.canvas.delete("preview")
            self.show_type_selector(event.x_root, event.y_root, pts, self.tool)

    def close_polygon(self):
        if self.tool == "polygon" and len(self.current_points) >= 3:
            pts = self.current_points.copy()
            self.current_points = []
            self.canvas.delete("preview_line")
            x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
            self.show_type_selector(x, y, pts, "polygon")

    def show_type_selector(self, x, y, pts, shape):
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-alpha", 0.9)
        win.configure(bg=DARK_BG)

        screen_h = self.root.winfo_screenheight()
        win_h = 60
        if y + win_h > screen_h:
            y = y - win_h - 20

        win.geometry(f"+{x}+{y}")

        tk.Button(win, text="Opening", bg="red", fg="white",
                  font=("Arial", 10, "bold"),
                  width=10, height=2,
                  command=lambda: (self.add_polygon("opening", pts, shape), win.destroy())
                  ).pack(side="left", padx=6, pady=6)

        tk.Button(win, text="Balcony", bg="green", fg="white",
                  font=("Arial", 10, "bold"),
                  width=10, height=2,
                  command=lambda: (self.add_polygon("balcony", pts, shape), win.destroy())
                  ).pack(side="left", padx=6, pady=6)

    def add_polygon(self, tipo, pts, shape):
        if tipo == "opening":
            self.opening_count += 1
            name = f"Opening {self.opening_count}"
            color = "red"
        else:
            self.balcony_count += 1
            name = f"Balcony {self.balcony_count}"
            color = "green"

        self.shapes.append({"type": tipo, "shape": shape, "points": pts})
        self.poly_list.insert("end", name)

        if shape == "rectangle":
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         outline=color, fill=color, stipple="gray50")
        elif shape == "ellipse":
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            self.canvas.create_oval(x0, y0, x1, y1,
                                    outline=color, fill=color, stipple="gray50")
        elif shape == "polygon":
            self.canvas.create_polygon(pts, outline=color, fill=color, stipple="gray50")

        self.set_status(f"{name} added.")

    def set_status(self, msg, color="lime"):
        self.status_label.config(text=msg, fg=color)

    def undo(self):
        if self.current_points:
            self.current_points.pop()
            self.canvas.delete("preview_line")
            self.set_status("Last point removed.", color="yellow")
        elif self.shapes:
            self.shapes.pop()
            self.poly_list.delete("end")
            self.redraw()
            self.set_status("Last shape removed.", color="yellow")

    def redraw(self):
        self.canvas.delete("all")
        self.render_fit_image()
        for shape in self.shapes:
            pts = shape["points"]
            color = "red" if shape["type"] == "opening" else "green"
            if shape["shape"] == "rectangle":
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             outline=color, fill=color, stipple="gray50")
            elif shape["shape"] == "ellipse":
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                self.canvas.create_oval(x0, y0, x1, y1,
                                        outline=color, fill=color, stipple="gray50")
            elif shape["shape"] == "polygon":
                self.canvas.create_polygon(pts, outline=color, fill=color, stipple="gray50")
                if self.tool == "edit":
                    for vi, (vx, vy) in enumerate(pts):
                        r = 4
                        self.canvas.create_oval(vx-r, vy-r, vx+r, vy+r,
                                                fill="yellow", outline="black",
                                                tags=(f"vertex_{vi}", "vertex"))

    def delete_polygon_at(self, x, y):
        for i, shape in enumerate(self.shapes):
            pts = shape["points"]
            if shape["shape"] == "polygon":
                if self.point_in_polygon(x, y, pts):
                    self.shapes.pop(i)
                    self.poly_list.delete(i)
                    self.redraw()
                    self.set_status("Polygon deleted.", color="red")
                    return
            elif shape["shape"] == "rectangle":
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.shapes.pop(i)
                    self.poly_list.delete(i)
                    self.redraw()
                    self.set_status("Rectangle deleted.", color="red")
                    return
            elif shape["shape"] == "ellipse":
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                rx = abs(x1 - x0) / 2
                ry = abs(y1 - y0) / 2
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                if ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2) <= 1:
                    self.shapes.pop(i)
                    self.poly_list.delete(i)
                    self.redraw()
                    self.set_status("Circle deleted.", color="red")
                    return

    def select_polygon_at(self, x, y):
        for i, shape in enumerate(self.shapes):
            if shape["shape"] == "polygon":
                if self.point_in_polygon(x, y, shape["points"]):
                    self.editing_index = i
                    self.set_status(f"Editing {shape['type']}", color="yellow")
                    return

    def point_in_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def layer_image(self):
        if not self.image_dir:
            self.set_status("No image loaded.", color="red")
            return

        layer_img = Image.new('RGBA', self.original_image.size, color = (0, 0, 0))
        draw = ImageDraw.Draw(layer_img, "RGBA")

        export_data = []
        for shape in self.shapes:
            pts_img = self.canvas_to_image_points(shape["points"])
            color = (255, 0, 0, 120) if shape["type"] == "opening" else (0, 255, 0, 120)

            if shape["shape"] == "rectangle":
                draw.rectangle([pts_img[0], pts_img[1]], outline=color, fill=color)
            elif shape["shape"] == "ellipse":
                draw.ellipse([pts_img[0], pts_img[1]], outline=color, fill=color)
            elif shape["shape"] == "polygon":
                draw.polygon(pts_img, outline=color, fill=color)

            export_data.append({
                "type": shape["type"],
                "shape": shape["shape"],
                "points_canvas": shape["points"],
                "points_image": pts_img
            })
        self.highlight_trch = layer_img

    def save_all(self):
        self.layer_image()
        layer_img = self.highlight_trch
        layer_path = os.path.join(self.image_dir, f"{self.name}_lay.png")
        layer_img.save(layer_path)

        self.save_msg.config(text="Files saved successfully", fg="yellow")
        self.set_status("Save successful.", color="lime")

    def neural_network_prediction(self):
        self.layer_image()
        main_image = augmentation(
            self.original_image.convert('RGB'), 
            rand_vec = [],
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5])
        
        highlight_image = augmentation(
            self.highlight_trch.convert('RGB'),
            rand_vec = [],
            mean = [0.5, 0.5, 0.],
            std = [0.5, 0.5, 1.])
        tensor = torch.unsqueeze(torch.cat((main_image, highlight_image[0:2]), dim = 0), dim = 0)
        
        value_tensor = self.model(tensor).item()
        if value_tensor < 0.5:
            result = "Non soft story"
            prob = 1-value_tensor
        else:
            result = "Soft story"
            prob = value_tensor

        self.soft_text.insert("end", f"{result}, probability: {100*prob:.2f}%")

        


class GeoReferenceWindow(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Georeferenced Information")
        self.configure(bg=BLUE_BG)
        self.geometry("400x550")  # taller window for georeference

        # Variables
        self.country_var = tk.StringVar()
        self.dept_var = tk.StringVar()
        self.city_var = tk.StringVar()
        self.address_var = tk.StringVar()
        self.floors_var = tk.StringVar()
        self.use_var = tk.StringVar(value="Residential")
        self.year_var = tk.StringVar()
        self.material_var = tk.StringVar(value="Concrete")

        # Fields
        self.add_field("Country:", self.country_var)
        self.add_field("Department/State:", self.dept_var)
        self.add_field("City:", self.city_var)
        self.add_field("Address:", self.address_var)
        self.add_field("Number of floors:", self.floors_var)

        # Dropdown for Use of Building
        frame_use = tk.Frame(self, bg=BLUE_BG)
        frame_use.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_use, text="Use of Building:", bg=BLUE_BG, fg="white").pack(anchor="w")
        uses = ["Residential", "Commercial", "Mixed", "Industrial", "Educational"]
        tk.OptionMenu(frame_use, self.use_var, *uses).pack(fill="x")

        self.add_field("Year of Construction:", self.year_var)

        # Dropdown for Materials
        frame_mat = tk.Frame(self, bg=BLUE_BG)
        frame_mat.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_mat, text="Main Material:", bg=BLUE_BG, fg="white").pack(anchor="w")
        materials = ["Concrete", "Brick", "Wood", "Steel", "Adobe", "Mixed"]
        tk.OptionMenu(frame_mat, self.material_var, *materials).pack(fill="x")

        # Save button
        tk.Button(self, text="Save", command=self.save_info,
                  bg="green", fg="white", font=("Arial", 10, "bold")).pack(pady=20)

    def add_field(self, label_text, variable):
        frame = tk.Frame(self, bg=BLUE_BG)
        frame.pack(fill="x", padx=10, pady=5)
        tk.Label(frame, text=label_text, bg=BLUE_BG, fg="white").pack(anchor="w")
        tk.Entry(frame, textvariable=variable).pack(fill="x")

    def save_info(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        info = f"""
Georeferenced Information
----------------------------
Country: {self.country_var.get()}
Department/State: {self.dept_var.get()}
City: {self.city_var.get()}
Address: {self.address_var.get()}
Number of floors: {self.floors_var.get()}
Use of Building: {self.use_var.get()}
Year of Construction: {self.year_var.get()}
Main Material: {self.material_var.get()}
Date and Time of Record: {now}
"""

        # Save to file
        if self.app.image_dir:
            txt_path = os.path.join(self.app.image_dir, "georeferenced_info.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(info.strip())

        # Show in right panel
        self.app.geo_text.delete("1.0", "end")
        self.app.geo_text.insert("end", info.strip())

        messagebox.showinfo("Saved", "Georeferenced information saved successfully")
        self.destroy()

# ---- Main Execution ----
if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationUI(root)
    root.mainloop()