# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import Toplevel
from tkinter.filedialog import askopenfilename
from tkinter import filedialog, messagebox
from tkinter import ttk, messagebox, BooleanVar
from tkinter import Tk, Label, Entry, Button
from tkinter.ttk import Separator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

plt.switch_backend('Agg')
from matplotlib.backend_bases import PickEvent
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.legend_handler import HandlerLine2D
from PIL import ImageTk, Image
import numpy as np
import os
import stat
# from entest_backend import run_predictions  # Import only the required function from the backend script
# from backend_gui import run_pred
#from preprocessing_script_original1 import pre_processing
from preprocessing_script import pre_processing
from datetime import datetime
import pathlib
from pathlib import Path
import sys
import torch
import traceback
import tempfile
import json
import os
from pathlib import Path
#import bcrypt
import os
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from tkinter import ttk
from cryptography.fernet import Fernet
from cryptography.fernet import Fernet


####cat ~/.lock/credentials.json###for printing files in the terminal

# Move the resource_path function definition here
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # Folder PyInstaller uses in bundled mode
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

    # return os.path.join(base_path, relative_path)


# CREDENTIALS_FILE = "credentials.json"
# CREDENTIALS_DIR = os.path.join(os.path.expanduser("~"), ".credentials")
# CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, "credentials.json")
# CREDENTIALS_DIR = Path.home() / ".lock"
CREDENTIALS_DIR = Path.home() / "Documents" / "Battery Work Space" / ".lock"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

# The key must be 32 url‑safe base64‑encoded bytes.
# Generate once with keygen.py (see below) and store it securely.
# FERNET_KEY = os.getenv("CREDENTIALS_KEY")
FERNET_KEY = "_OAVB3tbSw4F9Qs9z8xztxCcdCLizisVtOWaNATyQ9o="
# FERNET_KEY = "_OAVB3tbSw4F9Qs9z8xztxCcdCLizisVtOZaNATyQ7o="
if not FERNET_KEY:
    raise RuntimeError(
        "Environment variable CREDENTIALS_KEY not set. "
        "Generate a key with keygen.py and export it."
    )
fernet = Fernet(FERNET_KEY.encode())


# ------------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------------
def load_credentials() -> dict:
    """
    Load credentials from disk.  If the file does not exist,
    return the default admin credentials.
    """
    if not CREDENTIALS_FILE.exists():
        return {"username": "Thermal2025", "password": "admin123"}

    with open(CREDENTIALS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Decrypt the stored password
    try:
        decrypted = fernet.decrypt(data["password"].encode()).decode()
        data["password"] = decrypted
    except Exception as e:
        # If decryption fails, fall back to the raw value
        # (you might want to raise an error instead)
        print(f"Decryption error: {e}")
        data["password"] = data["password"]

    return data


def save_credentials(credentials: dict) -> None:
    """
    Encrypt the password and write the JSON file.
    """
    # Encrypt the password
    encrypted = fernet.encrypt(credentials["password"].encode()).decode()

    payload = {
        "username": credentials["username"],
        "password": encrypted,
    }

    # Ensure the directory exists
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


# ------------------------------------------------------------------
#  GUI classes
# ------------------------------------------------------------------
class ChangePasswordWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Change Password")
        self.geometry("300x250")
        self.parent = parent

        # Center the window over the parent
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 300) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 250) // 2
        self.geometry(f"+{x}+{y}")

        self.show_password = tk.BooleanVar(value=False)

        # Current Password
        ttk.Label(self, text="Current Password:").grid(row=0, column=0, padx=10, pady=5)
        self.current_password = ttk.Entry(self, show="*")
        self.current_password.grid(row=0, column=1, padx=10, pady=5)

        # New Password
        ttk.Label(self, text="New Password:").grid(row=1, column=0, padx=10, pady=5)
        self.new_password = ttk.Entry(self, show="*")
        self.new_password.grid(row=1, column=1, padx=10, pady=5)

        # Confirm Password
        ttk.Label(self, text="Confirm Password:").grid(row=2, column=0, padx=10, pady=5)
        self.confirm_password = ttk.Entry(self, show="*")
        self.confirm_password.grid(row=2, column=1, padx=10, pady=5)

        self.show_password_checkbox = ttk.Checkbutton(
            self,
            text="Show All Passwords",
            command=self.toggle_all_passwords,
            variable=self.show_password
        )
        self.show_password_checkbox.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        # Buttons
        ttk.Button(self, text="Change", command=self.change_password).grid(row=4, column=0, padx=10, pady=10)
        ttk.Button(self, text="Cancel", command=self.destroy).grid(row=4, column=1, padx=10, pady=10)

    def toggle_all_passwords(self):
        if self.show_password.get():
            show = ""
            text = "Hide All Passwords"
        else:
            show = "*"
            text = "Show All Passwords"

        self.current_password.configure(show=show)
        self.new_password.configure(show=show)
        self.confirm_password.configure(show=show)

        # Update the checkbox text
        self.show_password_checkbox.config(text=text)

    def change_password(self):
        credentials = load_credentials()

        # Validate current password
        if self.current_password.get().strip() != credentials["password"]:
            messagebox.showerror("Error", "Current password is incorrect.")
            return

        # Get new passwords
        new_pwd = self.new_password.get().strip()
        confirm_pwd = self.confirm_password.get().strip()

        # Basic checks
        if not new_pwd or not confirm_pwd:
            messagebox.showerror("Error", "Please fill in all fields.")
            return
        if new_pwd != confirm_pwd:
            messagebox.showerror("Error", "New passwords do not match.")
            return
        if new_pwd == credentials["password"]:
            messagebox.showerror("Error", "New password must be different from current password.")
            return

        # Update and save
        credentials["password"] = new_pwd
        save_credentials(credentials)
        messagebox.showinfo("Success", "Password changed successfully!")
        self.destroy()


class LoginWindow(tk.Toplevel):
    def __init__(self, parent, on_success):
        super().__init__(parent)
        self.title("Login")
        self.geometry("300x200")
        self.parent = parent
        self.on_success = on_success
        self.show_password = tk.BooleanVar(value=False)

        # Username field
        ttk.Label(self, text="Username:").grid(row=0, column=0, padx=10, pady=5)
        self.username = ttk.Entry(self)
        self.username.grid(row=0, column=1, padx=10, pady=5)

        # Password field with show/hide option
        ttk.Label(self, text="Password:").grid(row=1, column=0, padx=10, pady=5)
        self.password = ttk.Entry(self, show="*")
        self.password.grid(row=1, column=1, padx=10, pady=5)

        # Show password checkbox
        ttk.Checkbutton(
            self,
            text="Show Password",
            command=self.toggle_password_visibility,
            variable=self.show_password
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        # Login button
        ttk.Button(self, text="Login", command=self.login).grid(row=3, column=0, columnspan=2, pady=10)

        # Bind Enter key to login function
        self.bind('<Return>', lambda e: self.login())

    def toggle_password_visibility(self):
        show = "" if self.show_password.get() else "*"
        self.password.configure(show=show)

    def login(self):
        credentials = load_credentials()

        # If the file doesn't exist, fall back to the default
        if not CREDENTIALS_FILE.exists():
            expected_user = "Thermal2025"
            expected_pass = "admin123"
        else:
            expected_user = credentials["username"]
            expected_pass = credentials["password"]

        if (self.username.get() == expected_user and
                self.password.get() == expected_pass):
            messagebox.showinfo("Login Success", "Welcome Admin!")
            self.on_success()
            self.destroy()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
            self.destroy()


class VoltageDropApp:
    def __init__(self, *args, **kwargs):
        self.root = root
        self.root.title("THERMAL BATTERY VOLTAGE PREDICTON")
        # Set the window size to a fraction of the screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = int(screen_width * 0.8)  # 80% of the screen width
        height = int(screen_height * 0.8)  # 60% of the screen height
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1000, 400)
        self.style = ttk.Style()
        self.style = ttk.Style()
        self.style.map("Disabled.TEntry",
                       foreground=[('disabled', 'gray')])  # Set text color to gray for disabled entries

        self.style.configure('TFrame', background='white')
        self.style.configure('Header.TLabel', background='white', foreground='white', font=('Helvetica', 4, 'bold'))
        self.table_frame = ttk.Frame(root)
        self.allprofile_data = None
        self.new_input_df = None
        self.logged_in = False
        self.gt_aligned_df = None
        self.predicted_df = None
        self.predicted_temp = None
        self.is_admin = True
        self.gt_flag = False
        self.preprocessed_df = None
        # self.auth()
        # self.btn_lock = tk.Button(self.root, text="Lock")  # Ensure btn_lock is defined here
        self.prediction_frame = ttk.Frame(root)
        self.input_frame = ttk.Frame(self.root)
        self.default_values = {
            "Diameter(mm)": "0",
            "Height(mm)": "0",
            "Env Temp(°C)": "0",
            "Axis": "X",  # Default value for Combobox
            "Initial Voltage(V)": "0",
            "Skin Temp(°C)": "0",
            "Model": "Skin Temp",
            "Company": "VAR",  # Default value for Combobox
            "No.of stacks": "0",  # New parameter default
            "total cells": "0",  # New parameter default
            "cell voltage": "0"  # New parameter default
        }

        # self.current_cells = float(self.inputs["total cells"].get())

        self.create_input_controls(disabled=not self.logged_in)

        self.create_widgets()
        self.setup_layout()
        self.gt_df = None
        self.uploaded_gt = None
        self.current_df = None
        self.rmse_temp = None
        self.temp_df = None
        self.load_df = None
        self.current_project_path = None
        self.current_project_meta_path = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # self.load_images()
        # style = ttk.Style()
        # style.configure("Custom.LabelFrame", font=("Arial", 12))
        self.current_cells = float(self.inputs["total cells"].get())
        self.current_stacks = float(self.inputs["No.of stacks"].get())
        self.current_model = self.inputs["Model"].get()
        # current_height = float(self.inputs["Height(mm)"].get())
        self.prev_cells = None
        self.prev_stacks = None
        self.new_cells = None
        self.prev_model = "EnTest"
        #self.prev_model = None

    def create_widgets(self):
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)

        # Left pane with scrollbar
        self.left_pane = ttk.Frame(self.root, width=400)
        self.left_pane.pack_propagate(False)

        # Create canvas and scrollbar for left pane
        self.left_canvas = tk.Canvas(self.left_pane, highlightthickness=0)
        self.left_scrollbar = ttk.Scrollbar(self.left_pane, orient="vertical", command=self.left_canvas.yview)
        self.scrollable_left_frame = ttk.Frame(self.left_canvas)

        self.scrollable_left_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )

        self.canvas_frame = self.left_canvas.create_window((0, 0), window=self.scrollable_left_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        # Bind mousewheel scrolling
        def _on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Login button frame
        self.login_button_frame = ttk.Frame(self.scrollable_left_frame)

        # Login/Logout buttons
        self.login_button = ttk.Button(
            self.login_button_frame,
            text="Login",
            command=self.open_login_window,
            width=10
        )
        self.logout_button = ttk.Button(
            self.login_button_frame,
            text="Logout",
            command=self.logout,
            width=10,
            state=tk.DISABLED
        )

        self.middle_pane = ttk.PanedWindow(self.main_pane, orient=tk.VERTICAL)
        self.right_pane = ttk.PanedWindow(self.main_pane, orient=tk.VERTICAL)

        # Left Panel - Input Section
        self.input_frame = ttk.LabelFrame(self.scrollable_left_frame, text="INPUT PARAMETERS", padding=(10, 10))
        self.status_label = ttk.Label(self.input_frame, text="", font=('Quicksand Medium', 15), anchor='w')

        self.button_frame = ttk.Frame(self.scrollable_left_frame)
        self.battery_image_label = ttk.Label(self.scrollable_left_frame)

        # Middle Panel - Single Profile Plot
        self.profile_frame = ttk.LabelFrame(self.middle_pane, text="CURRENT PROFILE VISUALIZATION", padding=(10, 5))
        self.plot_frame = ttk.Frame(self.profile_frame)

        # Right Panel - Results
        self.result_frame = ttk.LabelFrame(self.right_pane, text="ANALYSIS RESULTS", padding=(10, 5))
        self.table_frame = ttk.Frame(self.result_frame)
        self.prediction_frame = ttk.Frame(self.result_frame)

    def setup_layout(self):
        self.main_pane.add(self.left_pane, weight=0)
        self.main_pane.add(self.middle_pane, weight=1)
        self.main_pane.add(self.right_pane, weight=2)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Pack canvas and scrollbar in left_pane
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Pack login button frame
        self.login_button_frame.pack(fill=tk.X, pady=5, padx=5)
        self.login_button.pack(pady=5, padx=5)
        self.logout_button.pack(pady=5, padx=5)

        # Pack input frame
        self.input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.status_label.pack(fill=tk.X, padx=5, pady=2)
        self.create_input_controls()

        # Pack button frame
        self.button_frame.pack(fill=tk.X, padx=10, pady=5)
        self.create_buttons()

        # Pack battery image
        self.battery_image_label.pack(fill=tk.X, padx=10, pady=5)

        # Middle pane layout
        self.middle_pane.add(self.profile_frame, weight=1)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Right pane layout
        self.right_pane.add(self.result_frame, weight=0)
        self.table_frame.pack(fill=tk.X, pady=5)
        Separator(self.result_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        self.prediction_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Logo and department name layout
        self.logo_frame = tk.Frame(self.root)
        self.logo_frame.pack(padx=10, pady=10)
        image_path = resource_path(os.path.join("rci_logo.png"))
        self.logo_image = ImageTk.PhotoImage(Image.open(image_path).resize((40, 40)))

        self.left_logo_label = tk.Label(self.logo_frame, image=self.logo_image)
        self.department_name = tk.Label(
            self.logo_frame,
            text="Designed & Developed by DEAIS & DPSS,RCI Team",
            font=("Helvetica", 10, "bold"),
            fg="black"
        )
        self.right_logo_label = tk.Label(self.logo_frame, image=self.logo_image)

        self.logo_frame.grid_columnconfigure(0, weight=1)
        self.logo_frame.grid_columnconfigure(1, weight=5)
        self.logo_frame.grid_columnconfigure(2, weight=1)

        self.left_logo_label.grid(row=0, column=0, sticky="w")
        self.department_name.grid(row=0, column=1, sticky="nsew")
        self.right_logo_label.grid(row=0, column=2, sticky="e")

    def password_button(self):
        # Change Password button
        self.change_password_button = ttk.Button(
            self.login_button_frame,
            text="Change Password",
            command=self.open_change_password_window,
            width=15,  # Match the width of the login and logout buttons
            state=tk.DISABLED  # Initially disabled
        )
        self.change_password_button.pack(pady=5, padx=5)



    def open_login_window(self):
        if not self.logged_in:
            login_window = LoginWindow(self.root, self.on_success_login)

            # Get main window's position and dimensions
            x_main = self.root.winfo_x()
            y_main = self.root.winfo_y()
            width_main = self.root.winfo_width()
            height_main = self.root.winfo_height()

            # Calculate the center coordinates relative to the main window
            login_window_width = 300
            login_window_height = 150
            x_login = x_main + (width_main - login_window_width) // 2
            y_login = y_main + (height_main - login_window_height) // 2

            # Set the position of the login window
            login_window.geometry(f"{login_window_width}x{login_window_height}+{x_login}+{y_login}")

    def create_input_controls(self, disabled=True):
        self.prev_cells = None
        self.prev_stacks = None
        self.inputs = {}
        top_button_frame = ttk.Frame(self.input_frame)
        top_button_frame.pack(fill=tk.X, pady=10)

        # Buttons setup (unchanged)
        self.btn_create = ttk.Button(top_button_frame, text="Create Project", command=self.create_project, width=13)
        self.btn_open = ttk.Button(top_button_frame, text="Open Project", command=self.open_project, width=13)
        self.btn_save = ttk.Button(top_button_frame, text="Save Project", command=self.save_project, width=13)
        #btn_upload = ttk.Button(top_button_frame, text="Upload Profile Data", command=self.upload_allprofile, width=20)

        if not self.logged_in:
            self.btn_create.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED)
        else:
            self.btn_create.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)

        self.btn_open.config(state=tk.NORMAL)  # Always enable Open
        self.btn_create.pack(side=tk.LEFT, ipadx=5)
        self.btn_open.pack(side=tk.LEFT, ipadx=5)
        self.btn_save.pack(side=tk.LEFT, ipadx=5)
        #btn_upload.pack(side=tk.LEFT, ipadx=5)
        # Create a new frame for the Upload Profile Data button and pack it below top_button_frame
        upload_button_frame = ttk.Frame(self.input_frame)
        upload_button_frame.pack(fill=tk.X, pady=10)  # Add some padding between frames

        btn_upload = ttk.Button(upload_button_frame, text="Upload Profile Data", command=self.upload_allprofile, width=20)
        btn_upload.pack(pady=2, padx=3, fill=tk.X) 

        params = [
            "Diameter(mm)", "Height(mm)", "Env Temp(°C)", "Axis",
            "No.of stacks", "total cells", "cell voltage",
            "Initial Voltage(V)", "Skin Temp(°C)", "Company", "Model"
        ]

        state = "readonly" if disabled else "normal"
        model_values = ["Skin Temp", "EnTest"]
        # self.inputs = {}

        # self.inputs = {}

        for param in params:
            frame = ttk.Frame(self.input_frame)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=param, width=18, anchor=tk.W).pack(side=tk.LEFT)

            if param == "Axis":
                combo = ttk.Combobox(frame, values=["X", "Y", "G"], state="disabled" if disabled else "normal")
                default_axis = self.default_values.get(param, "X")
                combo.set(default_axis)
                combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                self.inputs[param] = combo
            elif param == "Company":
                combo = ttk.Combobox(frame, values=["VAR", "HBL"], state="disabled" if disabled else "normal")
                default_company = self.default_values.get(param, "VAR")
                combo.set(default_company)
                combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                self.inputs[param] = combo
            elif param == "Model":
                combo = ttk.Combobox(frame, values=["Skin Temp", "EnTest"], state="disabled" if disabled else "normal")
                default_model = self.default_values.get(param, "EnTest")
                combo.set(default_model)
                combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                self.inputs[param] = combo
            else:
                default_value = self.default_values.get(param, "")
                entry = ttk.Entry(frame, state="normal")  # Temporarily enable the input field to set the default value
                entry.insert(0, default_value)
                if disabled:
                    entry.config(state="disabled", style="Disabled.TEntry")
                else:
                    entry.config(state="normal", style="TEntry")
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                self.inputs[param] = entry

        # Always enable Open button
        self.btn_open.config(state=tk.NORMAL)
        self.btn_create.pack(side=tk.LEFT, ipadx=5)
        self.btn_open.pack(side=tk.LEFT, ipadx=5)

        # Bind updates for "total cells" and "cell voltage" to update "Initial Voltage"
        if ("total cells" in self.inputs) and ("cell voltage" in self.inputs) and (
                "No.of stacks" in self.inputs):  # and ("Height(mm)" in self.inputs):
            self.inputs["total cells"].bind("<KeyRelease>", self.update_initial_voltage)
            self.inputs["cell voltage"].bind("<KeyRelease>", self.update_initial_voltage)
            self.inputs["No.of stacks"].bind("<KeyRelease>", self.update_initial_voltage)
            # Bind the update_height function to <FocusOut> event of "No.of stacks" input field

            # self.inputs["No.of stacks"].bind("<FocusOut>", self.update_height)
            self.inputs["No.of stacks"].bind("<FocusOut>", self.update_cells)
            # self.inputs["total cells"].bind("<FocusOut>", self.update_height)

            # self.inputs["total cells"].bind("<FocusOut>", lambda event: (self.update_height(event),self.update_cells(event)))  #, self.update_initial_voltage(event)))

            # self.inputs["total cells"].bind("<KeyRelease>", self.update_height)
            # self.current_cells = float(self.inputs["total cells"].get())
            self.current_cells = float(self.inputs["total cells"].get())
            self.current_stacks = float(self.inputs["No.of stacks"].get())
            self.current_model = self.inputs["Model"].get()
            # current_height = float(self.inputs["Height(mm)"].get())

    def enable_inputs(self):
        for param, widget in self.inputs.items():
            if isinstance(widget, ttk.Entry):
                widget.config(state="normal")
                current_value = widget.get().strip()
                if not current_value:
                    widget.insert(0, self.default_values[param])
            elif isinstance(widget, ttk.Combobox):
                widget.config(state="readonly")  # Changed to 'readonly' for comboboxes
                if not widget.get():
                    widget.set(self.default_values[param])

    def on_success_login(self):
        self.logged_in = True
        # Enable project-related buttons
        if hasattr(self, 'btn_create'):
            self.btn_create.config(state=tk.NORMAL)
        if hasattr(self, 'btn_open'):
            self.btn_open.config(state=tk.NORMAL)
        if hasattr(self, 'btn_save'):
            self.btn_save.config(state=tk.NORMAL)

        # Unlock input parameters by enabling them
        self.enable_inputs()
        self.password_button()
        self.logout_button.config(state=tk.NORMAL)
        self.change_password_button.config(state=tk.NORMAL)

        # Show Logout and Change Password buttons, hide Login button
        self.logout_button.pack(pady=5, padx=5)
        if hasattr(self, 'change_password_button'):
            self.change_password_button.pack(pady=5, padx=5)
        self.login_button.pack_forget()

        self.status_label.config(text="You are logged in!", foreground='green')

    def logout(self):
        # Disable project-related buttons
        if hasattr(self, 'btn_create'):
            self.btn_create.config(state=tk.DISABLED)
        if hasattr(self, 'btn_open'):
            self.btn_open.config(state=tk.NORMAL)
        if hasattr(self, 'btn_save'):
            self.btn_save.config(state=tk.DISABLED)

        # Lock input parameters
        self.lock_inputs()
        self.disable_inputs()

        # Reset login status
        self.logged_in = False
        self.status_label.config(text="You are logged out!", foreground='red')

        # Hide or remove any welcome messages
        if hasattr(self, 'msg_label'):
            self.msg_label.destroy()

        # Show Login button and hide Logout button
        self.login_button.pack(side=tk.TOP, pady=5)
        self.logout_button.pack_forget()
        if hasattr(self, 'change_password_button'):
            self.change_password_button.pack_forget()

    def open_change_password_window(self):
        change_password_window = ChangePasswordWindow(self.root)
        change_password_window.grab_set()

    def lock_inputs(self):
        for param, widget in self.inputs.items():
            if param == "Initial Voltage(V)":
                widget.config(state="readonly", foreground="gray")
            else:
                if isinstance(widget, ttk.Entry):
                    widget.config(state="readonly", foreground="gray")
                elif isinstance(widget, ttk.Combobox):
                    widget.config(state="disabled")

    def admin_lock_inputs(self):
        # Only admins can call this method
        self.lock_inputs()

    def unlock_parameters(self):
        LoginWindow(self.root, self.enable_inputs)

    def create_project(self):

        try:

            try:

                initial_voltage = float(self.inputs["Initial Voltage(V)"].get() or 5.0)
                diameter = float(self.inputs["Diameter(mm)"].get())
                height = float(self.inputs["Height(mm)"].get())
                env_temp = float(self.inputs["Env Temp(°C)"].get())
                sk_temp = float(self.inputs["Skin Temp(°C)"].get())
                axis_label = self.inputs["Axis"].get()
                company = self.inputs["Company"].get()
                model = self.inputs["Model"].get()
                stacks = self.inputs["No.of stacks"].get()
                cells = self.inputs["total cells"].get()
                cell_voltage = self.inputs["cell voltage"].get()
            except ValueError:
                if not self.logged_in:
                    messagebox.showinfo("Login Required", "Please Log in first.")
                    self.open_login_window()
                else:
                    messagebox.showerror("Login Error", "Invalid username or password")

                # event.widget.delete(0, tk.END)
                return

            project_data = {"Diameter(mm)": [diameter], "Height(mm)": [height], "Env Temp(°C)": [env_temp],
                            "Axis": [axis_label], "No.of stacks": [stacks], "total cells": [cells],
                            "cell voltage": [cell_voltage], "Initial Voltage(V)": [initial_voltage],
                            "Company": [company], "Model": [model], "Skin Temp(°C)":[sk_temp]}
            df_project = pd.DataFrame(project_data)
            # Get model folder name from input or default
            model_folder_name = self.inputs["Model"].get().strip() or "default_model"

            # Define paths
            documents_path = str(Path.home() / "Documents")
            workspace_path = os.path.join(documents_path, "Battery Work Space")
            model = self.inputs["Model"].get()
            if model == "EnTest":
                model_workspace = os.path.join(workspace_path, "Entest_Model_Results")
            if model == "Skin Temp":
                model_workspace = os.path.join(workspace_path, "SkinTemp_Model_Results")
            #model_folder_path = os.path.join(workspace_path, model_folder_name)

            # Create model folder inside workspace
            #os.makedirs(model_folder_path, exist_ok=True)

            project_name = tk.simpledialog.askstring("Create Project", "Enter a name for the new project:")
            if not project_name:
                messagebox.showinfo("Info", "Project creation cancelled.")
                return
            if df_project is not None:
                # Project path inside model folder
                #project_path = os.path.join(model_folder_path, project_name)
                #project_path = os.path.join(workspace_path, project_name)
                project_path = os.path.join(model_workspace, project_name)
                os.makedirs(project_path, exist_ok=True)

                # Save metadata as read-only
                metadata_path = os.path.join(project_path, "metadata.csv")
                df_project.to_csv(metadata_path, index=False)
                os.chmod(metadata_path, 0o444)  # Set to read-only
                # os.chmod(metadata_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

                # self.current_project_path = project_path
                # messagebox.showinfo("Success", f"Project created at: {project_path}")
                # metadata_path = os.path.join(project_path, f"metadata.csv")
                # df_project.to_csv(metadata_path, index=True)

                print(f"Project Created at:{project_path}, metadat saved at:{metadata_path}")
                messagebox.showinfo("Info:", f"Project Created at:{project_path}, metadat saved at:{metadata_path}")
                self.current_project_path = project_path

            else:

                # self.current_project_path = csv_file_path
                print(f"Project Created at:{project_path}")
                messagebox.showinfo("Info:", f"Project Created at:{project_path}")

            self.current_project_path = project_path

        except Exception as e:
            print(f"Failed to create project: {traceback.format_exc()}")
            messagebox.showerror("Error", f"{e}")

    def open_project(self):
        self.prev_cells = None
        self.prev_stacks = None
        folder_path = filedialog.askdirectory(title="Select Project Folder")

        if not folder_path:
            print("No folder selected.")
            return

        metadata_file = os.path.join(folder_path, "metadata.csv")
        if not os.path.exists(metadata_file):
            print("metadata.csv not found in the selected folder.")
            messagebox.showerror("Error", "metadata.csv not found in the selected folder.")
            return

        try:
            # Make the file readable
            os.chmod(metadata_file, stat.S_IREAD)

            df = pd.read_csv(metadata_file)
            inputs_data = {col: df.at[0, col] for col in
                           ["Diameter(mm)", "Height(mm)", "Env Temp(°C)", "Axis", "No.of stacks", "total cells",
                            "cell voltage", "Initial Voltage(V)","Skin Temp(°C)","Company", "Model"]}

            # Enable input fields before loading data
            self.enable_inputs()

            # Populate the input fields with project data
            for key, value in inputs_data.items():
                if isinstance(self.inputs[key], ttk.Entry):
                    self.inputs[key].delete(0, tk.END)
                    self.inputs[key].insert(0, str(value))
                elif isinstance(self.inputs[key], ttk.Combobox):
                    self.inputs[key].set(str(value))

            # Check login status to determine if inputs should be disabled
            if not self.logged_in:
                self.disable_inputs()

            self.current_project_path = folder_path
            print(f"Project Loaded from: {folder_path}")
            messagebox.showinfo("Info:", f"Project Loaded from: {folder_path}")
            self.current_cells = float(self.inputs["total cells"].get())
            self.current_stacks = float(self.inputs["No.of stacks"].get())
            self.current_model = self.inputs["Model"].get()
            # current_height = float(self.inputs["Height(mm)"].get())
            model = self.inputs["Model"].get()
            if model == "Skin Temp":
                # if voltage is not None:
                #     set_entry("Initial Voltage(V)", voltage)
                #     loaded_values.append(f"Loaded diameter = {voltage:.2f}")
                #     # if no_of_stacks is not None:
                #     set_entry("No.of stacks", 1)
                #     # loaded_values.append(f"Loaded number of stacks = {no_of_stacks}")
                #
                #     # if total_cells is not None:
                #     set_entry("total cells", 1)
                #     set_entry("cell voltage", 1)
                #     # entry = self.inputs["Initial Voltage(V)"]
                #     # entry.config(state="readonly")
                entry1 = self.inputs["No.of stacks"]
                entry1.config(state="readonly")
                entry2 = self.inputs["total cells"]
                entry2.config(state="readonly")
                entry = self.inputs["cell voltage"]
                entry.config(state="readonly")

        except Exception as e:
            print(f"Failed to load project file: {str(e)}")
            messagebox.showerror("Error",
                                 f"Failed to load project data.\n please check the path folder")  # \nPlease log in first to load your project data.

    def disable_inputs(self):
        for key, widget in self.inputs.items():
            if isinstance(widget, ttk.Combobox) and key in ["Axis", "Company", "Model"]:
                # Only set the state and style for "Axis" and "Company" comboboxes when logged out
                widget.config(state="disabled", style="Disabled.TCombobox")
                # self.update_combobox_arrow(key)
            elif isinstance(widget, ttk.Entry):
                # Disable all other input fields (ttk.Entries) as before
                widget.config(state="disabled", style="Disabled.TEntry")

    def save_project(self):
        if not self.logged_in:
            messagebox.showinfo("Login Required", "Please Log in first.")
            return  # Do not set logged_in to True here

        if not self.current_project_path:
            messagebox.showerror("Error", "No project opened!")
            return

        try:
            # Extract input values with validation
            initial_voltage = float(self.inputs["Initial Voltage(V)"].get() or 5.0)
            diameter = float(self.inputs["Diameter(mm)"].get())
            height = float(self.inputs["Height(mm)"].get())
            env_temp = float(self.inputs["Env Temp(°C)"].get())
            sk_temp = float(self.inputs["Skin Temp(°C)"].get())
            axis_label = self.inputs["Axis"].get()
            company = self.inputs["Company"].get()
            model_folder_name = self.inputs["Model"].get().strip() or "default_model"
            stacks = self.inputs["No.of stacks"].get()
            cells = self.inputs["total cells"].get()
            cell_voltage = self.inputs["cell voltage"].get()

        except ValueError:
            messagebox.showerror("Error", "Invalid input detected. Please enter numeric values for all fields.")
            return

        # Create DataFrame with validated data
        project_data = {
            "Diameter(mm)": [diameter],
            "Height(mm)": [height],
            "Env Temp(°C)": [env_temp],
            "Axis": [axis_label],
            "No.of stacks": [stacks],
            "total cells": [cells],
            "cell voltage": [cell_voltage],
            "Initial Voltage(V)": [initial_voltage],
            "Company": [company],
            "Model": [model_folder_name],
            "Skin Temp(°C)": [sk_temp]
        }
        df_project = pd.DataFrame(project_data)

        # Prepare metadata file path
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        metadata_file = os.path.join(self.current_project_path, f"metadata.csv")

        try:
            # Remove existing metadata file if it exists
            if os.path.exists(metadata_file):
                if os.name == 'nt':
                    # Ensure write permissions on Windows
                    os.chmod(metadata_file, stat.S_IWUSR)
                os.remove(metadata_file)

            # Save DataFrame to CSV
            df_project.to_csv(metadata_file, index=False)

            # Set file permissions
            if os.name == 'nt':
                # Read-only for everyone on Windows
                os.chmod(metadata_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            else:
                # Unix-based systems: read-only (444)
                os.chmod(metadata_file, 0o444)

            messagebox.showinfo("Success", f"Project saved to: {self.current_project_path}")

        except OSError as e:
            print(f"Error saving metadata file: {str(e)}")
            messagebox.showerror("Error", f"Unable to save project: {str(e)}")

        except Exception as e:
            print(f"Unexpected error while saving project: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred while saving the project: {str(e)}")

    def update_initial_voltage(self, event=None):
        try:
            total_cells = float(self.inputs["total cells"].get())
            number_of_stacks = float(self.inputs["No.of stacks"].get())
            cell_voltage = float(self.inputs["cell voltage"].get())

            # Calculate initial voltage using the new formula
            initial_voltage = (total_cells / number_of_stacks) * cell_voltage

            entry = self.inputs["Initial Voltage(V)"]
            entry.config(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, f"{initial_voltage:.2f}")
            entry.config(state="readonly")
        except (ValueError, ZeroDivisionError):
            # Handle invalid input or division by zero
            entry = self.inputs["Initial Voltage(V)"]
            entry.config(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, "")
            entry.config(state="readonly")

    def update_cells(self, event=None):
        try:
            current_cells = float(self.inputs["total cells"].get())
            new_stacks = float(self.inputs["No.of stacks"].get())

            # new_cells = float(self.inputs["total cells"].get())

            # self.new_cells =  new_cells
            current_height = float(self.inputs["Height(mm)"].get())

            model = self.inputs["Model"].get()

            print("self.prev_model before update:\n", self.prev_model)

            # print("self.current_model before if statement:\n", self.current_model)
            # if self.prev_model is not None:
            #     print("self.current_model inside if:\n", self.current_model)
            #     print("self.prev_model inside if:\n", self.prev_model)
            #     if self.prev_model != self.current_model:
            #         self.prev_stacks = None
            #         self.current_stacks = 0
            #         print("self.prev_stacks inside if:\n", self.prev_stacks)
            #         print("self.current_stacks inside if:\n", self.current_stacks)

            if new_stacks != self.prev_stacks:

                if new_stacks != 0:
                    if self.prev_stacks is not None and self.prev_stacks != 0:
                        self.new_cells = (current_cells * new_stacks) / self.prev_stacks
                        print("prev_stacks formula")
                    else:
                        self.new_cells = (current_cells * new_stacks) / self.current_stacks
                        print("current_stacks formula")

                    self.inputs["total cells"].delete(0, tk.END)
                    self.inputs["total cells"].insert(0, f"{self.new_cells:.2f}")
                    print(f"prev_stacks={self.prev_stacks}")
                    print(f"new_stacks={new_stacks}")

                    print(f"current_stacks={self.current_stacks}")
                    print(f"new_calc_cells={self.new_cells}")

                    # new_cells = self.new_cells

                    # if new_cells != self.prev_cells:
                    # if new_cells != 0:
                    # if new_stacks is not None:

                    if self.prev_stacks is not None:
                        new_height = (current_height * new_stacks) / self.prev_stacks
                        print("prev_cells formula")
                    else:
                        new_height = (current_height * new_stacks) / self.current_stacks
                        print("current_cells formula")
                    print(f"Calculated new_height: {new_height}")
                    print(f"current_height={current_height}")
                    # print(f"prev_cells={self.prev_cells}")
                    # print(f"current_cells={self.current_cells}")
                    # print(f"new_cells={new_cells}")
                    self.inputs["Height(mm)"].delete(0, tk.END)
                    self.inputs["Height(mm)"].insert(0, f"{new_height:.2f}")
                    self.prev_stacks = new_stacks
                    self.prev_model = model
                    print("self.prev_model after update:\n", self.prev_model)




                else:
                    messagebox.showwarning("Warning", "Total cells and stacks cannot be zero.")

                print(f"prev_stacks_updatedTo={self.prev_stacks}")

            # self.prev_cells = new_cells

            # print(f"update_height called with: new_cells={new_cells}, current_height={current_height}, current_cells={self.current_cells}, Calculated new_height: {new_height}")
        except (ValueError, ZeroDivisionError):
            # Handle invalid input or division by zero
            print(ValueError, ZeroDivisionError)

    # def update_height(self, event=None):
    #
    #     try:
    #         # self.new_cells =  new_cells
    #         current_height = float(self.inputs["Height(mm)"].get())
    #         current_cells = float(self.inputs["total cells"].get())
    #         new_stacks = float(self.inputs["No.of stacks"].get())
    #         if new_stacks != self.prev_stacks:
    #
    #             if self.prev_stacks is not None:
    #                 new_height = (current_height * new_stacks) / self.prev_stacks
    #                 print("prev_cells formula")
    #             else:
    #                 new_height = (current_height * new_stacks) / self.current_stacks
    #             print("current_cells formula")
    #         self.prev_stacks = new_stacks
    #         print(f"Calculated new_height: {new_height}")
    #         print(f"current_height={current_height}")
    #         # print(f"prev_cells={self.prev_cells}")
    #         # print(f"current_cells={self.current_cells}")
    #         # print(f"new_cells={new_cells}")
    #         self.inputs["Height(mm)"].delete(0, tk.END)
    #         self.inputs["Height(mm)"].insert(0, f"{new_height:.2f}")
    #
    #
    #
    #
    #     else:
    #         messagebox.showwarning("Warning", "Total cells and stacks cannot be zero.")
    #
    #         # self.prev_cells = new_cells
    #
    #         # print(f"update_height called with: new_cells={new_cells}, current_height={current_height}, current_cells={self.current_cells}, Calculated new_height: {new_height}")
    # except (ValueError, ZeroDivisionError):
    #     # Handle invalid input or division by zero
    #     print(ValueError, ZeroDivisionError)

    # def update_height(self, event=None):
    #     try:
    #
    #         #new_cells = float(self.inputs["total cells"].get())
    #         new_cells = self.new_cells
    #         #self.new_cells =  new_cells
    #         current_height = float(self.inputs["Height(mm)"].get())
    #
    #
    #
    #         #if new_cells != 0:  # Avoid division by zero
    #         if new_cells != self.prev_cells:
    #             if new_cells != 0:
    #                 if self.prev_cells is not None:
    #                     new_height = (current_height * new_cells) / self.prev_cells
    #                     print("prev_cells formula")
    #                 else:
    #                     new_height = (current_height * new_cells) / self.current_cells
    #                     print("current_cells formula")
    #                 print(f"Calculated new_height: {new_height}")
    #                 print(f"current_height={current_height}")
    #                 print(f"prev_cells={self.prev_cells}")
    #                 print(f"current_cells={self.current_cells}")
    #                 print(f"new_cells={new_cells}")
    #                 self.inputs["Height(mm)"].delete(0, tk.END)
    #                 self.inputs["Height(mm)"].insert(0, f"{new_height:.2f}")
    #             else:
    #                 messagebox.showwarning("Warning", "Total cells cannot be zero.")
    #         self.prev_cells = new_cells
    #
    #         #print(f"update_height called with: new_cells={new_cells}, current_height={current_height}, current_cells={self.current_cells}, Calculated new_height: {new_height}")
    #     except (ValueError, ZeroDivisionError):
    #         # Handle invalid input or division by zero
    #         print(ValueError, ZeroDivisionError)

    def create_buttons(self):
        #btn = ttk.Button(self.button_frame, text="Upload Profile Data", command=self.upload_allprofile, width=20)
        #btn.pack(pady=2, padx=3, fill=tk.X)

        image_path = resource_path("thermal_battery_1.png")
        img = Image.open(image_path).resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        btn = ttk.Button(self.button_frame, text="Click battery image to Generate Result", image=photo, compound=tk.TOP,
                         command=self.generate_result)

        btn.image = photo  # Keep a reference to prevent garbage collection
        btn.pack(pady=2, padx=5, fill=tk.X)
        btn = ttk.Button(self.button_frame, text="Export Result", command=self.export_result, width=20)
        btn.pack(pady=2, padx=3, fill=tk.X)
        btn = ttk.Button(self.button_frame, text="Upload Ground Truth data", command=self.upload_gt, width=20)
        btn.pack(pady=2, padx=3, fill=tk.X)
        btn = ttk.Button(self.button_frame, text="Reset", command=self.reset_app)
        btn.pack(pady=2, padx=3, fill=tk.X)

    # def change_model(self):
    #     model = self.inputs["Model"].get()

    def update_allprofile_data(self):
        # self.inputs["Model"].get()
        if self.allprofile_data is None:
            return

        # Get current input values
        initial_voltage = float(self.inputs["Initial Voltage(V)"].get() or 5.0)
        skin_temp = float(self.inputs["Skin Temp(°C)"].get() or 25)
        diameter = float(self.inputs["Diameter(mm)"].get())
        height = float(self.inputs["Height(mm)"].get())
        env_temp = float(self.inputs["Env Temp(°C)"].get())
        # axis = float(self.inputs["Axis"].get())
        axis_label = self.inputs["Axis"].get()
        axis = 0 if axis_label.lower() == 'x' else 1 if axis_label.lower() == 'y' else 2
        company = self.inputs["Company"].get()
        model = self.inputs["Model"].get()
        if skin_temp != 0:
            if env_temp >= -40 and env_temp <= 20:
                if skin_temp < 1 or skin_temp > 10:
                    messagebox.showerror("Error",
                                         "Invalid skin temperature value for the given environment temperature. Skin temperature should be between 1°C and 10°C.")
                    return "exit"
            elif env_temp == 55:
                if skin_temp < 45 or skin_temp > 55:
                    messagebox.showerror("Error",
                                         "Invalid skin temperature value for the given environment temperature. Skin temperature should be between 45°C and 55°C.")
                    return "exit"
            elif env_temp == 71:
                if skin_temp < 55 or skin_temp > 71:
                    messagebox.showerror("Error",
                                         "Invalid skin temperature value for the given environment temperature. Skin temperature should be greater than or equal to 71°C.")
                    return "exit"

        print(f"selected Model: {model}")
        print(f"Selected company: {company}")
        print(f"initial voltage in update profile: {initial_voltage}")
        print(f"diameter in update profile: {diameter}")
        print(f"height in update profile: {height}")
        print(f"env_temp in update profile: {env_temp}")
        print(f"Skin_temp in update profile: {skin_temp}")

        # Create new DataFrame with current inputs
        num_rows = len(self.allprofile_data)

        inputs = {
            'voltage': [initial_voltage] * num_rows,
            'diameter': [diameter] * num_rows,
            'height': [height] * num_rows,
            'skin_temp': [skin_temp] * num_rows,
            'env_temp': [env_temp] * num_rows,
            'axis': [axis] * num_rows
        }

        # Create a DataFrame from the inputs
        new_inputs_df = pd.DataFrame(inputs)
        print("self.allprofile before update profile", self.allprofile_data)

        # Update the allprofile_data with new inputs
        self.allprofile_data[['voltage', 'diameter', 'height', 'skin_temp',
                              'env_temp', 'axis']] = new_inputs_df

        print("self.allprofile after update profile", self.allprofile_data)
        # print(new_inputs_df)
        #self.run_backend_predictions()

    def run_backend_predictions(self):
        model = self.inputs["Model"].get()
        #if self.gt_aligned_df is None:
        try:
            if self.allprofile_data is None:
                print("Data not selected.")
                messagebox.showinfo("Info", "Data not selected.")
                return
            predicted_temp=[]
            if model == "Skin Temp":
                print(f"profile data when using {model} model before running backend :\n {self.allprofile_data}")
                #if self.gt_aligned_df is None:
                print(
                        f"profile data when using {model} model before running backend and gt is none :\n {self.allprofile_data}")
                print("allprofile_data in run pred before skin Temp:\n", self.allprofile_data)
                from Temp_backend import run_skinTemp_pred

                new_input_df, predicted_temp, Rmse = run_skinTemp_pred(self.allprofile_data)
                self.allprofile_data = None
                self.new_input_df = None
                self.new_input_df = new_input_df
                self.allprofile_data = new_input_df
                print("new_df:\n", new_input_df)
                print("new_df allprofile_data:\n", self.allprofile_data)

                #else:
                    #print(f"profile data when using {model} model and gt is none before running backend :\n {self.allprofile_data}")
                #if self.gt_aligned_df is None:
                from model2_backend import run_pred
                data_actual, predicted_voltages, rmse = run_pred(self.allprofile_data)
                    #print(f"predicted voltages when using {model} model before creating pred_df :\n {predicted_voltages}")
            	    #predicted_times = np.arange(0.1, len(predicted_voltages) * 10, 10)

            if model == "EnTest":
                print(f"profile data when using {model} model before running backend :\n {self.allprofile_data}")
                from model1_backend import run_predictions
                data_actual, predicted_voltages, rmse = run_predictions(self.allprofile_data)
                #print(f"predicted voltages when using {model} model before creating pred_df :\n {predicted_voltages}")
            	#predicted_times = np.arange(0.1, len(predicted_voltages) * 10, 10)

            # Store RMSE value
            #self.rmse_value = rmse
            
            print(f"predicted voltages when using {model} model before creating pred_df :\n {predicted_voltages}")
            predicted_times = np.arange(0.1, len(predicted_voltages) * 10, 10)
            self.predicted_df = pd.DataFrame({'time': predicted_times, 'Predicted Voltage': predicted_voltages})
            
            self.predicted_df = pd.DataFrame({'time': predicted_times, 'Predicted Voltage': predicted_voltages})
            self.predicted_temp =pd.DataFrame({'Predicted Temp':predicted_temp})
            # Clear and update table
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            tree = ttk.Treeview(self.table_frame, columns=('time', 'Predicted Voltage'), show='headings', height=5)
            tree.heading('time', text='Time (s)')
            tree.heading('Predicted Voltage', text='Voltage (V)')
            for _, row in self.predicted_df.iterrows():
                tree.insert('', tk.END, values=(f"{row['time']:.2f}", f"{row['Predicted Voltage']:.2f}"))
            tree.pack(fill=tk.X)
            # call the update_load method
            self.update_load()
            
            print("self.temp_df before update temp:\n", self.temp_df)
            self.update_skinTemp()
            print("self.temp_df after update temp:\n", self.temp_df)
            # Update prediction plot
            self.update_prediction_plot()
            #self.update_prediction_plot(model_name)
            # Update prediction plot
            # self.update_prediction_plot(model_name)
            #print("Plots generated")
            self.current_cells = float(self.inputs["total cells"].get())
            self.current_stacks = float(self.inputs["No.of stacks"].get())
            self.current_model = self.inputs["Model"].get()

            # current_height = float(self.inputs["Height(mm)"].get())

        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("Error", f"{traceback.format_exc()}")

    def on_click(self, event):
        mouse_event = event.mouseevent

        # Determine which subplot was clicked and pick the matching annotation
        if mouse_event.inaxes == self.ax:
            active_ax = self.ax
            active_annotation = self.annotation
        elif hasattr(self, 'ax_current') and mouse_event.inaxes == self.ax_current:
            active_ax = self.ax_current
            active_annotation = self.annotation_current
        else:
            return

        # Check each line in the active subplot for a pick event
        for line in active_ax.get_lines():
            if line == event.artist:
                ind = event.ind[0]
                x = line.get_xdata()[ind]
                y = line.get_ydata()[ind]

                active_annotation.xy = (x, y)

                if line.get_label() == 'Predicted':
                    active_annotation.set_text(f"Time: {x:.2f}s\nPredicted Voltage: {y:.2f} V")
                elif line.get_label() == 'Current (A)':
                    active_annotation.set_text(f"Time: {x:.2f}s\nCurrent: {y:.2f} A")
                else:
                    active_annotation.set_text(f"Time: {x:.2f}s\nGround Truth: {y:.2f}")

                active_annotation.set_visible(True)
                break

        self.prediction_figure.canvas.draw()

    def update_prediction_plot(self):  # , model_name):
        model_name = self.inputs["Model"].get()
        try:
            for widget in self.prediction_frame.winfo_children():
                widget.destroy()

            # ── Two subplots: top = Current, bottom = Voltage ──────────────
            self.prediction_figure, (self.ax_current, self.ax) = plt.subplots(
                2, 1, figsize=(8, 7), sharex=True,
                gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.35}
            )

            print("predicted_df  in update prediction plot:\n", self.predicted_df)

            # ── TOP subplot: Current (A) ────────────────────────────────────
            if hasattr(self, 'current_df') and self.current_df is not None:
                print("current_df  in update prediction plot:\n", self.current_df)
                self.ax_current.plot(self.current_df['time'],
                                     self.current_df['current'],
                                     color="purple", linewidth=2,
                                     marker='o', markersize=4, linestyle='-',
                                     label='Current (A)', picker=True)
                self.ax_current.legend(loc='upper right', fontsize=10)
            self.ax_current.set_title('Current Profile', fontsize=12)
            self.ax_current.set_ylabel('Current (A)', fontsize=10)
            self.ax_current.grid(True, linestyle='--', alpha=0.6)
            self.ax_current.set_xlim(left=0)

            # Annotation for Current subplot
            self.annotation_current = self.ax_current.annotate(
                "",
                xy=(0, 0), xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                arrowprops=dict(arrowstyle="->")
            )
            self.annotation_current.set_visible(False)

            # ── BOTTOM subplot: Predicted + Ground Truth Voltage ───────────
            self.ax.plot(self.predicted_df['time'],
                         self.predicted_df['Predicted Voltage'],
                         color='#FF5733', linewidth=2,
                         marker='o', markersize=4,
                         label='Predicted', picker=True)

            if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                print("GT_aligned df in update prediction plot:\n", self.gt_aligned_df)
                merged_df = pd.merge(self.gt_aligned_df,
                                     self.predicted_df, on='time', how='inner')
                merged_df.dropna(inplace=True)
                gt_voltage = merged_df['voltage']
                predicted_voltage = merged_df['Predicted Voltage']

                mse = mean_squared_error(gt_voltage, predicted_voltage)
                rmse = np.sqrt(mse)

                self.ax.plot(self.gt_aligned_df['time'],
                             self.gt_aligned_df['voltage'],
                             color="green", linewidth=2,
                             marker='o', markersize=4, linestyle='--',
                             label=f"Ground Truth (RMSE: {rmse:.3f} V)",
                             picker=True)

            # Legend below the bottom subplot
            self.ax.legend(bbox_to_anchor=(0., -0.22),
                           loc='lower left', borderaxespad=0.,
                           ncol=len(self.ax.lines),
                           fontsize=10)

            self.ax.set_title('Voltage Prediction', fontsize=12)
            self.ax.set_xlabel('Time (s)', fontsize=10)
            self.ax.set_ylabel('Voltage (V)', fontsize=10)
            self.ax.grid(True, linestyle='--', alpha=0.6)
            self.ax.set_xlim(left=0)
            self.ax.set_ylim(bottom=0)

            # Annotation for Voltage subplot
            self.annotation = self.ax.annotate(
                "",
                xy=(0, 0), xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                arrowprops=dict(arrowstyle="->")
            )
            self.annotation.set_visible(False)

            plt.tight_layout()

            # Create and pack the canvas
            canvas = FigureCanvasTkAgg(self.prediction_figure, master=self.prediction_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.prediction_frame)
            toolbar.update()

            # Connect pick event
            self.prediction_figure.canvas.mpl_connect('pick_event', self.on_click)

        except Exception as e:
            print(f"Error updating prediction plot: {traceback.format_exc()}{e}")

    def upload_allprofile(self):

        #print("self.current_model before if statement:\n", self.current_model)
        if self.prev_model is not None:
            #print("self.current_model inside if:\n", self.current_model)
            #print("self.prev_model inside if:\n", self.prev_model)
            if self.prev_model != self.current_model:
                self.prev_stacks = None
                self.current_stacks = 0
                #print("self.prev_stacks inside if:\n", self.prev_stacks)
                #print("self.current_stacks inside if:\n", self.current_stacks)

        missing_params = []
        invalid_params = []
        # zero_value_params = []

        for param, entry in self.inputs.items():
            if not entry.get().strip():  # Use strip() to remove leading/trailing whitespace
                missing_params.append(param)
                continue

            if param == "Axis" or param == "Company" or param == "Model":
                continue

            try:
                float(entry.get())
            except ValueError:
                invalid_params.append(param)

        if missing_params or invalid_params:  # or zero_value_params:
            error_messages = []
            if missing_params:
                error_messages.extend([f"{param}: Missing value." for param in missing_params])
                # return
            if invalid_params:
                error_messages.extend(
                    [f"{param}: Invalid input detected. Please enter a numeric value." for param in invalid_params])
            # if zero_value_params:
            #     error_messages.extend([f"{param}: Input can't be zero." for param in zero_value_params])
            messagebox.showwarning("Input Errors", "\n".join(error_messages))
            return

        # Reset existing data structures
        # self.allprofile_data = None
        self.predicted_df = None
        self.rmse_value = None
        self.gt_aligned_df = None  # Clear ground truth data

        # Clear UI elements
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        if hasattr(self, 'prediction_figure'):
            plt.close(self.prediction_figure)
            self.prediction_figure = None

        file_path = filedialog.askopenfilename()

        if not file_path:
            return

        # print("PATH", file_path)
        if file_path:
            self.file_name = os.path.splitext(os.path.basename(file_path))[
                0]  # store the uploaded file name without extension
        try:
            # df = pre_processing(file_path, uniform_sampling=True)
            # df = pd.read_csv(file_path)
            # print("column names:", df.columns)
            model = self.inputs["Model"].get()
            df = None

            if model == "EnTest":
                df = pre_processing(file_path, uniform_sampling=True)
            elif model == "Skin Temp":
                df = pre_processing(file_path, uniform_sampling=True)
            else:
                raise ValueError(f"Unknown model: {model}")

            #print(f"DF AFTER PRE PROCESSING {df}")
            try:

                required_columns = ["voltage", "skin_temp" , "diameter", "height", "env_temp", "axis", "No.of stacks", "total cells", "Company", "Model"]
                missing_columns = [c for c in required_columns if c not in df.columns]
                if missing_columns:
                    #print("\nPlease enter the following values manually:")
                    for col in missing_columns:
                        print(f"  - {col}")
                        # messagebox.showinfo("info", f"Loaded the available inputs from the uploaded file, Please enter the following values manually:{col}")

                # Helper functions

                def get_first(col, default=None):
                    return df[col].iloc[0] if col in df.columns else default

                axis_mapping = {0: "X", 1: "Y", 2: "G"}
                company_mapping = {1: "VAR", 2: "HBL"}
                model_mapping = {1: "EnTest", 2: "Skin Temp"}

                # Extract values
                voltage = get_first("voltage")
                skin_temp = get_first("skin_temp")
                diameter = get_first("diameter")
                height = get_first("height")
                env_temp = get_first("env_temp")
                axis_code = get_first("axis")
                axis = axis_mapping.get(axis_code, "X") if axis_code is not None else None
                no_of_stacks = get_first("No.of stacks")
                total_cells = get_first("total cells")
                company_code = get_first("Company")
                company = company_mapping.get(company_code, "VAR") if company_code is not None else None
                model_code = get_first("Model")
                model = model_mapping.get(model_code, "EnTest") if model_code is not None else None

                # print("inputs:--", env_temp)

                # Widget update helpers

                def set_entry(key, value):
                    widget = self.inputs[key]
                    widget.config(state="normal")
                    widget.delete(0, tk.END)
                    widget.insert(0, f"{value:.2f}" if isinstance(value, float) else str(value))
                    # widget.config(state="readonly")  # lock after loading

                def set_combobox(key, value):
                    widget = self.inputs[key]
                    widget.set(value)

                # Populate widgets

                loaded_values = []
                missing_columns = []

                models = self.inputs["Model"].get()
                if models == "EnTest":
                    if voltage is not None:
                        set_entry("Initial Voltage(V)", 0)
                        entry = self.inputs["Initial Voltage(V)"]
                        entry.config(state="readonly")
                        #loaded_values.append(f"Loaded diameter = {voltage:.2f}")
                        # if no_of_stacks is not None:
                        set_entry("No.of stacks", 0)
                        # loaded_values.append(f"Loaded number of stacks = {no_of_stacks}")

                        # if total_cells is not None:
                        set_entry("total cells", 0)
                        set_entry("cell voltage", 0)
                        # entry = self.inputs["Initial Voltage(V)"]
                        # entry.config(state="readonly")
                        entry1 = self.inputs["No.of stacks"]
                        entry1.config(state="enable")
                        entry2 = self.inputs["total cells"]
                        entry2.config(state="enable")
                        entry3 = self.inputs["cell voltage"]
                        entry3.config(state="enable")
                        # loaded_values.append(f"Loaded total cells = {total_cells}")

                if diameter is not None:
                    set_entry("Diameter(mm)", diameter)
                    loaded_values.append(f"Loaded diameter = {diameter:.2f}")
                else:
                    missing_columns.append("Diameter(mm)")
                if skin_temp is not None:
                    set_entry("Skin Temp(°C)", skin_temp)
                    loaded_values.append(f"Loaded Skin Temp = {skin_temp:.2f}")
                else:
                    set_entry("Skin Temp(°C)", 0)


                if height is not None:
                    set_entry("Height(mm)", height)
                    loaded_values.append(f"Loaded height = {height:.2f}")
                else:
                    missing_columns.append("Height(mm)")

                if env_temp is not None:
                    set_entry("Env Temp(°C)", env_temp)
                    loaded_values.append(f"Loaded env temp = {env_temp:.2f}")
                else:
                    missing_columns.append("Env Temp(°C)")

                if axis is not None:
                    set_combobox("Axis", axis)
                    loaded_values.append(f"Loaded axis = {axis}")
                else:
                    missing_columns.append("Axis")

                if no_of_stacks is not None:
                    set_entry("No.of stacks", no_of_stacks)
                    loaded_values.append(f"Loaded number of stacks = {no_of_stacks}")
                else:
                    missing_columns.append("No.of stacks")

                if total_cells is not None:
                    set_entry("total cells", total_cells)
                    loaded_values.append(f"Loaded total cells = {total_cells}")
                else:
                    missing_columns.append("total cells")

                if company is not None:
                    set_combobox("Company", company)
                    loaded_values.append(f"Loaded company = {company}")
                else:
                    missing_columns.append("Company")
                if model is not None:
                    set_combobox("Model", model)
                    loaded_values.append(f"Loaded model = {model}")
                else:
                    missing_columns.append("Model")

                #     # Build the message string
                parts = []

                if loaded_values:
                    parts.append("The following input values have been loaded successfully from uploaded profile data:")
                    for v in loaded_values:
                        parts.append(f"  - {v}")

                if missing_columns:
                    parts.append("\nThe following values are missing, you need to enter these manually to continue:")
                    for c in missing_columns:
                        parts.append(f"  - {c}")
                message = "\n".join(parts)
                #messagebox.showinfo("Loaded input entries from profile data", message)

                if models == "Skin Temp":
                    if voltage is not None:
                        set_entry("Initial Voltage(V)", voltage)
                        loaded_values.append(f"Loaded Voltage = {voltage:.2f}")
                        #if no_of_stacks is not None:
                        set_entry("No.of stacks", 1)
                        #loaded_values.append(f"Loaded number of stacks = {no_of_stacks}")

                        #if total_cells is not None:
                        set_entry("total cells", 1)
                        set_entry("cell voltage", 1)
                        # entry = self.inputs["Initial Voltage(V)"]
                        # entry.config(state="readonly")
                        entry1 = self.inputs["No.of stacks"]
                        entry1.config(state="readonly")
                        entry2 = self.inputs["total cells"]
                        entry2.config(state="readonly")
                        entry = self.inputs["cell voltage"]
                        entry.config(state="readonly")
                if models == "Skin Temp":
                    set_combobox("Axis", "G")
                    loaded_values.append(f"Loaded axis = G")

                if not self.logged_in:
                    self.disable_inputs()


            except Exception as e:
                print(traceback.format_exc(), e)
            models = self.inputs["Model"].get()
            #print("Models:", models)

            if models == "EnTest":
                #df = pre_processing(file_path, uniform_sampling=True)
                df = df[['current', 'acceleration', 'randomvibration', 'shock']]
            elif models == "Skin Temp":
                #df = df[['Current', 'Skin Temp']]
                df = df[['current']]
                print("input df after selecting Skin Temp model:\n", df)
            else:
                raise ValueError(f"Unexpected model: {models}")

            # Create time series
            current_length = len(df['current'])
            time_values = np.arange(0, current_length * 0.1, 0.1)

            # Retrieve the values from the inputs dictionary and create a new DataFrame with these values
            initial_voltage = float(self.inputs["Initial Voltage(V)"].get() or 5.0)
            skin_temp = float(self.inputs["Skin Temp(°C)"].get() or 25)
            diameter = float(self.inputs["Diameter(mm)"].get())
            #print("diameter in upload prof data before creating self.all pro:\n", diameter)
            height = float(self.inputs["Height(mm)"].get())
            #print("height in upload prof data before creating self.all pro:\n", height)
            env_temp = float(self.inputs["Env Temp(°C)"].get())
            #print("env_temp in upload prof data before creating self.all pro:\n", env_temp)
            # axis = float(self.inputs["Axis"].get())
            axis_label = self.inputs["Axis"].get()
            axis = 0 if axis_label.lower() == 'x' else 1 if axis_label.lower() == 'y' else 2
            #print("axis in upload prof data before creating self.all pro:\n", axis)
            company = self.inputs["Company"].get()
            #print("company in upload prof data before creating self.all pro:\n", company)
            model = self.inputs["Model"].get()
            # print(f"Selected company: {company}")
            #print("intitial voltage in upload prof data before creating self.all pro:\n", initial_voltage)
            if skin_temp != 0:
                if env_temp >= -40 and env_temp <= 20:
                    if skin_temp < 1 or skin_temp > 10:
                        messagebox.showerror("Error",
                                             "Invalid skin temperature value for the given environment temperature. Skin temperature should be between 1°C and 10°C.")
                        return "exit"
                        
                elif env_temp == 55:
                    if skin_temp < 45 or skin_temp > 55:
                        messagebox.showerror("Error",
                                             "Invalid skin temperature value for the given environment temperature. Skin temperature should be between 45°C and 55°C.")
                        return "exit"
                elif env_temp == 71:
                    if skin_temp < 55 or skin_temp > 71:
                        messagebox.showerror("Error",
                                             "Invalid skin temperature value for the given environment temperature. Skin temperature should be greater than or equal to 71°C.")
                        return "exit"

            inputs = {
                'time': time_values,
                'voltage': [initial_voltage] * current_length,
                'skin_temp': [skin_temp] * current_length,
                'diameter': [diameter] * current_length,
                'height': [height] * current_length,
                'env_temp': [env_temp] * current_length,
                'axis': [axis] * current_length
            }


            #self.allprofile_data[['Voltage', 'Diameter(mm)', 'Height(mm)', 'Skin Temp', 'Env Temp', 'Axis']] = new_inputs_df
            #self.allprofile_data = None
            
            inputs_df = pd.DataFrame(inputs)
            print(f"input_dataframe:\n {inputs_df}")
            #self.allprofile_data = pd.DataFrame(inputs)
            self.allprofile_data = inputs_df
            print(f"profile data after adding inputs:\n {self.allprofile_data}")
            self.allprofile_data = pd.concat([self.allprofile_data, df.reset_index(drop=True)], axis=1)
            print(f"profile data after resetting index:\n {self.allprofile_data}")

            # Concatenate the new DataFrame with the data read from the CSV file
            if models == "EnTest":
                desired_order = ['time', 'voltage', 'current', 'acceleration', 'diameter', 'height','env_temp','randomvibration', 'shock', 'axis']
            if models == "Skin Temp":
                desired_order = ['time', 'voltage', 'current', 'diameter', 'height', 'skin_temp','env_temp', 'axis']

            # # Concatenate the new DataFrame with the data read from the CSV file
            # if models == "EnTest":
            #     desired_order = ['Time', 'Voltage', 'Current', 'Acceleration', 'Diameter(mm)', 'Height(mm)', 'Skin Temp', 'Env Temp', 'RandomVibration(g2/Hz)', 'Shock', 'Axis']
            # if models == "Skin Temp":
            #     desired_order = ['Time', 'Voltage', 'Current', 'Acceleration', 'Diameter(mm)', 'Height(mm)', 'Skin Temp', 'Env Temp', 'RandomVibration(g2/Hz)', 'Shock', 'Axis']

            # desired_order = ['Time', 'Voltage', 'Current', 'Acceleration', 'Diameter(mm)', 'Height(mm)', 'Skin Temp', 'Env Temp', 'RandomVibration(g2/Hz)', 'Shock', 'Axis']
            #print(self.allprofile_data.head())
            self.allprofile_data = self.allprofile_data[desired_order]

            #print("\nAll rows of the created dataframe:(belonging to", company, "):\n")
            #print("in upload profile data after desired order:\n", self.allprofile_data)
            #print("\nFirst few rows of the desired dataframe:\n")
            #print(self.allprofile_data.head())
            messagebox.showinfo("Info", "Data uploaded successfully.")
            self.gt_flag = False
            # print("ooo")
            # Create a new folder named "preprocessed" if it doesn't exist
        except Exception as e:
            model = self.inputs["Model"].get()
            print(traceback.format_exc())
            if model == "EnTest":
                messagebox.showerror("Error",
                                     f"Failed to upload file: {str(e)}\n\nUse This application the input file format should be like(['time', 'voltage', 'current', 'acceleration', 'diameter', 'height','env_temp','randomvibration', 'shock', 'axis']")
            if model == "Skin Temp":
                messagebox.showerror("Error",
                                     f"Failed to upload file: {str(e)}\n\nUse This application the input file format should be like['time', 'voltage', 'current', 'diameter', 'height', 'skin_temp','env_temp', 'axis']")
                

    def update_load(self):
        
        try:

            if self.allprofile_data is not None:
                #print("allprofile_data:\n", self.allprofile_data.head())

                self.load_df = self.allprofile_data[['time', 'current']]
                #print("Load df:\n", self.load_df)
            if self.predicted_df is not None:
                # self.load_df = self.allprofile_data[['Time', 'Voltage']]
                stride = 100
                step = stride
                # gt_downsampled = self.gt_df.iloc[::step].copy().reset_index(drop=True)
                load_downsampled = self.load_df.iloc[::step].copy()
                load_downsampled.reset_index(drop=True, inplace=True)
                #print("load down:\n", load_downsampled, load_downsampled.shape)
                # adjusting time to match the predicted time values
                load_downsampled['time'] = [None] * len(load_downsampled)
                #print("load down Time column:\n", load_downsampled['Time'])
                # print("Predicted DF Time column:\n", self.predicted_df['Time'], self.predicted_df.shape)

                # Set time values using the corresponding indices from self.predicted_df
                load_downsampled.loc[:, 'time'] = self.predicted_df.loc[load_downsampled.index, 'time']

                # gt_downsampled['Time'] = self.predicted_df['Time']
                # print("GT down Time column after align:\n", gt_downsampled['Time'])
                # print("GT down after align:\n", gt_downsampled, gt_downsampled.shape)
                self.current_df = load_downsampled
                #print("current_df:\n", self.current_df)
                # print('GT downsampled and aligned with predicted DF:\n', self.gt_aligned_df)
                # print(self.gt_aligned_df.head())
                
        except Exception as e:
            print(traceback.format_exc())

    def update_skinTemp(self):
        
        try:
            if self.gt_aligned_df is not None:
                return

            if self.allprofile_data is None:
                return
            if self.new_input_df is None:
                return
            model = self.inputs.get("Model").get() if "Model" in self.inputs else "Skin Temp"

            if model != "Skin Temp":
                # Skip if not Skin Temp model
                self.temp_df = None
                return

            # Only create temp_df if 'Skin Temp' column exists
            if 'skin_temp' in self.allprofile_data.columns:
                #print(f"ALL PROFILE DATA{self.allprofile_data} in update")
                self.temp_df = self.allprofile_data[['time', 'skin_temp']]
            else:
                self.temp_df = None
                return
            if self.predicted_df is not None:
                #if self.new_input_df is not None:
                # self.load_df = self.allprofile_data[['Time', 'Voltage']]
                stride = 100
                step = stride
                
                # gt_downsampled = self.gt_df.iloc[::step].copy().reset_index(drop=True)
                temp_downsampled = self.temp_df.iloc[::step].copy()
                temp_downsampled.reset_index(drop=True, inplace=True)
                #print("temp load :\n", temp_downsampled, temp_downsampled.shape)
                # adjusting time to match the predicted time values
                temp_downsampled['time'] = [None] * len(temp_downsampled)
                #print("temp load  Time column:\n", temp_downsampled['Time'])
                # print("Predicted DF Time column:\n", self.predicted_df['Time'], self.predicted_df.shape)

                # Set time values using the corresponding indices from self.predicted_df
                temp_downsampled.loc[:, 'time'] = self.predicted_df.loc[temp_downsampled.index, 'time']

                # gt_downsampled['Time'] = self.predicted_df['Time']
                # print("GT down Time column after align:\n", gt_downsampled['Time'])
                # print("GT down after align:\n", gt_downsampled, gt_downsampled.shape)
                self.temp_df = temp_downsampled
                #print("temp_df in updateSkin Temp:\n", self.temp_df)
                # print('GT downsampled and aligned with predicted DF:\n', self.gt_aligned_df)
                # print(self.gt_aligned_df.head())
                #print("Hello..")
        except Exception as e:
            print(traceback.format_exc())

    def upload_gt(self):
        if self.allprofile_data is None:
            messagebox.showwarning("Warning", "Please upload load profile data first.")
            return

        file_path = askopenfilename(title="select Ground Truth CSV")
        if not file_path:
            return
        try:
            # Determine the model to decide how to read the file
            current_model = self.inputs["Model"].get()
            

            if current_model == "EnTest":
                df = pre_processing(file_path, uniform_sampling=True)
            elif current_model == "Skin Temp":
                df = pre_processing(file_path, uniform_sampling=True)
                
                # You might need to adjust the columns or processing here
                # For example:
                # df = pre_processing(df, some_parameter=True)
            else:
                raise ValueError("Unsupported model selected for Ground Truth upload.")

            # df = df[['Current', 'Acceleration', 'RandomVibration(g2/Hz)', 'Shock']]
            
            if current_model == "EnTest":
                self.gt_df = df[['time', 'voltage']]
            if current_model == "Skin Temp":
                self.gt_df = df[['time', 'voltage', 'skin_temp']]
            print("Predicted_Df in gt:\n", self.predicted_df)
            if self.predicted_df is not None:
                stride = 100
                step = stride
                # gt_downsampled = self.gt_df.iloc[::step].copy().reset_index(drop=True)
                gt_downsampled = self.gt_df.iloc[::step].copy()
                gt_downsampled.reset_index(drop=True, inplace=True)
                # print("GT down:\n", gt_downsampled, gt_downsampled.shape)
                # adjusting time to match the predicted time values
                gt_downsampled['time'] = [None] * len(gt_downsampled)
                # print("GT down Time column:\n", gt_downsampled['Time'])
                # print("Predicted DF Time column:\n", self.predicted_df['Time'], self.predicted_df.shape)

                # Set time values using the corresponding indices from self.predicted_df
                print("UPLOAD GT",self.predicted_df.columns)
                gt_downsampled.loc[:, 'time'] = self.predicted_df.loc[gt_downsampled.index, 'time']

                # gt_downsampled['Time'] = self.predicted_df['Time']
                # print("GT down Time column after align:\n", gt_downsampled['Time'])
                # print("GT down after align:\n", gt_downsampled, gt_downsampled.shape)
                self.gt_aligned_df = gt_downsampled
                print('GT downsampled and aligned with predicted DF:\n', self.gt_aligned_df)
                # print(self.gt_aligned_df.head())
                messagebox.showinfo("Info", "Ground Truth uploaded successfully.")
                self.gt_flag=True
                
                
                #update_prediction plot
                #self.update_prediction_plot()
                
        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to upload Ground Truth: {traceback.format_exc()}{str(e)}")
            self.gt_aligned_df = None

    def generate_result(self):
        try:
            if self.allprofile_data is None:
                print("Data not selected.")
                messagebox.showinfo("Info", "Data not selected.")
                return

            zero_value_params = []
            invalid_params = []

            for param, entry in self.inputs.items():
                if param == "Axis" or param == "Company":
                    continue

                try:
                    value = float(entry.get())
                    if value == 0 and param != "Skin Temp(°C)":
                        zero_value_params.append(param)
                except ValueError:
                    invalid_params.append(param)

            if zero_value_params:
                error_messages = []
                if zero_value_params:
                    error_messages.extend([f"{param}: Input can't be zero." for param in zero_value_params])
                messagebox.showwarning("Input Errors", "\n".join(error_messages))
                return
            #predicted_df = self.run_backend_predictions()
            model = self.inputs["Model"].get()
            if model == "Skin Temp":
                print("temp plot of previous profile in generate:\n", self.allprofile_data["skin_temp"])

            #if predicted_df is n
            if self.allprofile_data is not None: # and not self.gt_flag:
                update_status = self.update_allprofile_data()
                if update_status == "exit":
                	return "exit"
            #if self.update_allprofile_data():
                   
            
            #if self.gt_aligned_df is None:  
                
            self.run_backend_predictions()
             
            if self.allprofile_data is not None:

                #model = self.inputs.get("Model").get() if "Model" in self.inputs else "Skin Temp"

                # Clear previous plot if it exists
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()

                if model == "EnTest":
                    self.fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(
                        self.allprofile_data["time"],
                        self.allprofile_data["current"],
                        color="tab:blue",
                    )
                    ax.set_title("Current vs Time")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Current (A)")

                else:  # Skin Temp
                    #if self.gt_aligned_df is None:
                    if self.new_input_df is not None:
                        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                                                      tight_layout=True)
                        self.ax1.plot(
                            self.new_input_df["time"],
                            self.new_input_df["current"],
                            color="tab:blue",
                        )
                        self.ax1.set_title("Current vs Time")
                        self.ax1.set_ylabel("Current (A)")
                        if "skin_temp" in self.allprofile_data.columns and hasattr(self,
                                                                                   'temp_df') and self.temp_df is not None:
                            #if self.gt_aligned_df is None:

                            #if "Skin Temp" in self.new_input_df.columns:
                            print("temp plot of temp_df in tablerate:\n",
                                  self.temp_df["skin_temp"])
                            print("temp plot of profile data plot after run_pred in generate:\n",
                                  self.allprofile_data["skin_temp"])
                            self.ax2.plot(
                                self.temp_df["time"],
                                self.temp_df["skin_temp"],
                                color="tab:red",
                                linewidth=2,
                                marker='o', markersize=4, linestyle='-',
                                label=f"Skin Temp",
                                picker=True
                            )
                            self.ax2.set_title("Skin Temp vs Time")
                            self.ax2.set_ylabel("Skin Temp (°C)")
                        else:
                            self.ax2.set_visible(False)
                        if self.gt_aligned_df is not None:
                            gt_temp = self.gt_aligned_df["skin_temp"]
                            predicted_temp = self.temp_df["skin_temp"]

                            mse_temp = mean_squared_error(gt_temp, predicted_temp)
                            self.rmse_temp = np.sqrt(mse_temp)
                            print("self.rmse_temp in generate:\n", self.rmse_temp)
                            print("Gt_df in generate:\n", self.gt_aligned_df)
                            if "skin_temp" in self.gt_aligned_df.columns:
                                #if self.gt_aligned_df is None:

                                #if "Skin Temp" in self.new_input_df.columns:
                                print("temp plot after gt in generate:\n", self.gt_aligned_df["skin_temp"])
                                self.ax2.plot(
                                    self.gt_aligned_df["time"],
                                    self.gt_aligned_df["skin_temp"],
                                    color="tab:green", linewidth=2,
                                    marker='o', markersize=4, linestyle='--',
                                    label=f"Actual Temp (RMSE: {self.rmse_temp:.3f} °C)",
                                    picker=True
                                )
                                self.ax2.set_title("Skin Temp vs Time")
                                self.ax2.set_ylabel("Skin Temp (°C)")
                            else:
                                self.ax2.set_visible(False)

                        self.ax2.set_xlabel("Time (s)")
                        #Adjust legend position and font size
                        self.ax2.legend(bbox_to_anchor=(0., -0.25),
                                        loc='lower left', borderaxespad=0.,
                                        ncol=len(self.ax2.lines),
                                        fontsize=10)  # Reduced font size for better fit

                # ------------------------------------------------------------------
                # 7️⃣  Embed the figure in the Tkinter frame
                # ------------------------------------------------------------------
                canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # Add toolbar
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()

                # Annotation setup
                if model == "Skin Temp":
                    self.annotation1 = self.ax2.annotate("",
                                                         xy=(0, 0),
                                                         xytext=(6, 6),
                                                         textcoords="offset points",
                                                         bbox=dict(boxstyle="round,pad=0.05", fc="white", alpha=0.7),
                                                         arrowprops=dict(arrowstyle="->"), fontsize=10)
                    self.annotation1.set_visible(False)

                # # Set axis limits
                # self.ax2.set_xlim(left=0)
                # self.ax2.set_ylim(bottom=0)
                # # Set axis limits
                # self.ax1.set_xlim(left=0)
                # self.ax1.set_ylim(bottom=0)

                # Connect pick event
                self.fig.canvas.mpl_connect('pick_event', self.on_click_profile)



        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("Error", f"{traceback.format_exc()}")

    def on_click_profile(self, event):
        # if not isinstance(event, matplotlib.backend_bases.PickEvent):
        # return

        # Get the mouse event details
        mouse_event = event.mouseevent
        if mouse_event.inaxes != self.ax2:
            return

        # Get all lines in the plot
        lines = self.ax2.get_lines()

        # Check each line for a pick event
        for line in lines:
            if line == event.artist:
                ind = event.ind[0]
                x = line.get_xdata()[ind]
                y = line.get_ydata()[ind]

                # Update annotation text and position
                self.annotation1.xy = (x, y)

                if line.get_label() == "Skin Temp":
                    self.annotation1.set_text(f"Time: {x:.2f}s\npredicted: {y:.2f} °C")
                elif line.get_label().startswith("Actual Temp"):
                    self.annotation1.set_text(f"Time: {x:.2f}s\nActual: {y:.2f} °C")
                else:
                    self.annotation1.set_text(f"Time: {x:.2f}s\n{line.get_label()}: {y:.2f}")
                # elif line.get_label() == 'Current (A)':
                #     self.annotation.set_text(f"Time: {x:.2f}s\nCurrent: {y:.2f} A")
                # else:
                #     self.annotation.set_text(f"Time: {x:.2f}s\nGround Truth: {y:.2f}")

                # self.annotation.set_text(f"Time: {x:.2f}s\n{line.get_label()}: {y:.2f}")
                self.annotation1.set_visible(True)
                break

        self.fig.canvas.draw()

    def create_profile_plot(self, x, y, profile_frame, title, xlabel, ylabel):
        for widget in self.profile_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, color='#3498db', linewidth=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.profile_frame)
        canvas.draw()

        # tool bar
        toolbar = NavigationToolbar2Tk(canvas, self.profile_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Set the limits of both axes to start from zero
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # def on_click(event):
        #     if event.inaxes:
        #         x = event.xdata
        #         y_current = event.ydata  # Assuming you want to display current data point when clicked
        #         messagebox.showinfo("Data Point", f"{xlabel}: {x:.2f}\nCurrent: {y_current:.2f} A")

        # canvas.mpl_connect('button_press_event', on_click)

        # def on_click(event):
        #     if event.inaxes:
        #         x = event.xdata
        #         y = event.ydata
        #         messagebox.showinfo("Data Point", f"{xlabel}: {x:.2f}\n{ylabel}: {y:.2f}")
        #
        # canvas.mpl_connect('button_press_event', on_click)

    def export_result(self):
        try:

            if self.predicted_df is None:
                messagebox.showerror("Error", "No prediction data available to export results.")
                return
                
            model = self.inputs["Model"].get()
            if model == "EnTest":

                if self.current_project_path is not None and hasattr(self, 'current_project_path'):

                    if hasattr(self, 'file_name'):  # self.predicted_df is not None and
                        try:

                            timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

                            project_path = self.current_project_path
                            # Create the full path for the results folder with the uploaded file name and current date and time in the filename
                            result_folder = os.path.join(project_path, f"{self.file_name}_{timestamp}")

                            os.makedirs(f'{result_folder}', exist_ok=True)

                            # Create the full file paths for the CSV and PNG files in the result folder
                            csv_file_path = os.path.join(result_folder, "results.csv")
                            png_file_path = os.path.join(result_folder, "prediction_plot.png")
                            preprocessed_csv_file_path = os.path.join(result_folder, "preprocessed_data.csv")

                            # Save preprocessed data as CSV in the result folder
                            self.allprofile_data.to_csv(preprocessed_csv_file_path, index=False)
                            print(f"Preprocessed data successfully saved to {preprocessed_csv_file_path}")

                            if self.allprofile_data is not None:

                                if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:

                                    combined_df = pd.DataFrame({
                                        'Time': self.predicted_df['time'],
                                        'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                        'Ground Truth': self.gt_aligned_df['voltage'],
                                        'Current Profile': self.current_df['current']})
                                else:
                                    combined_df = pd.DataFrame({
                                        'Time': self.predicted_df['time'],
                                        'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                        'Current Profile': self.current_df['current']})

                                combined_df.to_csv(csv_file_path, index=False)
                                print("exporting combined results")
                            else:

                                # Save DataFrame as CSV and plot as PNG in the result folder
                                self.predicted_df.to_csv(csv_file_path, index=False)
                            if self.prediction_figure:
                                self.prediction_figure.savefig(png_file_path, dpi=300, bbox_inches='tight')
                                # Save preprocessed data into a .csv file in the same folder as results.csv and prediction_plot.png

                        except Exception as e:
                            print(f"Error exporting results: {e}")
                            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
                        else:
                            print("Results Exported successfully...")
                            success_message = f"Results saved successfully in the following location:\n\n{csv_file_path}"
                            messagebox.showinfo("Info", success_message)

                else:
                    if self.current_project_path is None:

                        messagebox.showerror("Warning:",
                                             "No project created or opened, saving the project in  'Unsaved Projects'!..")
                        # if hasattr(self, 'meta_file_name'):
                        if hasattr(self, 'file_name'):
                            try:
                                # Define the base folder where files will be saved (Documents folder)
                                # documents_folder = os.path.expanduser("~/Documents")
                                documents_folder = str(Path.home() / 'Documents')
                                # print(documents_folder)
                                workspace_path = os.path.join(documents_folder, "Battery Work Space")
                                # dafualt_path = os.path.join(workspace_path, "Unsaved Results")
                                unsaved = os.path.join(workspace_path, "Unsaved Results")
                                dafualt_path = os.path.join(unsaved, "Entest_Results")

                                # Define the subfolder for battery prediction results
                                # results_folder = os.path.join(workspace_path, "Battery Prediction Results")

                                # Get current date and time as a string in 'YYYY-MM-DD_HH-MM-SS' format
                                timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                                # print(timestamp)
                                # project_path = os.path.join(workspace_path, "Project_" + timestamp)
                                # project_path = self.current_project_meta_path
                                # Create the full path for the results folder with the uploaded file name and current date and time in the filename
                                result_folder = os.path.join(dafualt_path, f"{self.file_name}_{timestamp}")

                                # Create the result folder if it doesn't exist
                                # print(F'RESULT FOLDER {result_folder}')
                                os.makedirs(f'{result_folder}', exist_ok=True)

                                # Create the full file paths for the CSV and PNG files in the result folder
                                csv_file_path = os.path.join(result_folder, "results.csv")
                                png_file_path = os.path.join(result_folder, "prediction_plot.png")
                                preprocessed_csv_file_path = os.path.join(result_folder, "preprocessed_data.csv")

                            # Save preprocessed data as CSV in the result folder
                                self.allprofile_data.to_csv(preprocessed_csv_file_path, index=False)
                                print(f"Preprocessed data successfully saved to {preprocessed_csv_file_path}")
                                if self.allprofile_data is not None:

                                    if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:

                                        combined_df = pd.DataFrame({
                                            'Time': self.predicted_df['time'],
                                            'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                            'Ground Truth': self.gt_aligned_df['voltage'],
                                            'Current Profile': self.current_df['current']})
                                    else:
                                        combined_df = pd.DataFrame({
                                            'Time': self.predicted_df['time'],
                                            'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                            'Current Profile': self.current_df['current']})

                                    combined_df.to_csv(csv_file_path, index=False)
                                    print("exporting combined results")
                                else:

                                    # Save DataFrame as CSV and plot as PNG in the result folder
                                    self.predicted_df.to_csv(csv_file_path, index=False)
                                if self.prediction_figure:
                                    self.prediction_figure.savefig(png_file_path, dpi=300, bbox_inches='tight')


                            except Exception as e:
                                print(f"Error exporting results: {e}")
                                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
                            else:
                                print("Results Exported successfully...")
                                success_message = f"Results saved successfully in the following location:\n\n{csv_file_path}"
                                messagebox.showinfo("Info", success_message)

            if model == "Skin Temp":

                if self.current_project_path is not None and hasattr(self, 'current_project_path'):

                    if hasattr(self, 'file_name'):  # self.predicted_df is not None and
                        try:

                            timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

                            project_path = self.current_project_path
                            # Create the full path for the results folder with the uploaded file name and current date and time in the filename
                            result_folder = os.path.join(project_path, f"{self.file_name}_{timestamp}")

                            os.makedirs(f'{result_folder}', exist_ok=True)

                            # Create the full file paths for the CSV and PNG files in the result folder
                            csv_file_path = os.path.join(result_folder, "results.csv")
                            png_file_path = os.path.join(result_folder, "prediction_plot.png")
                            preprocessed_csv_file_path = os.path.join(result_folder, "preprocessed_data.csv")

                            # Save preprocessed data as CSV in the result folder
                            self.allprofile_data.to_csv(preprocessed_csv_file_path, index=False)
                            print(f"Preprocessed data successfully saved to {preprocessed_csv_file_path}")
                                #skin_temp_png_file_path = os.path.join(result_folder, "skin_temperature_plot.png")

                            if self.allprofile_data is not None:

                                if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                    if hasattr(self, 'temp_df') and self.temp_df is not None:
                                        print("temp_df in export:", )
                                        combined_df = pd.DataFrame({
                                            'Time': self.predicted_df['time'],
                                            'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                            'Ground Truth': self.gt_aligned_df['voltage'],
                                            'Current Profile': self.current_df['current'],
                                            'Skin Temp': self.temp_df['skin_temp']})
                                        if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                            combined_df['Actual Temp'] = self.gt_aligned_df['skin_temp']
                                    else:

                                        combined_df = pd.DataFrame({
                                            'Time': self.predicted_df['time'],
                                            'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                            'Ground Truth': self.gt_aligned_df['voltage'],
                                            'Current Profile': self.current_df['current']})

                                        if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                            combined_df['Actual Temp'] = self.gt_aligned_df['skin_temp']


                                else:
                                    if hasattr(self, 'temp_df') and self.temp_df is not None:
                                        print("temp_df in export:", )
                                        combined_df = pd.DataFrame({
                                            'Time': self.predicted_df['time'],
                                            'Predicted Voltage': self.predicted_df['Predicted Voltage'],

                                            'Current Profile': self.current_df['current'],
                                            'Skin Temp': self.temp_df['skin_temp']})
                                    else:
                                        combined_df = pd.DataFrame({
                                            'Time': self.predicted_df['time'],
                                            'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                            'Current Profile': self.current_df['current']})

                                # if self.allprofile_data is not None:
                                #
                                #     if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                #
                                #         combined_df = pd.DataFrame({
                                #             'Time': self.predicted_df['Time'],
                                #             'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                #             'Ground Truth': self.gt_aligned_df['Voltage'],
                                #             'Current Profile': self.current_df['Current'],
                                #             'Skin Temp': self.temp_df['Skin Temp']})
                                #     else:
                                #         combined_df = pd.DataFrame({
                                #             'Time': self.predicted_df['Time'],
                                #             'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                #             'Current Profile': self.current_df['Current']})

                                combined_df.to_csv(csv_file_path, index=False)
                                print("exporting combined results")
                            else:

                                # Save DataFrame as CSV and plot as PNG in the result folder
                                self.predicted_df.to_csv(csv_file_path, index=False)
                            if self.prediction_figure:
                                self.prediction_figure.savefig(png_file_path, dpi=300, bbox_inches='tight')
                                # Save preprocessed data into a .csv file in the same folder as results.csv and prediction_plot.png


                                # Save time vs skin temperature plot for Skin Temp model
                            if model == "Skin Temp" and hasattr(self, 'temp_df') and self.temp_df is not None:
                                skin_temp_png_file_path = os.path.join(result_folder, "skin_temperature_plot.png")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                if self.gt_aligned_df is not None and 'skin_temp' in self.gt_aligned_df:
                                    gt_temp = self.gt_aligned_df["skin_temp"]
                                    predicted_temp = self.temp_df["skin_temp"]
                                    mse_temp = mean_squared_error(gt_temp, predicted_temp)
                                    rmse_temp = np.sqrt(mse_temp)
                                    ax.plot(self.gt_aligned_df["time"], self.gt_aligned_df["skin_temp"],
                                            label=f'Actual Temperature (RMSE: {rmse_temp:.3f} °C)')
                                ax.plot(self.temp_df["time"], self.temp_df["skin_temp"], label='Predicted Temperature')
                                ax.set_title("Skin Temperature vs Time")
                                ax.set_xlabel("Time (s)")
                                ax.set_ylabel("Skin Temperature (°C)")
                                ax.legend()
                                fig.savefig(skin_temp_png_file_path, dpi=300, bbox_inches='tight')

                        except Exception as e:
                            print(f"Error exporting results: {e}")
                            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
                        else:
                            print("Results Exported successfully...")
                            success_message = f"Results saved successfully in the following location:\n\n{csv_file_path}"
                            messagebox.showinfo("Info", success_message)

                else:
                    if self.current_project_path is None:

                        messagebox.showerror("Warning:",
                                             "No project created or opened, saving the project in  'Unsaved Projects'!..")
                        # if hasattr(self, 'meta_file_name'):
                        if hasattr(self, 'file_name'):
                            try:
                                # Define the base folder where files will be saved (Documents folder)
                                # documents_folder = os.path.expanduser("~/Documents")
                                documents_folder = str(Path.home() / 'Documents')
                                # print(documents_folder)
                                workspace_path = os.path.join(documents_folder, "Battery Work Space")
                                unsaved = os.path.join(workspace_path, "Unsaved Results")
                                dafualt_path = os.path.join(unsaved, "SkinTemp_Results")

                                # Define the subfolder for battery prediction results
                                # results_folder = os.path.join(workspace_path, "Battery Prediction Results")

                                # Get current date and time as a string in 'YYYY-MM-DD_HH-MM-SS' format
                                timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                                # print(timestamp)
                                # project_path = os.path.join(workspace_path, "Project_" + timestamp)
                                # project_path = self.current_project_meta_path
                                # Create the full path for the results folder with the uploaded file name and current date and time in the filename
                                result_folder = os.path.join(dafualt_path, f"{self.file_name}_{timestamp}")

                                # Create the result folder if it doesn't exist
                                # print(F'RESULT FOLDER {result_folder}')
                                os.makedirs(f'{result_folder}', exist_ok=True)

                                # Create the full file paths for the CSV and PNG files in the result folder
                                csv_file_path = os.path.join(result_folder, "results.csv")
                                png_file_path = os.path.join(result_folder, "prediction_plot.png")

                                if self.allprofile_data is not None:

                                    if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                        if hasattr(self, 'temp_df') and self.temp_df is not None:
                                            print("temp_df in export:", )
                                            combined_df = pd.DataFrame({
                                                'Time': self.predicted_df['time'],
                                                'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                                'Ground Truth': self.gt_aligned_df['voltage'],
                                                'Current Profile': self.current_df['current'],
                                                'Skin Temp': self.temp_df['skin_temp']})
                                            if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                                combined_df['Actual Temp'] = self.gt_aligned_df['skin_temp']
                                        else:

                                            combined_df = pd.DataFrame({
                                                'Time': self.predicted_df['time'],
                                                'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                                'Ground Truth': self.gt_aligned_df['voltage'],
                                                'Current Profile': self.current_df['current']})

                                            if hasattr(self, 'gt_aligned_df') and self.gt_aligned_df is not None:
                                                combined_df['Actual Temp'] = self.gt_aligned_df['skin_temp']


                                    else:
                                        if hasattr(self, 'temp_df') and self.temp_df is not None:
                                            print("temp_df in export:", )
                                            combined_df = pd.DataFrame({
                                                'Time': self.predicted_df['time'],
                                                'Predicted Voltage': self.predicted_df['Predicted Voltage'],

                                                'Current Profile': self.current_df['current'],
                                                'Skin Temp': self.temp_df['skin_temp']})
                                        else:
                                            combined_df = pd.DataFrame({
                                                'Time': self.predicted_df['time'],
                                                'Predicted Voltage': self.predicted_df['Predicted Voltage'],
                                                'Current Profile': self.current_df['current']})

                                    combined_df.to_csv(csv_file_path, index=False)
                                    print("exporting combined results, Skin Temp")
                                else:

                                    # Save DataFrame as CSV and plot as PNG in the result folder
                                    self.predicted_df.to_csv(csv_file_path, index=False)
                                if self.prediction_figure:
                                    self.prediction_figure.savefig(png_file_path, dpi=300, bbox_inches='tight')
                                    # Save preprocessed data into a .csv file in the same folder as results.csv and prediction_plot.png


                                if model == "Skin Temp" and hasattr(self, 'temp_df') and self.temp_df is not None:
                                    skin_temp_png_file_path = os.path.join(result_folder, "skin_temperature_plot.png")
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    if self.gt_aligned_df is not None and 'skin_temp' in self.gt_aligned_df:
                                        gt_temp = self.gt_aligned_df["skin_temp"]
                                        predicted_temp = self.temp_df["skin_temp"]
                                        mse_temp = mean_squared_error(gt_temp, predicted_temp)
                                        rmse_temp = np.sqrt(mse_temp)
                                        ax.plot(self.gt_aligned_df["time"], self.gt_aligned_df["skin_temp"],
                                                label=f'Actual Temperature (RMSE: {rmse_temp:.3f} °C)')
                                    ax.plot(self.temp_df["time"], self.temp_df["skin_temp"],
                                            label='Predicted Temperature')
                                    ax.set_title("Skin Temperature vs Time")
                                    ax.set_xlabel("Time (s)")
                                    ax.set_ylabel("Skin Temperature (°C)")
                                    ax.legend()
                                    fig.savefig(skin_temp_png_file_path, dpi=300, bbox_inches='tight')

                            except Exception as e:
                                print(f"Error exporting results: {e}")
                                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
                            else:
                                print("Results Exported successfully...")
                                success_message = f"Results saved successfully in the following location:\n\n{csv_file_path}"
                                messagebox.showinfo("Info", success_message)

        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("Error", f"{traceback.format_exc()}")

    def reset_app(self):
        try:
            # Clear input fields
            for entry in self.inputs.values():
                entry.delete(0, tk.END)

            # Remove plots and clear relevant attributes
            for widget in self.profile_frame.winfo_children():
                widget.destroy()
            for widget in self.prediction_frame.winfo_children():
                widget.destroy()

            if hasattr(self, 'table_frame'):
                for widget in self.table_frame.winfo_children():
                    widget.destroy()

            # Reset relevant attributes
            self.allprofile_data = None
            self.predicted_df = None
            self.gt_aligned_df = None
            self.current_project_path = None
            self.file_name = None  # Add this line to reset file_name attribute

            # Re‑create the plot frame
            self.plot_frame = tk.Frame(self.profile_frame)
            self.plot_frame.pack(fill="both", expand=True)

            def set_entry(key, value):
                widget = self.inputs[key]
                widget.config(state="normal")
                widget.delete(0, tk.END)
                widget.insert(0, f"{value:.2f}" if isinstance(value, float) else str(value))

            def set_combobox(key, value):
                widget = self.inputs[key]
                widget.set(value)

            for param, entry in self.inputs.items():
                if param == "Axis":
                    set_combobox(param, "X")
                elif param == "Company":
                    set_combobox(param, "HBL")
                elif param == "Model":
                    set_combobox(param, "EnTest")
                else:
                    set_entry(param, 0)
            if not self.logged_in:
                self.disable_inputs()

            messagebox.showinfo("Info", "App Reset Successfully.\n\nYou can now use it again.")

        except Exception as e:
            print(traceback.format_exc())
            messagebox.showerror("Error", f"{traceback.format_exc()}")

    def on_closing(self):
        root.quit()  # Stop the mainloop() function
        root.destroy()  # Destroy the tkinter window


if __name__ == "__main__":
    root = tk.Tk()

    app = VoltageDropApp(root)
    # root.protocol("WM_DELETE_WINDOW", on_closing)  # Bind the protocol handler for the window close event
    root.mainloop()
