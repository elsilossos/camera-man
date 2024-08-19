import tkinter as tk
from tkinter import messagebox
import json
import os
import subprocess

# Path to your settings.json file
SETTINGS_PATH = os.path.normpath('./settings/settings.json')

# Function to load settings
def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        messagebox.showerror("Error", "settings.json not found!")
        return {}
    with open(SETTINGS_PATH, 'r') as f:
        return json.load(f)

# Function to save settings
def save_settings(settings):
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(settings, f, indent=4)

# Function to toggle a setting
def toggle_setting(setting_name):
    settings = load_settings()
    settings[setting_name] = not settings.get(setting_name, False)
    save_settings(settings)
    update_button_states()

# Function to update button states
def update_button_states():
    settings = load_settings()
    for setting, button in buttons.items():
        button.config(relief=tk.SUNKEN if settings.get(setting, False) else tk.RAISED)

# Function to run the input-test.py script
def run_input_test():
    print(os.getcwd())
    #subprocess.Popen(["python3", os.path.join(os.getcwd(), "input-test.py")])

# Function to run the main.py script
def run_main_script():
    subprocess.Popen(["python3", os.path.normpath(os.getcwd(), "main.py")])

# Create the main application window
root = tk.Tk()
root.title("Settings Manager")
root.geometry("400x300")

# Dictionary to hold button references
buttons = {}

# List of settings
settings_list = [
    "tech-preview",
    "preview",
    "web-cam-out",
    "input-test",
    "auto-zoom",
    "auto-switch"
]

# Create buttons for each setting
for setting in settings_list:
    btn = tk.Button(root, text=setting, width=20, command=lambda s=setting: toggle_setting(s))
    btn.pack(pady=5)
    buttons[setting] = btn

# Create buttons to run the scripts
input_test_button = tk.Button(root, text="Run Input Test", width=20, command=run_input_test)
input_test_button.pack(pady=10)

main_script_button = tk.Button(root, text="Run Main Script", width=20, command=run_main_script)
main_script_button.pack(pady=10)

# Initialize the button states
update_button_states()

# Run the Tkinter event loop
root.mainloop()
