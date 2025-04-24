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
    #update_button_states()

# Function to update button states
def update_button_states():
    root.update_idletasks()

# Function to run the input-test.py script
def run_input_test():
    print(os.getcwd())
    #subprocess.Popen(["python3", os.path.join(os.getcwd(), "input-test.py")])

# Function to run the main.py script
def run_main_script():
    subprocess.Popen(["python3", os.path.normpath(os.getcwd(), "main.py")])

# Function that runs upon closing the ui window
def on_close():
    print("Window closed! Running cleanup code...")
    toggle_setting('running')
    root.destroy()  # Make sure to call `destroy()` to close the window


# Create the main application window
root = tk.Tk()
root.title("Settings Manager")
root.geometry("400x300")

# Dictionary to hold button references
buttons = {}

#initialise settings:
save_settings({"running": True, "Automatischer Zoom": False, "Auto-Bild-in-Bild": False, "gross-Bild-in-Bild": False, "klein-Bild-in-Bild": False, "chroma-key": False, "tech-preview": False})

# Create buttons to run the scripts
#input_test_button = tk.Button(root, text="Run Input Test", width=20, command=run_input_test)
#input_test_button.pack(pady=10)

#main_script_button = tk.Button(root, text="Run Main Script", width=20, command=run_main_script)
#main_script_button.pack(pady=10)

def auto_zoom_func():
    toggle_setting('Automatischer Zoom')
    settings = load_settings()
    if settings['Automatischer Zoom'] == False: pass
        #auto_switch_botton.config(text='Auto-Zoom einschalten', foreground='white', background='green')
    else: 
        #auto_switch_botton.config(text='Auto-Zoom ausschalten', foreground='white', background='red')
        pass
    #update_button_states()

auto_zoom_button = tk.Button(root, text='Auto-Zoom einschalten', width=20, command=auto_zoom_func)
auto_zoom_button.pack(pady=10)

# change value and style for auto-pip and the other variants
def auto_switch_func():
    settings = load_settings()
    if not settings["Auto-Bild-in-Bild"]:
        settings["Auto-Bild-in-Bild"] = True
        #auto_switch_botton.config(text='Auto-BiB ausschalten', foreground='white', background='red')
        settings["gross-Bild-in-Bild"] = False
        #big_switch_botton.config(text='grosses BiB einschalten', foreground='white', background='green')
        settings["klein-Bild-in-Bild"] = False
        #small_switch_botton.config(text='Kleines BiB einschalten', foreground='white', background='green')
        settings["chroma-key"] = False
    else:
        settings["Auto-Bild-in-Bild"] = False
        #auto_switch_botton.config(text='Auto-BiB einschalten', foreground='white', background='green')
    save_settings(settings)
    #update_button_states()

auto_switch_botton = tk.Button(root, text='Auto-Bild-in-Bild einschalten', width=20, command=auto_switch_func)
auto_switch_botton.pack(pady=10)

# change value and style for bigpip and the other variants
def big_switch_func():
    settings = load_settings()
    if not settings["gross-Bild-in-Bild"]:
        settings["Auto-Bild-in-Bild"] = False
        #auto_switch_botton.config(text='Auto-BiB einschalten', foreground='white', background='green')
        settings["gross-Bild-in-Bild"] = True
        #big_switch_botton.config(text='grosses BiB ausschalten', foreground='white', background='red')
        settings["klein-Bild-in-Bild"] = False
        #small_switch_botton.config(text='Kleines BiB einschalten', foreground='white', background='green')
        settings["chroma-key"] = False
    else:
        settings["gross-Bild-in-Bild"] = False
        #big_switch_botton.config(text='grosses BiB einschalten', foreground='white', background='green')
    save_settings(settings)
    #update_button_states()

big_switch_botton = tk.Button(root, text='grosses Bild-in-Bild einschalten', width=20, command=big_switch_func)
big_switch_botton.pack(pady=10)

# change value and style for small pip and the other variants
def small_switch_func():
    settings = load_settings()
    if not settings["klein-Bild-in-Bild"]:
        settings["Auto-Bild-in-Bild"] = False
        #auto_switch_botton.config(text='Auto-BiB einschalten', foreground='white', background='green')
        settings["gross-Bild-in-Bild"] = False
        #big_switch_botton.config(text='grosses BiB einschalten', foreground='white', background='green')
        settings["klein-Bild-in-Bild"] = True
        #small_switch_botton.config(text='Kleines BiB ausschalten', foreground='white', background='red')
        settings["chroma-key"] = False
    else:
        settings["klein-Bild-in-Bild"] = False
        #big_switch_botton.config(text='Kleines BiB einschalten', foreground='white', background='green')
    save_settings(settings)
    #update_button_states()

small_switch_botton = tk.Button(root, text='Kleines Bild-in-Bild einschalten', width=20, command=small_switch_func)
small_switch_botton.pack(pady=10)


# button for chroma-key control:
# change value and style for small pip and the other variants
def cc_butt_switch_func():
    settings = load_settings()
    if not settings["chroma-key"]:
        settings["Auto-Bild-in-Bild"] = False
        #auto_switch_botton.config(text='Auto-BiB einschalten', foreground='white', background='green')
        settings["gross-Bild-in-Bild"] = False
        #big_switch_botton.config(text='grosses BiB einschalten', foreground='white', background='green')
        settings["klein-Bild-in-Bild"] = False
        #small_switch_botton.config(text='Kleines BiB ausschalten', foreground='white', background='red')
        settings["chroma-key"] = True
    else:
        settings["chroma-key"] = False
        #big_switch_botton.config(text='Kleines BiB einschalten', foreground='white', background='green')
    save_settings(settings)
    #update_button_states()

cc_botton = tk.Button(root, text='chroma-key', width=20, command=cc_butt_switch_func)
cc_botton.pack(pady=10)

# change value and style for tech-preview
def tech_preview_func():
    settings = load_settings()
    if not settings.get("tech-preview", False):
        settings["tech-preview"] = True
        #tech_preview_button.config(text='Tech-Preview ausschalten', foreground='white', background='red')
    else:
        settings["tech-preview"] = False
        #tech_preview_button.config(text='Tech-Preview einschalten', foreground='white', background='green')
    save_settings(settings)
    #update_button_states()

tech_preview_button = tk.Button(root, text='Tech-Preview einschalten', width=20, command=tech_preview_func)
tech_preview_button.pack(pady=10)

# Initialize the button states
#update_button_states()

# Set the behavior when the window is closed
root.protocol("WM_DELETE_WINDOW", on_close)

# Run the Tkinter event loop
root.mainloop()
