import os
import json


settings_path = os.path.join(os.getcwd(), 'settings', 'settings.json')
    

def init():
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    
    if not os.path.exists(settings_path):
        print('Did not find settings file. Made it.')
        settings = {
            'tech-preview': False,
            'preview': False,
            'web-cam-out': False,
            'input-test': False,
            'auto-zoom': False,
            'auto-switch': False,
        }
        with open(settings_path, 'w') as s:
            s.write(json.dumps(settings, indent=1))  # Corrected this line to use json.dump() directly

def drop_settings(setting: str, status):
    # Open the file in read mode first to load the settings
    with open(settings_path, 'r') as s:
        settings = json.load(s.read())
    
    # Update the setting
    settings[setting] = status
    
    # Write the updated settings back to the file
    with open(settings_path, 'w') as s:
        s.write(json.dump(settings, indent=1))  # Again, using json.dump() directly

def get_settings():
    with open(settings_path, 'r') as s:
        settings = json.loads(s.read())
        return settings

# Initialize the settings
init()
settings = get_settings()
print(settings)
