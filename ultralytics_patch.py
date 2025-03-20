import os
import sys
import warnings

# First, set environment variables to disable signal handlers
os.environ['ULTRALYTICS_SKIP_SIGNAL_HANDLERS'] = '1'
os.environ['ULTRALYTICS_SKIP_ROPE'] = '1'  # Skip ROPE checks

# Monkey patch signal.signal before importing ultralytics
import signal
original_signal = signal.signal

def patched_signal(signalnum, handler):
    try:
        return original_signal(signalnum, handler)
    except ValueError:
        # Return a placeholder value when called from a non-main thread
        return None

signal.signal = patched_signal

# Add safe globals for torch loading
import torch
from torch.serialization import add_safe_globals

# Try to add the relevant classes to the safe globals list
try:
    # Import ultralytics so we can access its classes
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    
    # Add the DetectionModel class to torch's safe globals
    add_safe_globals([DetectionModel])
    
    # Add any other classes that might be needed (backup approach)
    import inspect
    import importlib
    
    # Try to import and register other potentially needed modules
    modules_to_try = [
        'ultralytics.nn.modules',
        'ultralytics.nn.modules.conv',
        'ultralytics.yolo.engine',
        'ultralytics.yolo.engine.model'
    ]
    
    for module_name in modules_to_try:
        try:
            module = importlib.import_module(module_name)
            # Get all classes from the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                add_safe_globals([obj])
        except (ImportError, ModuleNotFoundError):
            pass
            
except Exception as e:
    print(f"Warning: Could not add all safe globals: {e}")

# Original torch.load function
original_load = torch.load

# Patched torch.load function that handles weights_only properly
def patched_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    try:
        # First try with weights_only=False (less secure but needed for older models)
        return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                           weights_only=False, **pickle_load_args)
    except Exception as e1:
        print(f"Failed to load with weights_only=False: {e1}")
        try:
            # Then try with weights_only=True (more secure, default in PyTorch 2.6+)
            return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=True, **pickle_load_args)
        except Exception as e2:
            print(f"Failed to load with weights_only=True: {e2}")
            # Re-raise the exception
            raise

# Replace torch.load with our patched version
torch.load = patched_load

# Function to get the YOLO model - will be imported safely from main.py
def get_yolo_model(model_path):
    try:
        # Import here to ensure all patches are applied first
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise