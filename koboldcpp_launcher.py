import os
import sys
import json
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import platform
import ctypes
import math
import traceback
import time
import re

# Add GPU detection imports
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
except Exception as e:
    TORCH_AVAILABLE = False
    torch = None
    print(f"Warning: PyTorch import failed: {e}", file=sys.stderr)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    print("Warning: psutil library not found. RAM and CPU information may be limited.", file=sys.stderr)

# Debug logging function
def debug_log(message):
    """Log debug messages to stderr with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] DEBUG: {message}", file=sys.stderr)

def get_gpu_info_static():
    """Get GPU information using PyTorch (static method)."""
    if not torch or not TORCH_AVAILABLE:
        msg = "PyTorch not found." if not torch else "CUDA not available via PyTorch."
        return {"available": False, "message": msg, "device_count": 0, "devices": []}

    try:
        device_count = torch.cuda.device_count()
        gpu_info = {
            "available": True,
            "device_count": device_count,
            "devices": []
        }

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info["devices"].append({
                "id": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            })
        return gpu_info
    except Exception as e:
        print(f"Error querying CUDA devices: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"available": False, "message": f"Error querying CUDA devices: {e}", "device_count": 0, "devices": []}

def get_ram_info_static():
    """Get system RAM information (static method)."""
    try:
        if sys.platform == "win32":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                kernel32 = ctypes.windll.kernel32
                memoryInfo = MEMORYSTATUSEX()
                memoryInfo.dwLength = ctypes.sizeof(memoryInfo)

                if kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryInfo)):
                    return {
                        "total_ram_bytes": memoryInfo.ullTotalPhys,
                        "total_ram_gb": round(memoryInfo.ullTotalPhys / (1024**3), 2),
                        "available_ram_bytes": memoryInfo.ullAvailPhys,
                        "available_ram_gb": round(memoryInfo.ullAvailPhys / (1024**3), 2)
                    }
                else:
                    if PSUTIL_AVAILABLE and psutil:
                        try:
                            mem = psutil.virtual_memory()
                            return {
                                "total_ram_bytes": mem.total,
                                "total_ram_gb": round(mem.total / (1024**3), 2),
                                "available_ram_bytes": mem.available,
                                "available_ram_gb": round(mem.available / (1024**3), 2)
                            }
                        except Exception as e_psutil_win:
                            print(f"Windows psutil RAM check failed: {e_psutil_win}", file=sys.stderr)
                            return {"error": f"Windows RAM checks failed (ctypes: GlobalMemoryStatusEx failed, psutil: {e_psutil_win})"}
                    else:
                        print("Windows ctypes GlobalMemoryStatusEx failed, psutil not available.", file=sys.stderr)
                        return {"error": "Windows RAM check failed (ctypes: GlobalMemoryStatusEx failed, psutil not available)"}

            except Exception as e_win:
                if PSUTIL_AVAILABLE and psutil:
                    try:
                        mem = psutil.virtual_memory()
                        return {
                            "total_ram_bytes": mem.total,
                            "total_ram_gb": round(mem.total / (1024**3), 2),
                            "available_ram_bytes": mem.available,
                            "available_ram_gb": round(mem.available / (1024**3), 2)
                        }
                    except Exception as e_psutil:
                        print(f"Windows psutil RAM check failed: {e_psutil}", file=sys.stderr)
                        return {"error": f"Windows RAM checks failed (ctypes: {e_win}, psutil: {e_psutil})"}
                else:
                    print(f"Windows RAM check failed (ctypes: {e_win}, psutil not available)", file=sys.stderr)
                    return {"error": f"Windows RAM check failed (ctypes: {e_win}, psutil not available)"}

        elif PSUTIL_AVAILABLE and psutil:
            try:
                mem = psutil.virtual_memory()
                return {
                    "total_ram_bytes": mem.total,
                    "total_ram_gb": round(mem.total / (1024**3), 2),
                    "available_ram_bytes": mem.available,
                    "available_ram_gb": round(mem.available / (1024**3), 2)
                }
            except Exception as e_psutil:
                print(f"psutil RAM check failed: {e_psutil}", file=sys.stderr)
                return {"error": f"psutil RAM check failed: {e_psutil}"}
        else:
            return {"error": "psutil not installed, cannot get RAM info on this platform."}
    except Exception as e:
        print(f"Failed to get RAM info: {str(e)}", file=sys.stderr)
        return {"error": f"Failed to get RAM info: {str(e)}"}

def get_cpu_info_static():
    """Get system CPU information (static method)."""
    try:
        if PSUTIL_AVAILABLE and psutil:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            return {
                "logical_cores": logical_cores if logical_cores is not None else 4,
                "physical_cores": physical_cores if physical_cores is not None else (logical_cores // 2 if logical_cores is not None and logical_cores > 0 else 2),
                "model_name": "N/A"
            }
        else:
            return {"error": "psutil not installed, cannot get CPU info.", "logical_cores": 4, "physical_cores": 2}
    except Exception as e:
        print(f"Failed to get CPU info: {str(e)}", file=sys.stderr)
        return {"error": f"Failed to get CPU info: {str(e)}", "logical_cores": 4, "physical_cores": 2}

class KoboldCPPLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("KoboldCPP Launcher")
        self.root.geometry("800x800")
        
        # Initialize variables first
        self.status_text = tk.StringVar(value="Ready")
        self.error_text = tk.StringVar()
        self.koboldcpp_path = tk.StringVar()
        self.config_dir = tk.StringVar()
        self.selected_config = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.model_dirs_listvar = tk.StringVar()
        self.model_path = tk.StringVar()
        self.config_var = tk.StringVar()
        self.launch_nvidia_smi = tk.BooleanVar(value=False)
        self.high_priority_var = tk.BooleanVar(value=False)
        self.override_flashattention = tk.BooleanVar(value=False)  # Add this line
        
        # Initialize thread-related variables
        self.threads_var = tk.StringVar(value="4")
        self.threads_batch_var = tk.StringVar(value="4")
        self.recommended_threads_var = tk.StringVar(value="Recommended: Based on CPU cores")
        self.recommended_threads_batch_var = tk.StringVar(value="Recommended: Based on CPU cores")
        self.override_threads = tk.StringVar(value="4")
        self.override_threads_batch = tk.StringVar(value="4")
        self.blas_threads_var = tk.StringVar(value="4")
        
        # Initialize TTS parameters
        self.override_ttsthreads = tk.StringVar(value="4")
        self.override_ttsvoice = tk.StringVar()
        self.override_ttsrate = tk.StringVar(value="1.0")
        self.override_ttspitch = tk.StringVar(value="1.0")
        self.override_ttsvolume = tk.StringVar(value="1.0")
        
        # Initialize GPU-related variables
        self.override_gpu_layers = tk.StringVar()
        self.override_tensor_split = tk.StringVar()
        self.recommended_tensor_split_var = tk.StringVar(value="Recommended: Based on GPU count")
        self.override_usecublas = tk.BooleanVar(value=False)
        self.override_usevulkan = tk.BooleanVar(value=False)
        self.override_useclblast = tk.BooleanVar(value=False)
        self.override_usecpu = tk.BooleanVar(value=False)
        self.gpu_layers_slider_var = tk.IntVar(value=0)
        self.gpu_layers_value_label = None
        self.gpu_backend_vars = {
            "CUDA": tk.BooleanVar(value=TORCH_AVAILABLE),
            "CLBlast": tk.BooleanVar(value=False),
            "Vulkan": tk.BooleanVar(value=False),
            "Metal": tk.BooleanVar(value=False)
        }
        self.gpu_vars = {}
        self.use_all_gpus = tk.BooleanVar(value=False)
        
        # Initialize context size variables
        self.override_contextsize = tk.StringVar()
        self.context_size_var = tk.StringVar(value="2048")
        self.context_slider_var = tk.IntVar(value=2048)
        self.context_value_label = None
        
        # Initialize rope config variables
        self.override_ropeconfig = tk.StringVar(value="[10000.0, 1.0]")  # Initialize with default RoPE values
        self.rope_config_var = tk.StringVar(value="[10000.0, 1.0]")  # Add back the missing variable
        self.rope_freq_base_var = tk.StringVar(value="10000.0")
        self.rope_freq_scale_var = tk.StringVar(value="1.0")
        
        # Initialize memory-related variables
        self.override_blasbatchsize = tk.StringVar()
        self.override_blasthreads = tk.StringVar()
        self.override_lora = tk.StringVar()
        self.override_noshift = tk.BooleanVar(value=False)
        self.override_nofastforward = tk.BooleanVar(value=False)
        self.override_usemmap = tk.BooleanVar(value=False)
        self.override_usemlock = tk.BooleanVar(value=False)
        self.override_noavx2 = tk.BooleanVar(value=False)
        self.override_noblas = tk.BooleanVar(value=False)
        self.override_nommap = tk.BooleanVar(value=False)
        self.override_usemirostat = tk.BooleanVar(value=False)
        self.override_mirostat_tau = tk.StringVar(value="5.0")
        self.override_mirostat_eta = tk.StringVar(value="0.1")
        self.override_temperature = tk.StringVar(value="0.7")
        self.override_min_p = tk.StringVar(value="0.0")
        self.override_seed = tk.StringVar(value="-1")
        self.override_n_predict = tk.StringVar(value="-1")
        self.override_ignore_eos = tk.BooleanVar(value=False)
        
        # Initialize security parameters
        self.override_ssl = tk.BooleanVar(value=False)
        self.override_nocertify = tk.BooleanVar(value=False)
        self.override_password = tk.StringVar()
        
        # Initialize additional variables for configuration
        self.port_var = tk.StringVar(value="5001")
        self.host_var = tk.StringVar(value="0.0.0.0")
        self.launch_var = tk.BooleanVar(value=False)
        self.use_mlock_var = tk.BooleanVar(value=False)
        
        # Initialize lists
        self.model_dirs = []
        self.model_files = []
        self.config_files = []
        self.custom_params = []
        
        # Initialize UI elements
        self.custom_params_listbox = None
        self.custom_param_entry = None
        
        # Initialize settings file as Path
        self.settings_file = Path(os.path.expanduser("~")) / ".koboldcpp_launcher_settings.json"
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding=15)
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Set app theme colors
        self.bg_color = "#f5f5f5"
        self.accent_color = "#4a6ba0"
        self.button_color = "#5f87c7"
        self.hover_color = "#3a5a8f"
        self.error_color = "#f44336"
        
        # Initialize system information variables
        self.gpu_info = get_gpu_info_static()
        self.ram_info = get_ram_info_static()
        self.cpu_info = get_cpu_info_static()
        
        # Set recommended thread values based on CPU info
        if self.cpu_info.get('logical_cores'):
            self.threads_var.set(str(self.cpu_info['logical_cores']))
            self.threads_batch_var.set(str(self.cpu_info['logical_cores']))
            self.recommended_threads_var.set(f"Recommended: {self.cpu_info['logical_cores']} (Your CPU logical cores)")
            self.recommended_threads_batch_var.set(f"Recommended: {self.cpu_info['logical_cores']} (Your CPU logical cores)")
        
        # Load settings
        self.load_settings()
        
        # Create widgets
        self.create_widgets()
        
        # Update model list after loading settings
        self.update_model_list()
        
        # Variables
        self.process = None
        self.running = False
        
        # Model selection variables
        self.model_dirs = []
        self.model_dirs_listvar = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.model_files = []
        
        # GPU selection variables
        self.gpu_selection = {}
        for gpu in self.gpu_info.get("devices", []):
            self.gpu_selection[gpu["id"]] = tk.BooleanVar(value=True)
        
        # Initialize GPU override variables
        self.override_tensor_split = tk.StringVar()
        self.recommended_tensor_split_var = tk.StringVar(value="Recommended: Based on GPU count")
        self.override_usecublas = tk.BooleanVar(value=False)
        self.override_usevulkan = tk.BooleanVar(value=False)
        self.override_useclblast = tk.BooleanVar(value=False)
        self.override_usecpu = tk.BooleanVar(value=False)
        
        # Initialize lists
        self.model_dirs = []
        self.model_files = []
        self.config_files = []
        self.custom_params = []
        
        # Custom parameters variables
        self.custom_params = []
        self.custom_params_listvar = tk.StringVar()
        
        # Load settings
        self.load_settings()
        
        # Set up grid
        self.main_frame.columnconfigure(0, weight=1)
        for i in range(11):
            self.main_frame.rowconfigure(i, weight=1)
        
        # Update config list if directory is set
        if self.config_dir.get():
            self.update_config_list()
            
        # Display system information
        self.display_system_info()
        
    def display_system_info(self):
        """Display system information in the UI."""
        # Create system info frame with fixed height
        system_info_frame = ttk.LabelFrame(self.main_frame, text="System Information", padding=5)
        system_info_frame.pack(fill="x", padx=10, pady=(2, 0))
        
        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(system_info_frame, height=100)  # Reduced height to 100 pixels
        scrollbar = ttk.Scrollbar(system_info_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure canvas scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # CPU and RAM Info on same row
        cpu_ram_frame = ttk.Frame(scrollable_frame)
        cpu_ram_frame.pack(fill="x", pady=1)
        
        cpu_info_text = f"CPU Cores: {self.cpu_info.get('physical_cores', 'N/A')} physical, {self.cpu_info.get('logical_cores', 'N/A')} logical"
        ttk.Label(cpu_ram_frame, text=cpu_info_text).pack(side="left", padx=(0, 20))
        
        if "error" not in self.ram_info:
            ram_info_text = f"RAM: {self.ram_info.get('total_ram_gb', 'N/A')} GB total, {self.ram_info.get('available_ram_gb', 'N/A')} GB available"
            ttk.Label(cpu_ram_frame, text=ram_info_text).pack(side="left")
        
        # GPU Info
        if self.gpu_info["available"]:
            gpu_info_text = f"GPUs: {self.gpu_info['device_count']} detected"
            ttk.Label(scrollable_frame, text=gpu_info_text).pack(anchor="w", pady=1)
            
            # Display individual GPU info
            for gpu in self.gpu_info["devices"]:
                gpu_details = f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']} GB VRAM)"
                ttk.Label(scrollable_frame, text=gpu_details).pack(anchor="w", pady=1)
        else:
            gpu_info_text = f"GPU: {self.gpu_info.get('message', 'Not available')}"
            ttk.Label(scrollable_frame, text=gpu_info_text).pack(anchor="w", pady=1)
            
        # Display recommendations
        recommendations = self.recommend_parameters()
        if recommendations:
            ttk.Label(scrollable_frame, text="Recommended Parameters:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=1)
            for rec in recommendations:
                ttk.Label(scrollable_frame, text=f"• {rec}", font=("Segoe UI", 9)).pack(anchor="w", pady=1)
    
    def create_widgets(self):
        # Create main notebook for tabs
        main_notebook = ttk.Notebook(self.main_frame)
        main_notebook.pack(expand=True, fill=tk.BOTH, padx=1, pady=0)
        
        # Create tabs
        main_tab = ttk.Frame(main_notebook)
        config_tab = ttk.Frame(main_notebook)
        model_tab = ttk.Frame(main_notebook)
        params_tab = ttk.Frame(main_notebook)
        monitor_tab = ttk.Frame(main_notebook)
        settings_tab = ttk.Frame(main_notebook)
        
        main_notebook.add(main_tab, text="Main")
        main_notebook.add(config_tab, text="Configuration")
        main_notebook.add(model_tab, text="Models")
        main_notebook.add(params_tab, text="Parameters")
        main_notebook.add(monitor_tab, text="Monitoring")
        main_notebook.add(settings_tab, text="Settings")
        
        # --- Main Tab ---
        # Title, System Info, and Controls in a single frame
        main_header_frame = ttk.Frame(main_tab)
        main_header_frame.pack(fill=tk.X, padx=1, pady=0)
        
        # Title and System Info in a single row
        title_label = ttk.Label(main_header_frame, text="KoboldCPP Launcher", style="Header.TLabel")
        title_label.pack(side=tk.LEFT, padx=(0, 2))
        
        # System Info in a compact frame
        system_info_frame = ttk.Frame(main_header_frame)
        system_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Combine all system info into a single line
        system_info_text = []
        system_info_text.append(f"CPU: {self.cpu_info.get('physical_cores', 'N/A')}p/{self.cpu_info.get('logical_cores', 'N/A')}l")
        if "error" not in self.ram_info:
            system_info_text.append(f"RAM: {self.ram_info.get('total_ram_gb', 'N/A')}GB")
        if self.gpu_info["available"]:
            system_info_text.append(f"GPU: {self.gpu_info['device_count']} detected")
        
        ttk.Label(system_info_frame, text=" | ".join(system_info_text)).pack(side=tk.LEFT, padx=1)
        
        # Control Buttons in the same row
        stop_btn = tk.Button(main_header_frame, text="Stop", 
                            command=self.stop_koboldcpp,
                            bg="#e74c3c", fg="white",
                            activebackground="#c0392b", activeforeground="white",
                            relief=tk.FLAT, padx=2, pady=0)
        stop_btn.pack(side=tk.RIGHT, padx=1)
        
        generate_btn = tk.Button(main_header_frame, text="PS1", 
                                command=self.create_launcher_file,
                                bg=self.button_color, fg="white",
                                activebackground=self.hover_color, activeforeground="white",
                                relief=tk.FLAT, padx=2, pady=0)
        generate_btn.pack(side=tk.RIGHT, padx=1)
        
        launch_btn = tk.Button(main_header_frame, text="Launch", 
                              command=self.launch_koboldcpp,
                              bg=self.accent_color, fg="white",
                              activebackground=self.hover_color, activeforeground="white",
                              relief=tk.FLAT, padx=2, pady=0)
        launch_btn.pack(side=tk.RIGHT, padx=1)
        
        # Status Bar - reduced height
        status_bar = ttk.Label(main_tab, textvariable=self.status_text, 
                              relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=1, pady=0)
        
        # --- Configuration Tab ---
        # Executable Selection
        exe_frame = ttk.LabelFrame(config_tab, text="KoboldCPP Executable", padding=10)
        exe_frame.pack(fill=tk.X, padx=10, pady=5)
        
        exe_path_entry = ttk.Entry(exe_frame, textvariable=self.koboldcpp_path, width=50)
        exe_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        exe_browse_btn = tk.Button(exe_frame, text="Browse", 
                                  command=self.browse_executable,
                                  bg=self.button_color, fg="white",
                                  activebackground=self.hover_color, activeforeground="white",
                                  relief=tk.FLAT, padx=10)
        exe_browse_btn.pack(side=tk.RIGHT)
        
        # Config Directory Selection
        config_dir_frame = ttk.LabelFrame(config_tab, text="Configuration Files Directory", padding=10)
        config_dir_frame.pack(fill=tk.X, padx=10, pady=5)
        
        config_dir_entry = ttk.Entry(config_dir_frame, textvariable=self.config_dir, width=50)
        config_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        config_dir_browse_btn = tk.Button(config_dir_frame, text="Browse", 
                                         command=self.browse_config_dir,
                                         bg=self.button_color, fg="white",
                                         activebackground=self.hover_color, activeforeground="white",
                                         relief=tk.FLAT, padx=10)
        config_dir_browse_btn.pack(side=tk.RIGHT)
        
        # Config Files List
        config_list_frame = ttk.LabelFrame(config_tab, text="Available Configuration Files", padding=10)
        config_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create a Frame for the listbox and scrollbar
        list_container = ttk.Frame(config_list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox with config files
        self.config_listbox = tk.Listbox(list_container, height=10, 
                                        selectmode=tk.SINGLE,
                                        yscrollcommand=scrollbar.set,
                                        bg="white", bd=1, relief=tk.SOLID)
        self.config_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.config_listbox.bind('<<ListboxSelect>>', self.on_config_select)
        
        scrollbar.config(command=self.config_listbox.yview)
        
        # Buttons frame
        buttons_frame = ttk.Frame(config_list_frame)
        buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Load config button
        load_config_btn = tk.Button(buttons_frame, text="Load Configuration", 
                                   command=self.load_selected_config,
                                   bg=self.button_color, fg="white",
                                   activebackground=self.hover_color, activeforeground="white",
                                   relief=tk.FLAT, padx=10)
        load_config_btn.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_btn = tk.Button(buttons_frame, text="Refresh List", 
                               command=self.update_config_list,
                               bg=self.button_color, fg="white",
                               activebackground=self.hover_color, activeforeground="white",
                               relief=tk.FLAT, padx=10)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Save current config button
        save_config_btn = tk.Button(buttons_frame, text="Save Current as Config", 
                                   command=self.save_current_config,
                                   bg=self.button_color, fg="white",
                                   activebackground=self.hover_color, activeforeground="white",
                                   relief=tk.FLAT, padx=10)
        save_config_btn.pack(side=tk.LEFT, padx=5)
        
        # Remove config button
        remove_config_btn = tk.Button(buttons_frame, text="Remove Selected", 
                                     command=self.remove_selected_config,
                                     bg=self.button_color, fg="white",
                                     activebackground=self.hover_color, activeforeground="white",
                                     relief=tk.FLAT, padx=10)
        remove_config_btn.pack(side=tk.LEFT, padx=5)
        
        # --- Model Tab ---
        # Model Directories Frame
        model_dirs_frame = ttk.LabelFrame(model_tab, text="Model Directories", padding=5)
        model_dirs_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model Directories Listbox with Scrollbar
        model_dirs_container = ttk.Frame(model_dirs_frame)
        model_dirs_container.pack(fill=tk.BOTH, expand=True)
        
        self.model_dirs_listbox = tk.Listbox(model_dirs_container, 
                                           height=5,
                                           selectmode=tk.SINGLE,
                                           bg="white",
                                           relief=tk.SUNKEN)
        self.model_dirs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        model_dirs_scrollbar = ttk.Scrollbar(model_dirs_container, 
                                           orient=tk.VERTICAL,
                                           command=self.model_dirs_listbox.yview)
        model_dirs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_dirs_listbox.config(yscrollcommand=model_dirs_scrollbar.set)
        
        # Model Directories Buttons
        model_dirs_buttons = ttk.Frame(model_dirs_frame)
        model_dirs_buttons.pack(fill=tk.X, pady=(5, 0))
        
        add_dir_btn = tk.Button(model_dirs_buttons, text="Add Directory",
                               command=self.add_model_dir,
                               bg=self.button_color, fg="white",
                               activebackground=self.hover_color, activeforeground="white",
                               relief=tk.FLAT, padx=10)
        add_dir_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        remove_dir_btn = tk.Button(model_dirs_buttons, text="Remove Directory",
                                  command=self.remove_model_dir,
                                  bg=self.button_color, fg="white",
                                  activebackground=self.hover_color, activeforeground="white",
                                  relief=tk.FLAT, padx=10)
        remove_dir_btn.pack(side=tk.LEFT)
        
        # Model Files Frame
        model_files_frame = ttk.LabelFrame(model_tab, text="Model Files", padding=10)
        model_files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Model Files Listbox with Scrollbar
        model_files_container = ttk.Frame(model_files_frame)
        model_files_container.pack(fill=tk.BOTH, expand=True)
        
        self.model_listbox = tk.Listbox(model_files_container,
                                      height=15,
                                      selectmode=tk.SINGLE,
                                      bg="white",
                                      relief=tk.SUNKEN)
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        model_files_scrollbar = ttk.Scrollbar(model_files_container,
                                            orient=tk.VERTICAL,
                                            command=self.model_listbox.yview)
        model_files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.config(yscrollcommand=model_files_scrollbar.set)
        
        # Refresh Button
        refresh_btn = tk.Button(model_files_frame, text="Refresh",
                               command=self.update_model_list,
                               bg=self.button_color, fg="white",
                               activebackground=self.hover_color, activeforeground="white",
                               relief=tk.FLAT, padx=10)
        refresh_btn.pack(side=tk.RIGHT, pady=(5, 0))
        
        # Bind selection event
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        
        # --- Parameters Tab ---
        # Create notebook for parameter categories
        param_notebook = ttk.Notebook(params_tab)
        param_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Basic Parameters Tab
        basic_frame = ttk.Frame(param_notebook)
        param_notebook.add(basic_frame, text="Basic")
        
        # GPU Backend Tab
        gpu_frame = ttk.Frame(param_notebook)
        param_notebook.add(gpu_frame, text="GPU Backend")
        
        # Memory Tab
        memory_frame = ttk.Frame(param_notebook)
        param_notebook.add(memory_frame, text="Memory")
        
        # Security Tab
        security_frame = ttk.Frame(param_notebook)
        param_notebook.add(security_frame, text="Security")
        
        # Advanced Tab
        advanced_frame = ttk.Frame(param_notebook)
        param_notebook.add(advanced_frame, text="Advanced")
        
        # Populate parameter frames
        self._create_basic_parameters(basic_frame)
        self._create_gpu_parameters(gpu_frame)
        self._create_memory_parameters(memory_frame)
        self._create_security_parameters(security_frame)
        self._create_advanced_parameters(advanced_frame)
        
        # --- Monitoring Tab ---
        # Performance monitoring frame
        monitor_frame = ttk.LabelFrame(monitor_tab, text="Performance Monitoring", padding=10)
        monitor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add monitoring widgets here
        self._create_monitoring_widgets(monitor_frame)
        
        # --- Settings Tab ---
        # Application settings frame
        settings_frame = ttk.LabelFrame(settings_tab, text="Application Settings", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add settings widgets here
        self._create_settings_widgets(settings_frame)
        
        # Error Display Area (at bottom of main window)
        error_frame = ttk.LabelFrame(self.main_frame, text="Errors and Messages", padding=10)
        error_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Error text with scrollbar
        error_text_frame = ttk.Frame(error_frame)
        error_text_frame.pack(fill=tk.BOTH, expand=True)
        
        error_scrollbar = ttk.Scrollbar(error_text_frame)
        error_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.error_display = tk.Text(error_text_frame, height=3, width=50, 
                                    wrap=tk.WORD, bg="white", fg=self.error_color,
                                    yscrollcommand=error_scrollbar.set)
        self.error_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        error_scrollbar.config(command=self.error_display.yview)
        
        # Copy button for error text
        copy_btn = tk.Button(error_frame, text="Copy to Clipboard", 
                            command=self.copy_error_to_clipboard,
                            bg=self.button_color, fg="white",
                            activebackground=self.hover_color, activeforeground="white",
                            relief=tk.FLAT, padx=5)
        copy_btn.pack(side=tk.RIGHT, pady=(5, 0))
    
    def add_model_dir(self):
        """Add a new model directory to the list."""
        debug_log("Starting to add model directory")
        directory = filedialog.askdirectory(title="Select Model Directory")
        if directory:
            try:
                # Convert to absolute path and normalize
                directory = os.path.abspath(directory)
                debug_log(f"Selected directory: {directory}")
                if directory not in self.model_dirs:
                    debug_log("Adding new directory to model_dirs")
                    self.model_dirs.append(directory)
                    self._update_model_dirs_listbox()
                    self.update_model_list()  # Update model list immediately
                    self.save_settings()
                    debug_log("Settings saved after adding directory")
                    self.status_text.set(f"Added model directory: {directory}")
                else:
                    debug_log("Directory already exists in model_dirs")
            except Exception as e:
                debug_log(f"Error adding directory: {str(e)}")
                self.status_text.set(f"Error adding directory: {str(e)}")
                self.show_error_message(f"Error adding directory: {str(e)}")

    def remove_model_dir(self):
        """Remove the selected model directory from the list."""
        debug_log("Starting to remove model directory")
        selection = self.model_dirs_listbox.curselection()
        if selection:
            index = selection[0]
            debug_log(f"Selected directory index: {index}")
            removed_dir = self.model_dirs.pop(index)
            debug_log(f"Removed directory: {removed_dir}")
            self._update_model_dirs_listbox()
            self.update_model_list()
            self.save_settings()
            debug_log("Settings saved after removing directory")
            self.status_text.set(f"Removed model directory: {removed_dir}")
        else:
            debug_log("No directory selected for removal")

    def _update_model_dirs_listbox(self):
        """Update the model directories listbox with current directories."""
        debug_log("Updating model directories listbox")
        try:
            # Clear the listbox
            self.model_dirs_listbox.delete(0, tk.END)
            debug_log("Cleared listbox")
            
            # Add each directory to the listbox
            for directory in self.model_dirs:
                debug_log(f"Adding directory to listbox: {directory}")
                self.model_dirs_listbox.insert(tk.END, directory)
            
            # Update the status to show number of directories
            self.status_text.set(f"Model directories: {len(self.model_dirs)}")
            debug_log(f"Updated status with {len(self.model_dirs)} directories")
        except Exception as e:
            debug_log(f"Error updating directories list: {str(e)}")
            self.status_text.set(f"Error updating directories list: {str(e)}")
            self.show_error_message(f"Error updating directories list: {str(e)}")

    def update_model_list(self):
        debug_log("Starting model list update")
        debug_log(f"Current selected model before update: {self.selected_model.get()}")
        self.model_listbox.delete(0, tk.END)
        self.model_files = []
        
        # Patterns to match multi-part files
        multipart_pattern = re.compile(r"^(.*?)(?:-\d{5}-of-\d{5}|-F\d+)\.gguf$", re.IGNORECASE)
        first_part_pattern = re.compile(r"^(.*?)-(?:00001-of-\d{5}|F1)\.gguf$", re.IGNORECASE)
        processed_multipart_bases = set()
        
        debug_log(f"Scanning {len(self.model_dirs)} model directories")
        for directory in self.model_dirs:
            try:
                debug_log(f"Scanning directory: {directory}")
                # Use os.walk to recursively search through all subdirectories
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith(('.gguf', '.bin', '.pt', '.safetensors')):
                            full_path = os.path.join(root, file)
                            debug_log(f"Found model file: {full_path}")
                            
                            # Skip non-model GGUF files
                            if file.lower().endswith('.gguf') and ('mmproj' in file.lower() or file.lower().endswith('.bin.gguf')):
                                debug_log(f"Skipping non-model file: {file}")
                                continue
                                
                            # Handle multi-part files
                            first_part_match = first_part_pattern.match(file)
                            if first_part_match:
                                base_name = first_part_match.group(1)
                                if base_name not in processed_multipart_bases:
                                    debug_log(f"Found first part of multi-part model: {base_name}")
                                    self.model_files.append(full_path)
                                    rel_path = os.path.relpath(root, directory)
                                    if rel_path == '.':
                                        self.model_listbox.insert(tk.END, f"{base_name} ({directory})")
                                    else:
                                        self.model_listbox.insert(tk.END, f"{base_name} ({directory}/{rel_path})")
                                    processed_multipart_bases.add(base_name)
                                continue
                                
                            # Handle subsequent parts of multi-part files
                            multi_match = multipart_pattern.match(file)
                            if multi_match:
                                base_name = multi_match.group(1)
                                debug_log(f"Found subsequent part of multi-part model: {base_name}")
                                processed_multipart_bases.add(base_name)
                                continue
                                
                            # Handle single-part files
                            if file.lower().endswith('.gguf') and file[:-5] not in processed_multipart_bases:
                                debug_log(f"Found single-part model: {file}")
                                self.model_files.append(full_path)
                                rel_path = os.path.relpath(root, directory)
                                if rel_path == '.':
                                    self.model_listbox.insert(tk.END, f"{file[:-5]} ({directory})")
                                else:
                                    self.model_listbox.insert(tk.END, f"{file[:-5]} ({directory}/{rel_path})")
            except Exception as e:
                debug_log(f"Error scanning directory {directory}: {str(e)}")
                self.status_text.set(f"Error scanning directory {directory}: {str(e)}")
        
        # Restore selection if it exists
        current_selection = self.selected_model.get()
        if current_selection:
            debug_log(f"Restoring previous selection: {current_selection}")
            try:
                # Normalize paths for comparison
                current_selection = os.path.normpath(current_selection)
                for i, model_file in enumerate(self.model_files):
                    if os.path.normpath(model_file) == current_selection:
                        self.model_listbox.selection_set(i)
                        debug_log(f"Restored selection at index: {i}")
                        break
            except ValueError:
                debug_log(f"Previous selection {current_selection} not found in updated list")
        
        if not self.model_files:
            debug_log("No model files found in any directory")
            self.status_text.set("No model files found in any directory")
        else:
            debug_log(f"Found {len(self.model_files)} model files")
            self.status_text.set(f"Found {len(self.model_files)} model files")

    def on_config_select(self, event):
        """Handle configuration file selection."""
        selection = self.config_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.config_files):
                selected_file = self.config_files[index]
                self.selected_config.set(selected_file)
                full_path = os.path.join(self.config_dir.get(), selected_file)
                self.status_text.set(f"Selected: {full_path}")
                self.save_settings()
                # Load the selected configuration
                self.load_selected_config()

    def load_selected_config(self):
        debug_log("Loading selected configuration")
        try:
            selected_file = self.selected_config.get()
            if not selected_file:
                debug_log("No configuration selected")
                return False
                
            config_path = os.path.join(self.config_dir.get(), selected_file)
            if not os.path.exists(config_path):
                debug_log(f"Config file not found: {config_path}")
                return False
                
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            debug_log(f"Loaded configuration data: {json.dumps(config_data, indent=2)}")
            
            # Update GPU selection
            if 'gpu' in config_data:
                gpu_selection = config_data['gpu']
                debug_log(f"Setting GPU selection to: {gpu_selection}")
                # Handle the "All GPUs" case
                if gpu_selection == [-1]:
                    self.use_all_gpus.set(True)
                    for gpu_id, var in self.gpu_vars.items():
                        var.set(True)
                else:
                    self.use_all_gpus.set(False)
                    for gpu_id, var in self.gpu_vars.items():
                        var.set(gpu_id in gpu_selection)
            
            # Update GPU backend
            if 'gpu_backend' in config_data:
                backend = config_data['gpu_backend']
                debug_log(f"Setting GPU backend to: {backend}")
                for backend_name, var in self.gpu_backend_vars.items():
                    var.set(backend_name == backend)
            
            # Update GPU layers
            if 'gpulayers' in config_data:
                gpulayers = config_data['gpulayers']
                debug_log(f"Setting GPU layers to: {gpulayers}")
                self.override_gpu_layers.set(str(gpulayers))
                self.gpu_layers_slider_var.set(gpulayers)
                self.gpu_layers_value_label.config(text=str(gpulayers))
            
            # Update GPU-specific flags
            self.override_flashattention.set(config_data.get('flashattention', False))
            self.override_usecublas.set(config_data.get('usecublas', False))
            self.override_usevulkan.set(config_data.get('usevulkan', False))
            self.override_useclblast.set(config_data.get('useclblast', False))
            self.override_usecpu.set(config_data.get('usecpu', False))
            
            # Update other parameters
            self.model_path.set(config_data.get('model', ''))
            self.threads_var.set(config_data.get('threads', 0))
            self.blas_threads_var.set(config_data.get('blasthreads', 0))
            self.context_size_var.set(config_data.get('contextsize', 2048))
            self.rope_config_var.set(config_data.get('ropeconfig', ''))
            self.rope_freq_base_var.set(config_data.get('ropefreqbase', 10000.0))
            self.rope_freq_scale_var.set(config_data.get('ropefreqscale', 1.0))
            self.override_usemlock.set(config_data.get('usemlock', False))
            self.override_noavx2.set(config_data.get('noavx2', False))
            self.override_noblas.set(config_data.get('noblas', False))
            self.override_nommap.set(config_data.get('nommap', False))
            self.override_usemirostat.set(config_data.get('usemirostat', False))
            self.override_mirostat_tau.set(config_data.get('mirostat_tau', 5.0))
            self.override_mirostat_eta.set(config_data.get('mirostat_eta', 0.1))
            self.override_temperature.set(config_data.get('temperature', 0.7))
            self.override_min_p.set(config_data.get('min_p', 0.0))
            self.override_seed.set(config_data.get('seed', -1))
            self.override_n_predict.set(config_data.get('n_predict', -1))
            self.override_ignore_eos.set(config_data.get('ignore_eos', False))
            self.override_ssl.set(config_data.get('ssl', False))
            self.override_nocertify.set(config_data.get('nocertify', False))
            self.override_password.set(config_data.get('password', ''))
            self.override_blasbatchsize.set(config_data.get('blasbatchsize', 512))
            self.override_blasthreads.set(config_data.get('blasthreads', 4))
            self.override_contextsize.set(config_data.get('contextsize', 2048))
            self.override_tensor_split.set(config_data.get('tensor_split', ''))
            self.override_usecublas.set(config_data.get('usecublas', False))
            self.override_usevulkan.set(config_data.get('usevulkan', False))
            self.override_useclblast.set(config_data.get('useclblast', False))
            self.override_usecpu.set(config_data.get('usecpu', False))
            self.override_usemmap.set(config_data.get('usemmap', False))
            self.override_usemlock.set(config_data.get('usemlock', False))
            self.override_noavx2.set(config_data.get('noavx2', False))
            self.override_noblas.set(config_data.get('noblas', False))
            self.override_nommap.set(config_data.get('nommap', False))
            self.override_usemirostat.set(config_data.get('usemirostat', False))
            self.override_mirostat_tau.set(config_data.get('mirostat_tau', 5.0))
            self.override_mirostat_eta.set(config_data.get('mirostat_eta', 0.1))
            self.port_var.set(config_data.get('port', 5001))
            self.host_var.set(config_data.get('host', '0.0.0.0'))
            self.launch_var.set(config_data.get('launch', False))
            
            debug_log("Configuration loaded successfully")
            return True
            
        except Exception as e:
            debug_log(f"Error loading configuration: {e}")
            return False

    def update_config_list(self):
        """Update the list of available configuration files."""
        self.config_listbox.delete(0, tk.END)
        self.config_files = []
        
        if not self.config_dir.get() or not os.path.isdir(self.config_dir.get()):
            self.show_error_message(f"Invalid configuration directory: {self.config_dir.get()}")
            self.status_text.set("Please select a valid configuration directory")
            return
        
        try:
            # Get all .kcpps files in the directory
            for file in os.listdir(self.config_dir.get()):
                if file.endswith(".kcpps"):
                    self.config_files.append(file)
                    self.config_listbox.insert(tk.END, file)
            
            if not self.config_files:
                self.show_error_message(f"No .kcpps configuration files found in: {self.config_dir.get()}")
                self.status_text.set("No .kcpps configuration files found")
            else:
                self.status_text.set("Configuration files loaded")
        except Exception as e:
            error_msg = f"Error reading directory: {str(e)}"
            self.show_error_message(error_msg)
            self.status_text.set("Error reading directory")

    def save_current_config(self):
        """Save current configuration to a file"""
        try:
            # Get enabled GPUs
            enabled_gpus = [gpu_id for gpu_id, var in self.gpu_vars.items() if var.get()]
            debug_log(f"Enabled GPUs: {enabled_gpus}")
            
            # Get selected GPU backend
            selected_backend = None
            for backend, var in self.gpu_backend_vars.items():
                if var.get():
                    selected_backend = backend
                    break
            debug_log(f"Selected GPU backend: {selected_backend}")
            
            # Get the current model path
            current_model = self.selected_model.get()
            debug_log(f"Current model path: {current_model}")
            
            # Helper function to convert string to int/float if possible
            def convert_numeric(value):
                try:
                    if isinstance(value, str):
                        if '.' in value:
                            return float(value)
                        return int(value)
                    return value
                except ValueError:
                    return value
            
            config_data = {
                "model": self.model_path.get(),
                "threads": int(self.override_threads.get()) if self.override_threads.get() else 4,
                "threads_batch": int(self.override_threads_batch.get()) if self.override_threads_batch.get() else 4,
                "contextsize": int(self.override_contextsize.get()) if self.override_contextsize.get() else 2048,
                "ropeconfig": self.override_ropeconfig.get(),
                "usemmap": self.override_usemmap.get(),
                "usemlock": self.override_usemlock.get(),
                "noavx2": self.override_noavx2.get(),
                "noblas": self.override_noblas.get(),
                "nommap": self.override_nommap.get(),
                "usemirostat": self.override_usemirostat.get(),
                "mirostat_tau": convert_numeric(self.override_mirostat_tau.get()),
                "mirostat_eta": convert_numeric(self.override_mirostat_eta.get()),
                "temperature": convert_numeric(self.override_temperature.get()),
                "min_p": convert_numeric(self.override_min_p.get()),
                "seed": int(self.override_seed.get()) if self.override_seed.get() else -1,
                "n_predict": int(self.override_n_predict.get()) if self.override_n_predict.get() else -1,
                "ignore_eos": self.override_ignore_eos.get(),
                "ssl": self.override_ssl.get(),
                "nocertify": self.override_nocertify.get(),
                "password": self.override_password.get(),
                "blasbatchsize": int(self.override_blasbatchsize.get()) if self.override_blasbatchsize.get() else 512,
                "blasthreads": int(self.override_blasthreads.get()) if self.override_blasthreads.get() else 4
            }
            
            # Add GPU-specific flags only if they are enabled
            if self.override_flashattention.get():
                config_data["flashattention"] = True
            if self.override_usecublas.get():
                config_data["usecublas"] = True
            if self.override_usevulkan.get():
                config_data["usevulkan"] = True
            if self.override_useclblast.get():
                config_data["useclblast"] = True
            if self.override_usecpu.get():
                config_data["usecpu"] = True
            
            debug_log(f"Configuration data to be saved: {json.dumps(config_data, indent=2)}")
            
            # Ask for filename
            filename = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".kcpps",
                filetypes=[("KoboldCPP Settings", "*.kcpps")],
                initialdir=self.config_dir.get(),
                initialfile="new_config.kcpps"
            )
            
            if not filename:
                return False
                
            # Save to file
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            debug_log(f"Configuration saved to {filename}")
            self.update_config_list()  # Refresh the config list
            return True
            
        except Exception as e:
            debug_log(f"Error saving configuration: {e}")
            return False

    def remove_selected_config(self):
        """Remove the selected configuration file."""
        selection = self.config_listbox.curselection()
        if not selection:
            self.show_error_message("Please select a configuration file to remove")
            return
            
        index = selection[0]
        if 0 <= index < len(self.config_files):
            config_file = self.config_files[index]
            full_path = os.path.join(self.config_dir.get(), config_file)
            
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {config_file}?"):
                try:
                    os.remove(full_path)
                    self.status_text.set(f"Removed configuration: {config_file}")
                    self.update_config_list()
                except Exception as e:
                    self.show_error_message(f"Error removing configuration: {str(e)}")

    def on_model_select(self, event):
        debug_log("Starting model selection process")
        selection = self.model_listbox.curselection()
        if selection:
            index = selection[0]
            debug_log(f"Selected index: {index}")
            if 0 <= index < len(self.model_files):
                selected_file = self.model_files[index]
                debug_log(f"Selected model file: {selected_file}")
                self.selected_model.set(selected_file)
                debug_log(f"Set selected_model to: {selected_file}")
                self.status_text.set(f"Selected model: {selected_file}")
                self.save_settings()
                debug_log("Settings saved after model selection")
            else:
                debug_log(f"Invalid selection index: {index} (max index: {len(self.model_files)-1})")
                self.show_error_message(f"Invalid model selection index: {index}")
        elif not self.selected_model.get():
            # Only show error if there's no current selection at all
            debug_log("No model selected and no previous selection exists")
            self.show_error_message("Please select a model")
    
    def validate_parameters(self):
        """Validate all parameters before launching"""
        try:
            # Validate numeric parameters
            if self.override_threads.get() and self.override_threads.get() != "None":
                threads = int(self.override_threads.get())
                if threads < 1 or threads > self.cpu_info.get('logical_cores', 64):
                    raise ValueError(f"Threads must be between 1 and {self.cpu_info.get('logical_cores', 64)}")
                    
            if self.override_gpu_layers.get() and self.override_gpu_layers.get() != "None":
                try:
                    gpu_layers = int(self.override_gpu_layers.get())
                    if gpu_layers < 0:
                        # If GPU layers is negative, set it to 0 instead of raising an error
                        self.override_gpu_layers.set("0")
                        debug_log("GPU layers was negative, setting to 0")
                except ValueError:
                    # If conversion fails, set to 0
                    self.override_gpu_layers.set("0")
                    debug_log("Invalid GPU layers value, setting to 0")
                    
            # Removed context size validation to allow values up to 132k
                    
            if self.override_blasbatchsize.get() and self.override_blasbatchsize.get() != "None":
                blas_batch = int(self.override_blasbatchsize.get())
                if blas_batch < 1 or blas_batch > 512:
                    raise ValueError("BLAS batch size must be between 1 and 512")
                    
            if self.override_blasthreads.get() and self.override_blasthreads.get() != "None":
                blas_threads = int(self.override_blasthreads.get())
                if blas_threads < 1 or blas_threads > self.cpu_info.get('logical_cores', 64):
                    raise ValueError(f"BLAS threads must be between 1 and {self.cpu_info.get('logical_cores', 64)}")
                    
            if self.override_maxrequestsize.get() and self.override_maxrequestsize.get() != "None":
                max_req = int(self.override_maxrequestsize.get())
                if max_req < 1 or max_req > 1024:
                    raise ValueError("Max request size must be between 1 and 1024 MB")
            
            # Validate forceversion parameter
            if self.override_forceversion.get() and self.override_forceversion.get() != "None":
                try:
                    forceversion = int(self.override_forceversion.get())
                    if forceversion < 0:
                        raise ValueError("Forceversion must be 0 or greater")
                except ValueError:
                    raise ValueError("Forceversion must be a valid integer")
            
            # Validate generation parameters
            if self.override_temperature.get() and self.override_temperature.get() != "None":
                temp = float(self.override_temperature.get())
                if temp < 0.0 or temp > 2.0:
                    raise ValueError("Temperature must be between 0.0 and 2.0")
                    
            if self.override_min_p.get() and self.override_min_p.get() != "None":
                min_p = float(self.override_min_p.get())
                if min_p < 0.0 or min_p > 1.0:
                    raise ValueError("Min P must be between 0.0 and 1.0")
                    
            if self.override_seed.get() and self.override_seed.get() != "None":
                seed = int(self.override_seed.get())
                if seed < -1:
                    raise ValueError("Seed must be -1 or greater")
                    
            if self.override_n_predict.get() and self.override_n_predict.get() != "None":
                n_predict = int(self.override_n_predict.get())
                if n_predict < -1:
                    raise ValueError("Max tokens must be -1 or greater")
                    
            # Validate tensor split format
            if self.override_tensor_split.get() and self.override_tensor_split.get() != "None":
                try:
                    values = [float(x.strip()) for x in self.override_tensor_split.get().split(',')]
                    if not all(0 <= x <= 100 for x in values):
                        raise ValueError("Tensor split values must be between 0 and 100")
                    if abs(sum(values) - 100) > 0.01:  # Allow for floating point imprecision
                        raise ValueError("Tensor split values must sum to 100")
                except ValueError as e:
                    raise ValueError(f"Invalid tensor split format: {str(e)}")
                    
            # Validate rope config format
            if self.rope_config_var.get():
                try:
                    # Remove any brackets and whitespace
                    rope_str = self.rope_config_var.get().strip('[] ')
                    if rope_str:  # Only try to parse if string is not empty
                        values = [float(x.strip()) for x in rope_str.split(',')]
                        if len(values) != 2:
                            raise ValueError("RoPE config must have exactly 2 values")
                except ValueError as e:
                    raise ValueError(f"Invalid RoPE config format: {str(e)}")
            else:
                # Set default values if empty
                self.rope_config_var.set("[10000.0, 1.0]")
                    
            return True
        except ValueError as e:
            # Let the error propagate to the terminal
            raise

    def validate_config_file(self, config_path):
        """Validate the configuration file format and content"""
        print(f"\nDEBUG: Starting config validation for {config_path}")
        try:
            print(f"DEBUG: Reading config file...")
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            print(f"DEBUG: Config data loaded: {json.dumps(config_data, indent=2)}")
                
            # Check required fields
            required_fields = ['model']
            print(f"DEBUG: Checking required fields: {required_fields}")
            for field in required_fields:
                if field not in config_data:
                    print(f"DEBUG: Missing required field: {field}")
                    raise ValueError(f"Missing required field in config: {field}")
            print("DEBUG: All required fields present")
                    
            # Validate field types and ranges
            if 'threads' in config_data and not isinstance(config_data['threads'], (int, str)):
                print(f"DEBUG: Invalid threads type: {type(config_data['threads'])}")
                raise ValueError("Threads must be a number")
                
            if 'gpulayers' in config_data and not isinstance(config_data['gpulayers'], (int, str)):
                print(f"DEBUG: Invalid gpulayers type: {type(config_data['gpulayers'])}")
                raise ValueError("GPU layers must be a number")
                
            if 'contextsize' in config_data and not isinstance(config_data['contextsize'], (int, str)):
                print(f"DEBUG: Invalid contextsize type: {type(config_data['contextsize'])}")
                raise ValueError("Context size must be a number")
                
            # Validate model path exists
            model_path = config_data.get('model')
            print(f"DEBUG: Checking if model exists at path: {model_path}")
            if not os.path.exists(model_path):
                print(f"DEBUG: Model file not found at path: {model_path}")
                raise ValueError(f"Model file not found: {model_path}")
            print(f"DEBUG: Model file found at: {model_path}")
            
            # Add the model's directory to the list of scanned directories if it's not already there
            model_dir = os.path.dirname(os.path.abspath(model_path))
            print(f"DEBUG: Model directory: {model_dir}")
            if model_dir not in self.model_dirs:
                print(f"DEBUG: Adding new model directory to scan list")
                self.model_dirs.append(model_dir)
                self._update_model_dirs_listbox()
                self.update_model_list()
                
            print("DEBUG: Config validation successful")
            return True
        except json.JSONDecodeError:
            print("DEBUG: Invalid JSON format in configuration file")
            raise ValueError("Invalid JSON format in configuration file")
        except Exception as e:
            print(f"DEBUG: Validation error: {str(e)}")
            raise ValueError(f"Configuration validation error: {str(e)}")

    def check_system_resources(self, config_path):
        """Check if system has sufficient resources for the configuration"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Only check if GPU is available if GPU acceleration is requested
            if self.override_gpu_layers.get() or int(config_data.get('gpulayers', 0)) > 0:
                if not self.gpu_info["available"]:
                    raise ValueError("GPU acceleration requested but no GPU detected")
                    
            return True
        except Exception as e:
            raise ValueError(f"System resource check failed: {str(e)}")

    def launch_koboldcpp(self):
        if not self.koboldcpp_path.get():
            self.show_error_message("Please select the KoboldCPP executable")
            return
            
        if not self.selected_config.get():
            self.show_error_message("Please select a configuration file")
            return
            
        if not self.selected_model.get():
            self.show_error_message("Please select a model file")
            return
            
        config_path = os.path.join(self.config_dir.get(), self.selected_config.get())
        if not os.path.exists(config_path):
            self.show_error_message(f"Configuration file not found: {config_path}")
            return
            
        model_path = self.selected_model.get()
        if not os.path.exists(model_path):
            self.show_error_message(f"Model file not found: {model_path}")
            return
            
        # Validate configuration file
        try:
            self.validate_config_file(config_path)
        except ValueError as e:
            self.show_error_message(str(e))
            return
            
        # Check system resources
        try:
            self.check_system_resources(config_path)
        except ValueError as e:
            self.show_error_message(str(e))
            return
            
        # Stop any running instance
        self.stop_koboldcpp()
        
        # Launch nvidia-smi if requested
        if self.launch_nvidia_smi.get():
            try:
                subprocess.Popen(["nvidia-smi", "-l", "1"])
            except Exception as e:
                self.show_error_message(f"Failed to launch nvidia-smi: {str(e)}")
        
        # Start KoboldCPP in a separate thread
        self.running = True
        self.status_text.set("Starting KoboldCPP...")
        launch_thread = threading.Thread(target=self._run_koboldcpp, args=(config_path, model_path))
        launch_thread.daemon = True
        launch_thread.start()
        
    def _run_koboldcpp(self, config_path, model_path):
        try:
            # Build command with config file parameter
            cmd = [self.koboldcpp_path.get(), "--config", config_path]
            
            # Launch process in a new terminal window
            if platform.system() == "Windows":
                # For Windows, use start to open a new cmd window
                cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
                subprocess.Popen(f'start cmd /k {cmd_str}', shell=True)
            else:
                # For Linux/Mac, use xterm or gnome-terminal
                try:
                    subprocess.Popen(['xterm', '-e'] + cmd)
                except FileNotFoundError:
                    try:
                        subprocess.Popen(['gnome-terminal', '--'] + cmd)
                    except FileNotFoundError:
                        # Fallback to running in current terminal
                        subprocess.Popen(cmd)
            
            self.status_text.set("KoboldCPP launched in new terminal")
            
        except Exception as e:
            print(f"Error launching KoboldCPP: {str(e)}", file=sys.stderr)
            self.running = False
            self.status_text.set("Error occurred")
    
    def stop_koboldcpp(self):
        """Stop the running KoboldCPP process"""
        if self.process is not None:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], 
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    
                self.process = None
                self.running = False
                self.status_text.set("KoboldCPP has been stopped")
                
            except Exception as e:
                error_msg = f"Error stopping KoboldCPP: {str(e)}"
                self.show_error_message(error_msg)
                self.status_text.set("Failed to stop KoboldCPP")
    
    def create_launcher_file(self):
        """Create a standalone launcher script for the current configuration, including overrides if set."""
        if not self.koboldcpp_path.get() or not os.path.isfile(self.koboldcpp_path.get()):
            self.show_error_message(f"Invalid KoboldCPP executable: {self.koboldcpp_path.get()}")
            return
        
        if not self.selected_config.get():
            self.show_error_message("Please select a configuration file")
            return
        
        config_path = os.path.join(self.config_dir.get(), self.selected_config.get())
        if not os.path.isfile(config_path):
            self.show_error_message(f"Configuration file not found: {config_path}")
            return
        
        is_windows = platform.system() == "Windows"
        # Ask user which type of script to create
        script_type = messagebox.askquestion(
            "Script Type",
            "Would you like to create a PowerShell (.ps1) script?\nClick 'No' for a Batch (.bat) script.",
            icon='question'
        )
        
        default_ext = ".ps1" if script_type == "yes" else ".bat"
        filename = f"Launch_{self.selected_config.get().split('.')[0]}{default_ext}"
        
        filetypes = [("PowerShell Scripts", "*.ps1")] if script_type == "yes" else [("Batch Files", "*.bat")]
        save_path = filedialog.asksaveasfilename(
            title="Save Launcher Script",
            defaultextension=default_ext,
            filetypes=filetypes,
            initialfile=filename
        )
        
        if not save_path:
            return
        
        try:
            exe_path = self.koboldcpp_path.get()
            exe_dir = os.path.dirname(exe_path)
            exe_name = os.path.basename(exe_path)
            
            # Build the command string starting with the config parameter
            cmd_parts = [f'"{exe_name}"', "--config", f'"{config_path}"']
            
            cmd_string = " ".join(cmd_parts)
            
            # Prepend nvidia-smi command if checkbox is checked
            if self.launch_nvidia_smi.get():
                if script_type == "yes":
                    nvidia_cmd = 'Start-Process nvidia-smi -ArgumentList "-l 1"'
                else:
                    nvidia_cmd = 'start nvidia-smi -l 1'
                cmd_string = nvidia_cmd + "\n" + cmd_string
            
            if script_type == "yes":
                script_content = f"""# PowerShell script to launch KoboldCPP
Set-Location -Path "{exe_dir}"
{cmd_string}
"""
            else:
                script_content = f"""@echo off
cd /d "{exe_dir}"
{cmd_string}
"""
            
            with open(save_path, 'w') as f:
                f.write(script_content)
                
            if script_type == "yes":
                # Set PowerShell script execution policy for the current user
                try:
                    subprocess.run(["powershell", "-Command", f"Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force"], 
                                 creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception as e:
                    self.show_error_message(f"Warning: Could not set PowerShell execution policy: {str(e)}")
                
            self.status_text.set(f"Launcher created: {save_path}")
            self.show_error_message(f"Launcher script created successfully at:\n{save_path}")
            
        except Exception as e:
            error_msg = f"Failed to create launcher: {str(e)}"
            self.show_error_message(error_msg)
            self.status_text.set("Launcher creation failed")
    
    def show_error_message(self, message):
        """Display an error message in the error text area"""
        self.error_display.delete(1.0, tk.END)
        self.error_display.insert(tk.END, message)
        self.error_display.see(1.0)
    
    def copy_error_to_clipboard(self):
        """Copy the contents of the error display to clipboard"""
        error_text = self.error_display.get(1.0, tk.END).strip()
        if error_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(error_text)
            self.status_text.set("Error message copied to clipboard")
    
    def load_settings(self):
        """Load settings from the settings file."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    
                self.koboldcpp_path.set(settings.get('koboldcpp_path', ''))
                self.config_dir.set(settings.get('config_dir', ''))
                self.model_dirs = settings.get('model_dirs', [])
                self._update_model_dirs_listbox()
                
                # Load GPU settings
                gpu_settings = settings.get('gpu_settings', {})
                for gpu_id in self.gpu_selection:
                    if str(gpu_id) in gpu_settings:
                        self.gpu_selection[gpu_id].set(gpu_settings[str(gpu_id)].get('enabled', True))
                
                self.update_model_list()
                self.update_config_list()
                
                # Load custom parameters
                self.custom_params = settings.get('custom_params', [])
                self._update_custom_params_listbox()
                
        except Exception as e:
            if hasattr(self, 'status_text'):
                self.status_text.set(f"Error loading settings: {str(e)}")
    
    def save_settings(self):
        """Save settings to the settings file."""
        try:
            settings = {
                'koboldcpp_path': self.koboldcpp_path.get(),
                'config_dir': self.config_dir.get(),
                'model_dirs': self.model_dirs,
                'gpu_settings': {
                    str(gpu_id): {
                        'enabled': self.gpu_selection[gpu_id].get()
                    }
                    for gpu_id in self.gpu_selection
                },
                'custom_params': self.custom_params
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
                
        except Exception as e:
            self.status_text.set(f"Error saving settings: {str(e)}")
            
    def __del__(self):
        """Cleanup when the app is closed"""
        if hasattr(self, 'process') and self.process is not None:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], 
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    self.process.terminate()
            except Exception:
                pass

    def recommend_parameters(self):
        """Recommend optimal parameters based on system capabilities"""
        recommendations = []
        
        # CPU recommendations
        if self.cpu_info.get('logical_cores'):
            logical_cores = self.cpu_info['logical_cores']
            physical_cores = self.cpu_info['physical_cores']
            
            # Main CPU threads recommendation
            recommended_threads = min(logical_cores, 32)
            if not self.override_threads.get():
                recommendations.append(f"Recommended CPU threads: {recommended_threads} (based on {logical_cores} logical cores)")
            
            # BLAS threads recommendation
            recommended_blas_threads = min(physical_cores, 16)
            if not self.override_blasthreads.get():
                recommendations.append(f"Recommended BLAS threads: {recommended_blas_threads} (based on {physical_cores} physical cores)")
            
            # BLAS batch size recommendation
            if not self.override_blasbatchsize.get():
                if physical_cores >= 8:
                    recommendations.append("Recommended BLAS batch size: 512 (optimal for 8+ core CPUs)")
                elif physical_cores >= 4:
                    recommendations.append("Recommended BLAS batch size: 256 (optimal for 4-7 core CPUs)")
                else:
                    recommendations.append("Recommended BLAS batch size: 128 (optimal for 1-3 core CPUs)")
            
            # Thread usage explanation
            if logical_cores > 32:
                recommendations.append("Note: Using more than 32 threads may not improve performance significantly")
            if logical_cores != physical_cores:
                recommendations.append(f"Note: {logical_cores - physical_cores} threads are hyperthreaded")
                
        # GPU recommendations
        if self.gpu_info["available"] and self.gpu_info["devices"]:
            total_vram = sum(gpu["total_memory_gb"] for gpu in self.gpu_info["devices"])
            recommended_layers = min(int(total_vram * 2), 100)  # Rough estimate
            
            # GPU layers recommendation
            if not self.gpu_layers_override.get():
                recommendations.append(f"Recommended GPU layers: {recommended_layers} (based on {total_vram}GB VRAM)")
            
            # Multi-GPU recommendations
            if len(self.gpu_info["devices"]) > 1:
                # Tensor split recommendation
                split_values = []
                for gpu in self.gpu_info["devices"]:
                    percentage = round((gpu["total_memory_gb"] / total_vram) * 100, 1)
                    split_values.append(str(percentage))
                if not self.override_tensor_split.get():
                    recommendations.append(f"Recommended tensor split: {', '.join(split_values)} (based on VRAM distribution)")
                
                # Multi-GPU thread recommendations
                recommendations.append("For multi-GPU setup:")
                recommendations.append("- Use 4-8 threads per GPU for optimal performance")
                recommendations.append("- Consider using BLAS threads for CPU acceleration")
            
            # Single GPU recommendations
            else:
                gpu = self.gpu_info["devices"][0]
                recommendations.append(f"For {gpu['name']} ({gpu['total_memory_gb']}GB VRAM):")
                recommendations.append("- Use 4-8 threads for optimal performance")
                if gpu["total_memory_gb"] < 8:
                    recommendations.append("- Consider using CPU threads for additional acceleration")
                    
        # Memory recommendations
        if "error" not in self.ram_info:
            available_ram = self.ram_info.get('available_ram_gb', 0)
            recommended_context = min(int(available_ram * 0.5), 32768)  # Rough estimate
            if not self.override_contextsize.get():
                recommendations.append(f"Recommended context size: {recommended_context} (based on {available_ram}GB available RAM)")
                
            # Thread-to-memory ratio recommendations
            if self.cpu_info.get('logical_cores'):
                threads_per_gb = min(self.cpu_info['logical_cores'] / available_ram, 8)
                recommendations.append(f"Recommended threads per GB of RAM: {round(threads_per_gb, 1)}")
                
        # Additional thread-related recommendations
        if self.cpu_info.get('logical_cores'):
            logical_cores = self.cpu_info['logical_cores']
            physical_cores = self.cpu_info['physical_cores']
            
            # TTS thread recommendations
            if not self.override_ttsthreads.get():
                recommended_tts_threads = min(physical_cores, 8)
                recommendations.append(f"Recommended TTS threads: {recommended_tts_threads} (optimal for speech generation)")
            
            # SD thread recommendations
            if not self.override_threads.get():
                recommended_sd_threads = min(physical_cores, 12)
                recommendations.append(f"Recommended SD threads: {recommended_sd_threads} (optimal for image generation)")
            
            # Thread distribution recommendations
            recommendations.append("Thread Distribution Guidelines:")
            recommendations.append("- Main threads: 4-8 per GPU (or 8-16 for CPU-only)")
            recommendations.append("- BLAS threads: 4-8 for CPU acceleration")
            recommendations.append("- TTS threads: 4-8 for speech generation")
            recommendations.append("- SD threads: 8-12 for image generation")
            
            # Performance optimization tips
            recommendations.append("Performance Tips:")
            recommendations.append("- For CPU-only: Use more threads (up to 32) with lower batch sizes")
            recommendations.append("- For GPU: Use fewer threads (4-8) with higher batch sizes")
            recommendations.append("- For mixed CPU/GPU: Balance threads between CPU and GPU workloads")
            
        return recommendations

    def browse_executable(self):
        """Browse for the KoboldCPP executable file."""
        filetypes = [("Executable files", "*.exe"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            title="Select KoboldCPP Executable",
            filetypes=filetypes,
            initialdir=os.path.dirname(self.koboldcpp_path.get()) if self.koboldcpp_path.get() else None
        )
        
        if filepath:
            self.koboldcpp_path.set(filepath)
            self.save_settings()
            self.status_text.set(f"KoboldCPP executable set to: {filepath}")

    def browse_config_dir(self):
        """Browse for the configuration files directory."""
        directory = filedialog.askdirectory(
            title="Select Configuration Files Directory",
            initialdir=self.config_dir.get() if self.config_dir.get() else None
        )
        
        if directory:
            self.config_dir.set(directory)
            self.update_config_list()
            self.save_settings()
            self.status_text.set(f"Configuration directory set to: {directory}")

    def _create_basic_parameters(self, parent):
        """Create basic parameter controls"""
        frame = ttk.LabelFrame(parent, text="Basic Parameters", padding="5")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Threads
        threads_frame = ttk.Frame(frame)
        threads_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(threads_frame, text="Threads:").pack(side="left")
        threads_entry = ttk.Entry(threads_frame, textvariable=self.threads_var, width=10)
        threads_entry.pack(side="left", padx=5)
        ttk.Label(threads_frame, textvariable=self.recommended_threads_var).pack(side="left")
        
        # Threads Batch
        threads_batch_frame = ttk.Frame(frame)
        threads_batch_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(threads_batch_frame, text="Threads Batch:").pack(side="left")
        threads_batch_entry = ttk.Entry(threads_batch_frame, textvariable=self.threads_batch_var, width=10)
        threads_batch_entry.pack(side="left", padx=5)
        ttk.Label(threads_batch_frame, textvariable=self.recommended_threads_batch_var).pack(side="left")
        
        # Generation Parameters
        gen_frame = ttk.LabelFrame(frame, text="Generation Parameters", padding="5")
        gen_frame.pack(fill="x", padx=5, pady=2)
        
        # Temperature
        temp_frame = ttk.Frame(gen_frame)
        temp_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(temp_frame, text="Temperature:").pack(side="left")
        temp_entry = ttk.Entry(temp_frame, textvariable=self.override_temperature, width=10)
        temp_entry.pack(side="left", padx=5)
        ttk.Label(temp_frame, text="Controls randomness (default: 0.8)").pack(side="left")
        
        # Min P
        minp_frame = ttk.Frame(gen_frame)
        minp_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(minp_frame, text="Min P:").pack(side="left")
        minp_entry = ttk.Entry(minp_frame, textvariable=self.override_min_p, width=10)
        minp_entry.pack(side="left", padx=5)
        ttk.Label(minp_frame, text="Minimum probability sampling (default: 0.05)").pack(side="left")
        
        # Seed
        seed_frame = ttk.Frame(gen_frame)
        seed_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(seed_frame, text="Seed:").pack(side="left")
        seed_entry = ttk.Entry(seed_frame, textvariable=self.override_seed, width=10)
        seed_entry.pack(side="left", padx=5)
        ttk.Label(seed_frame, text="RNG seed (-1 for random)").pack(side="left")
        
        # N Predict
        npred_frame = ttk.Frame(gen_frame)
        npred_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(npred_frame, text="Max Tokens:").pack(side="left")
        npred_entry = ttk.Entry(npred_frame, textvariable=self.override_n_predict, width=10)
        npred_entry.pack(side="left", padx=5)
        ttk.Label(npred_frame, text="Maximum tokens to generate (-1 for unlimited)").pack(side="left")
        
        # Ignore EOS
        eos_frame = ttk.Frame(gen_frame)
        eos_frame.pack(fill="x", padx=5, pady=2)
        ttk.Checkbutton(eos_frame, text="Ignore EOS", variable=self.override_ignore_eos).pack(side="left")
        ttk.Label(eos_frame, text="Ignore end-of-sequence tokens").pack(side="left")
        
        # Context Size
        context_frame = ttk.LabelFrame(frame, text="Context Size", padding="5")
        context_frame.pack(fill="x", padx=5, pady=2)
        
        # Create a frame for the slider and entry
        context_control_frame = ttk.Frame(context_frame)
        context_control_frame.pack(fill="x", padx=5, pady=2)
        
        # Entry for direct input
        ttk.Label(context_control_frame, text="Size:").pack(side="left")
        context_entry = ttk.Entry(context_control_frame, textvariable=self.override_contextsize, width=10)
        context_entry.pack(side="left", padx=5)
        
        # Add trace to update slider when entry changes
        self.override_contextsize.trace_add("write", self._update_context_slider)
        
        # Slider for context size
        self.context_slider_var = tk.IntVar(value=2048)
        context_slider = ttk.Scale(context_control_frame, from_=1024, to=135168,  # 132k
                                  orient="horizontal", variable=self.context_slider_var,
                                  command=self._update_context_size)
        context_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        # Label showing current value
        self.context_value_label = ttk.Label(context_control_frame, text="2048")
        self.context_value_label.pack(side="left", padx=5)
        
        # Set initial value
        if self.override_contextsize.get():
            try:
                value = int(self.override_contextsize.get())
                self.context_slider_var.set(value)
                self.context_value_label.config(text=str(value))
            except ValueError:
                pass
        
        return frame

    def _update_context_size(self, value):
        """Update context size when slider moves"""
        # No rounding to 1024 increments
        value = int(float(value))
        self.context_slider_var.set(value)
        self.override_contextsize.set(str(value))
        self.context_value_label.config(text=str(value))

    def _update_context_slider(self, *args):
        """Update slider when entry value changes"""
        try:
            value = int(self.override_contextsize.get())
            if 1024 <= value <= 135168:  # 132k
                self.context_slider_var.set(value)
                self.context_value_label.config(text=str(value))
        except ValueError:
            pass

    def _create_gpu_parameters(self, parent):
        debug_log("Creating GPU parameters section")
        gpu_frame = ttk.LabelFrame(parent, text="GPU Selection")
        gpu_frame.pack(fill="x", padx=5, pady=5)
        
        # Add GPU backend selection
        backend_frame = ttk.Frame(gpu_frame)
        backend_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(backend_frame, text="GPU Backend:").pack(side="left")
        
        # Only check CUDA if available, uncheck all others
        self.gpu_backend_vars = {
            "CUDA": tk.BooleanVar(value=TORCH_AVAILABLE),
            "CLBlast": tk.BooleanVar(value=False),
            "Vulkan": tk.BooleanVar(value=False),
            "Metal": tk.BooleanVar(value=False)
        }
        
        for backend, var in self.gpu_backend_vars.items():
            cb = ttk.Checkbutton(backend_frame, text=backend, variable=var,
                                command=lambda b=backend: self._update_gpu_backend(b))
            cb.pack(side="left", padx=5)
            debug_log(f"Created {backend} checkbox with initial value: {var.get()}")
        
        # Add GPU selection
        selection_frame = ttk.Frame(gpu_frame)
        selection_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(selection_frame, text="Select GPUs:").pack(side="left")
        
        # Add "All GPUs" checkbox
        self.use_all_gpus = tk.BooleanVar(value=False)
        all_gpus_cb = ttk.Checkbutton(selection_frame, text="All GPUs", 
                                     variable=self.use_all_gpus,
                                     command=self._toggle_all_gpus)
        all_gpus_cb.pack(side="left", padx=5)
        
        # Add individual GPU checkboxes
        self.gpu_vars = {}
        for gpu in self.gpu_info['devices']:
            var = tk.BooleanVar(value=False)
            self.gpu_vars[gpu['id']] = var
            cb = ttk.Checkbutton(selection_frame, 
                                text=f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']}GB)",
                                variable=var,
                                command=lambda gpu_id=gpu['id']: self._update_gpu_selection(gpu_id))
            cb.pack(side="left", padx=5)
            debug_log(f"Created GPU {gpu['id']} checkbox with initial value: {var.get()}")
        
        # Add GPU layers control
        layers_frame = ttk.LabelFrame(gpu_frame, text="GPU Layers", padding="5")
        layers_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a frame for the slider and entry
        layers_control_frame = ttk.Frame(layers_frame)
        layers_control_frame.pack(fill="x", padx=5, pady=2)
        
        # Entry for direct input
        ttk.Label(layers_control_frame, text="Layers:").pack(side="left")
        layers_entry = ttk.Entry(layers_control_frame, textvariable=self.override_gpu_layers, width=10)
        layers_entry.pack(side="left", padx=5)
        
        # Add trace to update slider when entry changes
        self.override_gpu_layers.trace_add("write", self._update_gpu_layers_slider)
        
        # Slider for GPU layers
        self.gpu_layers_slider_var = tk.IntVar(value=0)
        layers_slider = ttk.Scale(layers_control_frame, from_=0, to=100,
                                orient="horizontal", variable=self.gpu_layers_slider_var,
                                command=self._update_gpu_layers)
        layers_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        # Label showing current value
        self.gpu_layers_value_label = ttk.Label(layers_control_frame, text="0")
        self.gpu_layers_value_label.pack(side="left", padx=5)
        
        # Set initial value
        if self.override_gpu_layers.get():
            try:
                value = int(self.override_gpu_layers.get())
                self.gpu_layers_slider_var.set(value)
                self.gpu_layers_value_label.config(text=str(value))
            except ValueError:
                pass
        
        # Add tensor split control
        tensor_split_frame = ttk.LabelFrame(gpu_frame, text="Tensor Split", padding="5")
        tensor_split_frame.pack(fill="x", padx=5, pady=5)
        
        tensor_split_control_frame = ttk.Frame(tensor_split_frame)
        tensor_split_control_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(tensor_split_control_frame, text="Split:").pack(side="left")
        tensor_split_entry = ttk.Entry(tensor_split_control_frame, textvariable=self.override_tensor_split, width=30)
        tensor_split_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        return gpu_frame

    def _update_gpu_backend(self, backend):
        """Update GPU backend selection"""
        debug_log(f"GPU backend {backend} selection changed")
        # Ensure only one backend is selected at a time
        for b, var in self.gpu_backend_vars.items():
            if b != backend:
                var.set(False)
        debug_log("Current GPU backend state:")
        for b, var in self.gpu_backend_vars.items():
            debug_log(f"{b}: {var.get()}")

    def save_current_config(self):
        """Save current configuration to a file"""
        try:
            # Get enabled GPUs
            enabled_gpus = [gpu_id for gpu_id, var in self.gpu_vars.items() if var.get()]
            debug_log(f"Enabled GPUs: {enabled_gpus}")
            
            # Get selected GPU backend
            selected_backend = None
            for backend, var in self.gpu_backend_vars.items():
                if var.get():
                    selected_backend = backend
                    break
            debug_log(f"Selected GPU backend: {selected_backend}")
            
            # Get the current model path
            current_model = self.selected_model.get()
            debug_log(f"Current model path: {current_model}")
            
            # Helper function to convert string to int/float if possible
            def convert_numeric(value):
                try:
                    if isinstance(value, str):
                        if '.' in value:
                            return float(value)
                        return int(value)
                    return value
                except ValueError:
                    return value
            
            config_data = {
                "model": self.model_path.get(),
                "threads": int(self.override_threads.get()) if self.override_threads.get() else 4,
                "threads_batch": int(self.override_threads_batch.get()) if self.override_threads_batch.get() else 4,
                "contextsize": int(self.override_contextsize.get()) if self.override_contextsize.get() else 2048,
                "ropeconfig": self.override_ropeconfig.get(),
                "usemmap": self.override_usemmap.get(),
                "usemlock": self.override_usemlock.get(),
                "noavx2": self.override_noavx2.get(),
                "noblas": self.override_noblas.get(),
                "nommap": self.override_nommap.get(),
                "usemirostat": self.override_usemirostat.get(),
                "mirostat_tau": convert_numeric(self.override_mirostat_tau.get()),
                "mirostat_eta": convert_numeric(self.override_mirostat_eta.get()),
                "temperature": convert_numeric(self.override_temperature.get()),
                "min_p": convert_numeric(self.override_min_p.get()),
                "seed": int(self.override_seed.get()) if self.override_seed.get() else -1,
                "n_predict": int(self.override_n_predict.get()) if self.override_n_predict.get() else -1,
                "ignore_eos": self.override_ignore_eos.get(),
                "ssl": self.override_ssl.get(),
                "nocertify": self.override_nocertify.get(),
                "password": self.override_password.get(),
                "blasbatchsize": int(self.override_blasbatchsize.get()) if self.override_blasbatchsize.get() else 512,
                "blasthreads": int(self.override_blasthreads.get()) if self.override_blasthreads.get() else 4
            }
            
            # Add GPU-specific flags only if they are enabled
            if self.override_flashattention.get():
                config_data["flashattention"] = True
            if self.override_usecublas.get():
                config_data["usecublas"] = True
            if self.override_usevulkan.get():
                config_data["usevulkan"] = True
            if self.override_useclblast.get():
                config_data["useclblast"] = True
            if self.override_usecpu.get():
                config_data["usecpu"] = True
            
            debug_log(f"Configuration data to be saved: {json.dumps(config_data, indent=2)}")
            
            # Ask for filename
            filename = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".kcpps",
                filetypes=[("KoboldCPP Settings", "*.kcpps")],
                initialdir=self.config_dir.get(),
                initialfile="new_config.kcpps"
            )
            
            if not filename:
                return False
                
            # Save to file
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            debug_log(f"Configuration saved to {filename}")
            self.update_config_list()  # Refresh the config list
            return True
            
        except Exception as e:
            debug_log(f"Error saving configuration: {e}")
            return False

    def _create_memory_parameters(self, parent):
        """Create memory parameter controls"""
        frame = ttk.LabelFrame(parent, text="Memory Parameters", padding="5")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Memory Mapping
        mmap_frame = ttk.Frame(frame)
        mmap_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(mmap_frame, text="Use Memory Mapping", variable=self.override_usemmap).pack(anchor="w")
        
        # Memory Lock
        mlock_frame = ttk.Frame(frame)
        mlock_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(mlock_frame, text="Use Memory Lock", variable=self.override_usemlock).pack(anchor="w")
        
        # BLAS Batch Size
        blas_batch_frame = ttk.Frame(frame)
        blas_batch_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(blas_batch_frame, text="BLAS Batch Size:").pack(side="left")
        blas_batch_entry = ttk.Entry(blas_batch_frame, textvariable=self.override_blasbatchsize, width=10)
        blas_batch_entry.pack(side="left", padx=5)
        
        # BLAS Threads
        blas_threads_frame = ttk.Frame(frame)
        blas_threads_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(blas_threads_frame, text="BLAS Threads:").pack(side="left")
        blas_threads_entry = ttk.Entry(blas_threads_frame, textvariable=self.override_blasthreads, width=10)
        blas_threads_entry.pack(side="left", padx=5)
        
        return frame

    def _create_security_parameters(self, parent):
        """Create security parameter controls"""
        frame = ttk.LabelFrame(parent, text="Security Parameters", padding="5")
        frame.pack(fill="x", padx=5, pady=5)
        
        # SSL
        ssl_frame = ttk.Frame(frame)
        ssl_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(ssl_frame, text="Enable SSL", variable=self.override_ssl).pack(anchor="w")
        
        # Password
        password_frame = ttk.Frame(frame)
        password_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(password_frame, text="Password:").pack(side="left")
        password_entry = ttk.Entry(password_frame, textvariable=self.override_password, show="*", width=20)
        password_entry.pack(side="left", padx=5)
        
        # No Certificate Verification
        nocert_frame = ttk.Frame(frame)
        nocert_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Checkbutton(nocert_frame, text="Disable Certificate Verification", variable=self.override_nocertify).pack(anchor="w")
        
        return frame

    def _create_advanced_parameters(self, parent):
        """Create advanced parameter controls"""
        frame = ttk.LabelFrame(parent, text="Advanced Parameters", padding="5")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Custom Parameters Frame
        custom_frame = ttk.LabelFrame(frame, text="Custom Parameters", padding="5")
        custom_frame.pack(fill="x", padx=5, pady=5)
        
        # Instructions
        instructions = "Enter custom parameters in the format: '--parameter value' or '--flag'\nExample: '--usecublas normal' or '--launch'"
        ttk.Label(custom_frame, text=instructions, wraplength=400).pack(padx=5, pady=2)
        
        # Listbox for custom parameters
        list_container = ttk.Frame(custom_frame)
        list_container.pack(fill="x", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.custom_params_listbox = tk.Listbox(
            list_container,
            height=4,
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            bg="white",
            bd=1,
            relief=tk.SOLID
        )
        self.custom_params_listbox.pack(side="left", fill="x", expand=True)
        scrollbar.config(command=self.custom_params_listbox.yview)
        
        # Input frame for new parameter
        input_frame = ttk.Frame(custom_frame)
        input_frame.pack(fill="x", pady=5)
        
        ttk.Label(input_frame, text="Parameter:").pack(side="left", padx=5)
        self.custom_param_entry = ttk.Entry(input_frame, width=40)
        self.custom_param_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(custom_frame)
        buttons_frame.pack(fill="x", pady=5)
        
        # Add button
        add_btn = tk.Button(
            buttons_frame,
            text="Add Parameter",
            command=self.add_custom_parameter,
            bg=self.button_color,
            fg="white",
            activebackground=self.hover_color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=10
        )
        add_btn.pack(side="left", padx=5)
        
        # Remove button
        remove_btn = tk.Button(
            buttons_frame,
            text="Remove Parameter",
            command=self.remove_custom_parameter,
            bg=self.button_color,
            fg="white",
            activebackground=self.hover_color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=10
        )
        remove_btn.pack(side="left", padx=5)
        
        # Update listbox with saved parameters
        self._update_custom_params_listbox()
        
        return frame

    def add_custom_parameter(self):
        """Add a new custom parameter"""
        param = self.custom_param_entry.get().strip()
        if not param:
            messagebox.showerror("Error", "Please enter a parameter")
            return
            
        # Validate parameter format
        if not param.startswith("--"):
            messagebox.showerror("Error", "Parameter must start with '--'")
            return
            
        # Add new parameter
        self.custom_params.append(param)
        self._update_custom_params_listbox()
        self.save_settings()
        self.custom_param_entry.delete(0, tk.END)
        self.status_text.set(f"Added parameter: {param}")

    def remove_custom_parameter(self):
        """Remove selected custom parameter"""
        selection = self.custom_params_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a parameter to remove")
            return
        
        index = selection[0]
        if 0 <= index < len(self.custom_params):
            removed_param = self.custom_params.pop(index)
            self._update_custom_params_listbox()
            self.save_settings()
            self.status_text.set(f"Removed parameter: {removed_param}")

    def _update_custom_params_listbox(self):
        """Update the custom parameters listbox"""
        self.custom_params_listbox.delete(0, tk.END)
        for param in self.custom_params:
            self.custom_params_listbox.insert(tk.END, param)

    def _create_monitoring_widgets(self, parent):
        """Create monitoring widgets for performance tracking"""
        # Performance Metrics Frame
        metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding="5")
        metrics_frame.pack(fill="x", padx=5, pady=5)
        
        # CPU Usage
        cpu_frame = ttk.Frame(metrics_frame)
        cpu_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(cpu_frame, text="CPU Usage:").pack(side="left")
        self.cpu_usage_var = tk.StringVar(value="0%")
        ttk.Label(cpu_frame, textvariable=self.cpu_usage_var).pack(side="left", padx=5)
        
        # Memory Usage
        memory_frame = ttk.Frame(metrics_frame)
        memory_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(memory_frame, text="Memory Usage:").pack(side="left")
        self.memory_usage_var = tk.StringVar(value="0%")
        ttk.Label(memory_frame, textvariable=self.memory_usage_var).pack(side="left", padx=5)
        
        # GPU Usage (if available)
        if self.gpu_info["available"] and self.gpu_info["devices"]:
            gpu_frame = ttk.LabelFrame(metrics_frame, text="GPU Usage", padding="5")
            gpu_frame.pack(fill="x", padx=5, pady=2)
            
            self.gpu_usage_vars = {}
            for gpu in self.gpu_info["devices"]:
                gpu_row = ttk.Frame(gpu_frame)
                gpu_row.pack(fill="x", padx=5, pady=2)
                
                ttk.Label(gpu_row, text=f"GPU {gpu['id']}:").pack(side="left")
                self.gpu_usage_vars[gpu['id']] = tk.StringVar(value="0%")
                ttk.Label(gpu_row, textvariable=self.gpu_usage_vars[gpu['id']]).pack(side="left", padx=5)
        
        # Process Status
        status_frame = ttk.LabelFrame(parent, text="Process Status", padding="5")
        status_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side="left")
        self.process_status_var = tk.StringVar(value="Not Running")
        ttk.Label(status_frame, textvariable=self.process_status_var).pack(side="left", padx=5)
        
        # Start monitoring button
        monitor_btn = tk.Button(parent, text="Start Monitoring", 
                               command=self.start_monitoring,
                               bg=self.button_color, fg="white",
                               activebackground=self.hover_color, activeforeground="white",
                               relief=tk.FLAT, padx=10)
        monitor_btn.pack(pady=5)
        
        # Stop monitoring button
        stop_monitor_btn = tk.Button(parent, text="Stop Monitoring", 
                                    command=self.stop_monitoring,
                                    bg=self.button_color, fg="white",
                                    activebackground=self.hover_color, activeforeground="white",
                                    relief=tk.FLAT, padx=10)
        stop_monitor_btn.pack(pady=5)

    def start_monitoring(self):
        """Start the monitoring process"""
        if not hasattr(self, 'monitoring_thread') or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitor_performance)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.process_status_var.set("Monitoring Started")

    def stop_monitoring(self):
        """Stop the monitoring process"""
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread = None
            self.process_status_var.set("Monitoring Stopped")

    def _monitor_performance(self):
        """Monitor system performance metrics"""
        while hasattr(self, 'monitoring_thread') and self.monitoring_thread and self.monitoring_thread.is_alive():
            try:
                # CPU Usage
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage_var.set(f"{cpu_percent}%")
                
                # Memory Usage
                if PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    self.memory_usage_var.set(f"{memory_percent}%")
                
                # GPU Usage (if available)
                if self.gpu_info["available"] and self.gpu_info["devices"]:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        for gpu in gpus:
                            if gpu.id in self.gpu_usage_vars:
                                self.gpu_usage_vars[gpu.id].set(f"{gpu.load*100:.1f}%")
                    except ImportError:
                        pass
                
                # Process Status
                if self.process and self.process.poll() is None:
                    self.process_status_var.set("Running")
                else:
                    self.process_status_var.set("Not Running")
                
                time.sleep(1)
            except Exception as e:
                self.process_status_var.set(f"Error: {str(e)}")
                time.sleep(5)

    def _create_settings_widgets(self, parent):
        """Create application settings widgets"""
        # Theme Settings
        theme_frame = ttk.LabelFrame(parent, text="Theme Settings", padding="5")
        theme_frame.pack(fill="x", padx=5, pady=5)
        
        # Theme Selection
        theme_label = ttk.Label(theme_frame, text="Theme:")
        theme_label.pack(side="left", padx=5)
        
        theme_var = tk.StringVar(value="light")
        theme_combo = ttk.Combobox(theme_frame, textvariable=theme_var, 
                                  values=["light", "dark"], state="readonly", width=10)
        theme_combo.pack(side="left", padx=5)
        
        # Auto-save Settings
        auto_save_frame = ttk.LabelFrame(parent, text="Auto-save Settings", padding="5")
        auto_save_frame.pack(fill="x", padx=5, pady=5)
        
        self.auto_save_var = tk.BooleanVar(value=True)
        auto_save_check = ttk.Checkbutton(auto_save_frame, text="Auto-save settings on exit",
                                         variable=self.auto_save_var)
        auto_save_check.pack(anchor="w", padx=5)
        
        # Default Settings
        defaults_frame = ttk.LabelFrame(parent, text="Default Settings", padding="5")
        defaults_frame.pack(fill="x", padx=5, pady=5)
        
        # Save current settings as defaults button
        save_defaults_btn = tk.Button(defaults_frame, text="Save Current Settings as Defaults",
                                     command=self.save_current_as_defaults,
                                     bg=self.button_color, fg="white",
                                     activebackground=self.hover_color, activeforeground="white",
                                     relief=tk.FLAT, padx=10)
        save_defaults_btn.pack(pady=5)
        
        # Reset to defaults button
        reset_defaults_btn = tk.Button(defaults_frame, text="Reset to Default Settings",
                                      command=self.reset_to_defaults,
                                      bg=self.button_color, fg="white",
                                      activebackground=self.hover_color, activeforeground="white",
                                      relief=tk.FLAT, padx=10)
        reset_defaults_btn.pack(pady=5)
        
        # About Section
        about_frame = ttk.LabelFrame(parent, text="About", padding="5")
        about_frame.pack(fill="x", padx=5, pady=5)
        
        about_text = """KoboldCPP Launcher
Version 1.0
A GUI launcher for KoboldCPP
"""
        ttk.Label(about_frame, text=about_text, justify="left").pack(padx=5, pady=5)

    def save_current_as_defaults(self):
        """Save current settings as default settings"""
        try:
            # Create a copy of current settings
            defaults = {
                'koboldcpp_path': self.koboldcpp_path.get(),
                'config_dir': self.config_dir.get(),
                'model_dirs': self.model_dirs.copy(),
                'gpu_settings': {
                    str(gpu_id): {
                        'enabled': self.gpu_selection[gpu_id].get()
                    }
                    for gpu_id in self.gpu_selection
                },
                'custom_params': self.custom_params.copy()
            }
            
            # Save to defaults file
            defaults_file = Path(os.path.expanduser("~")) / ".koboldcpp_launcher_defaults.json"
            with open(defaults_file, 'w') as f:
                json.dump(defaults, f, indent=4)
                
            self.status_text.set("Current settings saved as defaults")
        except Exception as e:
            self.show_error_message(f"Error saving defaults: {str(e)}")

    def reset_to_defaults(self):
        """Reset settings to default values"""
        try:
            defaults_file = Path(os.path.expanduser("~")) / ".koboldcpp_launcher_defaults.json"
            if defaults_file.exists():
                with open(defaults_file, 'r') as f:
                    defaults = json.load(f)
                    
                # Apply defaults
                self.koboldcpp_path.set(defaults.get('koboldcpp_path', ''))
                self.config_dir.set(defaults.get('config_dir', ''))
                self.model_dirs = defaults.get('model_dirs', [])
                self._update_model_dirs_listbox()
                
                # Apply GPU settings
                gpu_settings = defaults.get('gpu_settings', {})
                for gpu_id in self.gpu_selection:
                    if str(gpu_id) in gpu_settings:
                        self.gpu_selection[gpu_id].set(gpu_settings[str(gpu_id)].get('enabled', True))
                
                # Apply custom parameters
                self.custom_params = defaults.get('custom_params', [])
                self._update_custom_params_listbox()
                
                self.status_text.set("Settings reset to defaults")
            else:
                self.show_error_message("No default settings found")
        except Exception as e:
            self.show_error_message(f"Error resetting to defaults: {str(e)}")

    def _toggle_all_gpus(self):
        """Toggle all GPU checkboxes based on the 'All GPUs' checkbox"""
        all_enabled = self.use_all_gpus.get()
        for gpu_id in self.gpu_selection:
            self.gpu_selection[gpu_id].set(all_enabled)
    
    def _update_gpu_selection(self, gpu_id=None):
        """Update the 'All GPUs' checkbox based on individual GPU selections"""
        if gpu_id is not None:
            debug_log(f"GPU {gpu_id} selection changed")
        all_selected = all(var.get() for var in self.gpu_vars.values())
        self.use_all_gpus.set(all_selected)
        debug_log(f"All GPUs checkbox set to: {all_selected}")
        debug_log("Current GPU selection state:")
        for gpu_id, var in self.gpu_vars.items():
            debug_log(f"GPU {gpu_id}: {var.get()}")

    def _update_gpu_layers(self, value):
        """Update GPU layers when slider moves"""
        value = int(float(value))
        self.gpu_layers_slider_var.set(value)
        self.override_gpu_layers.set(str(value))
        self.gpu_layers_value_label.config(text=str(value))
        debug_log(f"GPU layers updated to: {value}")

    def _update_gpu_layers_slider(self, *args):
        """Update slider when entry value changes"""
        try:
            value = int(self.override_gpu_layers.get())
            if 0 <= value <= 100:
                self.gpu_layers_slider_var.set(value)
                self.gpu_layers_value_label.config(text=str(value))
                debug_log(f"GPU layers slider updated to: {value}")
        except ValueError:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#f5f5f5")
    app = KoboldCPPLauncher(root)
    
    try:
        if getattr(sys, 'frozen', False):
            app_path = sys._MEIPASS
        else:
            app_path = os.path.dirname(os.path.abspath(__file__))
            
        icon_path = os.path.join(app_path, "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass
    
    def on_closing():
        if hasattr(app, 'process') and app.process is not None:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(app.process.pid)], 
                                     creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    app.process.terminate()
                app.process = None
                app.running = False
            except Exception:
                pass
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()