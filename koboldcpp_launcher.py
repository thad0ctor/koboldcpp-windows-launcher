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

class KoboldCPPLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Koboldcpp Launcher")
        self.root.geometry("700x1000")  # Increased height for error display
        self.root.minsize(600, 900)
        
        # Set app theme colors
        self.bg_color = "#f5f5f5"
        self.accent_color = "#4a6ba0"
        self.button_color = "#5f87c7"
        self.hover_color = "#3a5a8f"
        self.error_color = "#f44336"
        
        # Variables
        self.koboldcpp_path = tk.StringVar()
        self.config_dir = tk.StringVar()
        self.selected_config = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        self.error_text = tk.StringVar()
        self.config_files = []
        self.process = None
        self.running = False
        self.settings_file = Path(os.path.expanduser("~")) / ".koboldcpp_launcher_settings.json"
        
        # Override variables (for custom launch parameters)
        self.override_threads = tk.StringVar()
        self.override_gpu_layers = tk.StringVar()
        self.override_tensor_split = tk.StringVar()
        self.override_contextsize = tk.StringVar()  # New context override
        self.override_flashattention_var = tk.BooleanVar()
        
        # New variable for launching nvidia-smi
        self.launch_nvidia_smi = tk.BooleanVar()
        
        # Defaults loaded from config file (if available)
        self.default_threads = ""
        self.default_gpu_layers = ""
        self.default_tensor_split = ""
        self.default_contextsize = ""
        self.default_flashattention = False
        
        # Load settings
        self.load_settings()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding=15)
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("Status.TLabel", font=("Segoe UI", 9))
        
        # Create widgets
        self.create_widgets()
        
        # Set up grid (we now have 11 rows: 0..10)
        self.main_frame.columnconfigure(0, weight=1)
        for i in range(11):
            self.main_frame.rowconfigure(i, weight=1)
        
        # Update config list if directory is set
        if self.config_dir.get():
            self.update_config_list()
            
    def create_widgets(self):
        # Title
        title_label = ttk.Label(self.main_frame, text="KoboldCPP Launcher", style="Header.TLabel")
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 15))
        
        # New checkbox at top right for "Launch nvidia-smi"
        nvidia_checkbox = ttk.Checkbutton(self.main_frame, text="Launch nvidia-smi", variable=self.launch_nvidia_smi)
        nvidia_checkbox.grid(row=0, column=0, sticky="e", pady=(0, 15))
        
        # KoboldCPP Executable Section
        exe_frame = ttk.LabelFrame(self.main_frame, text="KoboldCPP Executable", padding=10)
        exe_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        exe_path_entry = ttk.Entry(exe_frame, textvariable=self.koboldcpp_path, width=50)
        exe_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        exe_browse_btn = tk.Button(exe_frame, text="Browse", 
                                   command=self.browse_executable,
                                   bg=self.button_color, fg="white",
                                   activebackground=self.hover_color, activeforeground="white",
                                   relief=tk.FLAT, padx=10)
        exe_browse_btn.pack(side=tk.RIGHT)
        
        # Config Directory Section
        config_dir_frame = ttk.LabelFrame(self.main_frame, text="Configuration Files Directory", padding=10)
        config_dir_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        config_dir_entry = ttk.Entry(config_dir_frame, textvariable=self.config_dir, width=50)
        config_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        config_dir_browse_btn = tk.Button(config_dir_frame, text="Browse", 
                                          command=self.browse_config_dir,
                                          bg=self.button_color, fg="white",
                                          activebackground=self.hover_color, activeforeground="white",
                                          relief=tk.FLAT, padx=10)
        config_dir_browse_btn.pack(side=tk.RIGHT)
        
        # Config Files List
        config_list_frame = ttk.LabelFrame(self.main_frame, text="Available Configuration Files", padding=10)
        config_list_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10), rowspan=3)
        config_list_frame.columnconfigure(0, weight=1)
        config_list_frame.rowconfigure(0, weight=1)
        
        # Create a Frame for the listbox and scrollbar
        list_container = ttk.Frame(config_list_frame)
        list_container.grid(row=0, column=0, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Listbox with config files
        self.config_listbox = tk.Listbox(list_container, height=10, 
                                         selectmode=tk.SINGLE,
                                         yscrollcommand=scrollbar.set,
                                         bg="white", bd=1, relief=tk.SOLID)
        self.config_listbox.grid(row=0, column=0, sticky="nsew")
        self.config_listbox.bind('<<ListboxSelect>>', self.on_config_select)
        
        scrollbar.config(command=self.config_listbox.yview)
        
        # Refresh button for config list
        refresh_btn = tk.Button(config_list_frame, text="Refresh List", 
                                command=self.update_config_list,
                                bg=self.button_color, fg="white",
                                activebackground=self.hover_color, activeforeground="white",
                                relief=tk.FLAT, padx=10)
        refresh_btn.grid(row=1, column=0, sticky="e", pady=(5, 0))
        
        # --- New Overrides Section ---
        overrides_frame = ttk.LabelFrame(self.main_frame, text="Overrides", padding=10)
        overrides_frame.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        
        # Row 0 of Overrides: Threads and GPU Layers
        threads_label = ttk.Label(overrides_frame, text="Threads:")
        threads_label.grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        threads_entry = ttk.Entry(overrides_frame, textvariable=self.override_threads, width=10)
        threads_entry.grid(row=0, column=1, sticky="w", padx=(0, 15), pady=2)
        
        gpulayers_label = ttk.Label(overrides_frame, text="GPU Layers:")
        gpulayers_label.grid(row=0, column=2, sticky="w", padx=(0, 5), pady=2)
        gpulayers_entry = ttk.Entry(overrides_frame, textvariable=self.override_gpu_layers, width=10)
        gpulayers_entry.grid(row=0, column=3, sticky="w", padx=(0, 15), pady=2)
        
        # Row 1 of Overrides: Tensor Split and FlashAttention
        tensorsplit_label = ttk.Label(overrides_frame, text="Tensor Split:")
        tensorsplit_label.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        # This entry now shows a comma-separated string (e.g., "50.0, 50.0")
        tensorsplit_entry = ttk.Entry(overrides_frame, textvariable=self.override_tensor_split, width=20)
        tensorsplit_entry.grid(row=1, column=1, sticky="w", padx=(0, 15), pady=2)
        
        flashattn_label = ttk.Label(overrides_frame, text="FlashAttention:")
        flashattn_label.grid(row=1, column=2, sticky="w", padx=(0, 5), pady=2)
        flashattn_check = ttk.Checkbutton(overrides_frame, variable=self.override_flashattention_var)
        flashattn_check.grid(row=1, column=3, sticky="w", padx=(0, 15), pady=2)
        
        # Row 2 of Overrides: Context Size
        contextsize_label = ttk.Label(overrides_frame, text="Context Size:")
        contextsize_label.grid(row=2, column=0, sticky="w", padx=(0, 5), pady=2)
        contextsize_entry = ttk.Entry(overrides_frame, textvariable=self.override_contextsize, width=20)
        contextsize_entry.grid(row=2, column=1, sticky="w", padx=(0, 15), pady=2)
        # --- End Overrides Section ---
        
        # Selected Config Display
        selected_frame = ttk.LabelFrame(self.main_frame, text="Selected Configuration", padding=10)
        selected_frame.grid(row=7, column=0, sticky="ew", pady=(0, 15))
        
        selected_label = ttk.Label(selected_frame, textvariable=self.selected_config)
        selected_label.pack(fill=tk.X)
        
        # --- Buttons Row (Stop, Create Launcher, Launch) ---
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.grid(row=8, column=0, sticky="ew", pady=(0, 10))
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        
        stop_btn = tk.Button(buttons_frame, text="Stop KoboldCPP", 
                             command=self.stop_koboldcpp,
                             bg="#e74c3c", fg="white",
                             activebackground="#c0392b", activeforeground="white",
                             relief=tk.FLAT, padx=15, pady=5,
                             font=("Segoe UI", 10))
        stop_btn.grid(row=0, column=0, padx=5, sticky="ew")
        
        create_launcher_btn = tk.Button(buttons_frame, text="Create Launcher File", 
                                        command=self.create_launcher_file,
                                        bg=self.button_color, fg="white",
                                        activebackground=self.hover_color, activeforeground="white",
                                        relief=tk.FLAT, padx=15, pady=5,
                                        font=("Segoe UI", 10))
        create_launcher_btn.grid(row=0, column=1, padx=5, sticky="ew")
        
        launch_btn = tk.Button(buttons_frame, text="Launch KoboldCPP", 
                               command=self.launch_koboldcpp,
                               bg=self.accent_color, fg="white",
                               activebackground=self.hover_color, activeforeground="white",
                               relief=tk.FLAT, padx=15, pady=5,
                               font=("Segoe UI", 10, "bold"))
        launch_btn.grid(row=0, column=2, padx=5, sticky="ew")
        # --- End Buttons Row ---
        
        # Error Display Area
        error_frame = ttk.LabelFrame(self.main_frame, text="Errors and Messages", padding=10)
        error_frame.grid(row=9, column=0, sticky="ew", pady=(5, 0))
        
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
        
        # Status Bar
        status_bar = ttk.Label(self.root, textvariable=self.status_text, 
                               relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_executable(self):
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
        directory = filedialog.askdirectory(
            title="Select Configuration Files Directory",
            initialdir=self.config_dir.get() if self.config_dir.get() else None
        )
        
        if directory:
            self.config_dir.set(directory)
            self.update_config_list()
            self.save_settings()
            self.status_text.set(f"Configuration directory set to: {directory}")
    
    def update_config_list(self):
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
                self.status_text.set(f"Found {len(self.config_files)} configuration files")
                self.show_error_message(f"Found {len(self.config_files)} configuration files in: {self.config_dir.get()}")
        except Exception as e:
            error_msg = f"Error reading directory: {str(e)}"
            self.show_error_message(error_msg)
            self.status_text.set("Error reading directory")
    
    def on_config_select(self, event):
        selection = self.config_listbox.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.config_files):
                selected_file = self.config_files[index]
                self.selected_config.set(selected_file)
                full_path = os.path.join(self.config_dir.get(), selected_file)
                self.status_text.set(f"Selected: {full_path}")
                self.save_settings()
                # Attempt to read the config file to populate override defaults
                try:
                    with open(full_path, "r") as f:
                        config_data = json.load(f)
                except Exception as e:
                    config_data = {}
                self.default_threads = str(config_data.get("threads", ""))
                self.override_threads.set(self.default_threads)
                
                self.default_gpu_layers = str(config_data.get("gpulayers", ""))
                self.override_gpu_layers.set(self.default_gpu_layers)
                
                ts = config_data.get("tensor_split", "")
                # Join list values with a comma and a space so the override field shows "50.0, 50.0"
                if isinstance(ts, list):
                    self.default_tensor_split = ", ".join(str(x) for x in ts)
                else:
                    self.default_tensor_split = str(ts)
                self.override_tensor_split.set(self.default_tensor_split)
                
                self.default_contextsize = str(config_data.get("contextsize", ""))
                self.override_contextsize.set(self.default_contextsize)
                
                self.default_flashattention = bool(config_data.get("flashattention", False))
                self.override_flashattention_var.set(self.default_flashattention)
    
    def launch_koboldcpp(self):
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
        
        # Launch in a separate thread
        launch_thread = threading.Thread(target=self._run_koboldcpp, args=(config_path,))
        launch_thread.daemon = True
        launch_thread.start()
        
        self.status_text.set(f"Launching KoboldCPP with config: {self.selected_config.get()}")
    
    def _run_koboldcpp(self, config_path):
        try:
            # Get the executable's directory and name
            exe_dir = os.path.dirname(self.koboldcpp_path.get())
            exe_name = os.path.basename(self.koboldcpp_path.get())
            
            # Log debug info
            self.show_error_message(f"DEBUG INFO:\n"
                                    f"- Executable directory: '{exe_dir}'\n"
                                    f"- Executable name: '{exe_name}'\n"
                                    f"- Full executable path: '{self.koboldcpp_path.get()}'\n"
                                    f"- Config path: '{config_path}'\n"
                                    f"- Directory exists: {os.path.exists(exe_dir)}\n"
                                    f"- Executable exists: {os.path.exists(self.koboldcpp_path.get())}\n"
                                    f"- Config exists: {os.path.exists(config_path)}\n")
            
            # Build the command using absolute paths
            full_exe_path = os.path.abspath(self.koboldcpp_path.get())
            cmd = [full_exe_path, "--config", config_path]
            
            # Add override parameters if they differ from the defaults
            if self.override_flashattention_var.get() != self.default_flashattention:
                if self.override_flashattention_var.get():
                    cmd.append("--flashattention")
                # No flag to disable flashattention if unchecked
                
            if self.override_threads.get() and self.override_threads.get() != self.default_threads:
                cmd.extend(["--threads", self.override_threads.get()])
            
            if self.override_gpu_layers.get() and self.override_gpu_layers.get() != self.default_gpu_layers:
                cmd.extend(["--gpulayers", self.override_gpu_layers.get()])
            
            if self.override_tensor_split.get() and self.override_tensor_split.get() != self.default_tensor_split:
                # Split on commas and trim whitespace so that the commas are preserved in the display and passed correctly
                ts_values = [v.strip() for v in self.override_tensor_split.get().split(',')]
                cmd.append("--tensor_split")
                cmd.extend(ts_values)
            
            if self.override_contextsize.get() and self.override_contextsize.get() != self.default_contextsize:
                cmd.extend(["--contextsize", self.override_contextsize.get()])
            
            # Log the final command
            self.show_error_message(f"Executing command: {' '.join(cmd)}")
            
            # Run the process in a new console (if on Windows)
            self.process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE if platform.system() == "Windows" else 0
            )
            
            # If the "Launch nvidia-smi" checkbox is checked, launch nvidia-smi -l 1
            if self.launch_nvidia_smi.get():
                try:
                    if platform.system() == "Windows":
                        subprocess.Popen(["nvidia-smi", "-l", "1"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                    else:
                        subprocess.Popen(["nvidia-smi", "-l", "1"])
                except Exception as e:
                    print(f"Error launching nvidia-smi: {e}")
            
            self.running = True
            
        except Exception as e:
            error_msg = f"Error launching KoboldCPP: {str(e)}\n\nCheck that both the executable and config file exist and are accessible."
            self.show_error_message(error_msg)
            self.status_text.set("Launch failed")
    
    def stop_koboldcpp(self):
        """Stop the running KoboldCPP process"""
        if self.process is None or not self.running:
            self.show_error_message("No KoboldCPP process is currently running.")
            return
            
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
            self.show_error_message("KoboldCPP process has been terminated.")
            
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
        default_ext = ".bat" if is_windows else ".sh"
        # Change filename if any override differs from default
        overrides_active = (
            (self.override_threads.get() and self.override_threads.get() != self.default_threads) or
            (self.override_gpu_layers.get() and self.override_gpu_layers.get() != self.default_gpu_layers) or
            (self.override_tensor_split.get() and self.override_tensor_split.get() != self.default_tensor_split) or
            (self.override_contextsize.get() and self.override_contextsize.get() != self.default_contextsize) or
            (self.override_flashattention_var.get() != self.default_flashattention)
        )
        if overrides_active:
            filename = f"Launch_{self.selected_config.get().split('.')[0]}_with_overrides{default_ext}"
        else:
            filename = f"Launch_{self.selected_config.get().split('.')[0]}{default_ext}"
        
        filetypes = [("Batch Files", "*.bat")] if is_windows else [("Shell Scripts", "*.sh")]
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
            
            if self.override_flashattention_var.get() != self.default_flashattention:
                if self.override_flashattention_var.get():
                    cmd_parts.append("--flashattention")
            
            if self.override_threads.get() and self.override_threads.get() != self.default_threads:
                cmd_parts.extend(["--threads", self.override_threads.get()])
            
            if self.override_gpu_layers.get() and self.override_gpu_layers.get() != self.default_gpu_layers:
                cmd_parts.extend(["--gpulayers", self.override_gpu_layers.get()])
            
            if self.override_tensor_split.get() and self.override_tensor_split.get() != self.default_tensor_split:
                ts_values = [v.strip() for v in self.override_tensor_split.get().split(',')]
                cmd_parts.append("--tensor_split")
                cmd_parts.extend(ts_values)
            
            if self.override_contextsize.get() and self.override_contextsize.get() != self.default_contextsize:
                cmd_parts.extend(["--contextsize", self.override_contextsize.get()])
            
            cmd_string = " ".join(cmd_parts)
            
            # Prepend nvidia-smi command if checkbox is checked
            if self.launch_nvidia_smi.get():
                if is_windows:
                    nvidia_cmd = 'start nvidia-smi -l 1'
                else:
                    nvidia_cmd = 'nvidia-smi -l 1 &'
                cmd_string = nvidia_cmd + "\n" + cmd_string
            
            if is_windows:
                script_content = f"""@echo off
cd /d "{exe_dir}"
{cmd_string}
"""
            else:
                script_content = f"""#!/bin/bash
cd "{exe_dir}"
{cmd_string}
"""
            
            with open(save_path, 'w') as f:
                f.write(script_content)
                
            if not is_windows:
                os.chmod(save_path, 0o755)
                
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
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.koboldcpp_path.set(settings.get('koboldcpp_path', ''))
                    self.config_dir.set(settings.get('config_dir', ''))
                    self.selected_config.set(settings.get('selected_config', ''))
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_settings(self):
        try:
            settings = {
                'koboldcpp_path': self.koboldcpp_path.get(),
                'config_dir': self.config_dir.get(),
                'selected_config': self.selected_config.get()
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
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