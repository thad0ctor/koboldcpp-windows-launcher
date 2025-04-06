# KoboldCPP Launcher

A simple, user-friendly GUI launcher for [KoboldCPP](https://github.com/LostRuins/koboldcpp) that streamlines the process of running different model configurations.

![KoboldCPP Launcher Screenshot](https://i.imgur.com/8dkT5T7.png)

## Features

- **Easy Model Configuration Management**: Browse and select from your .kcpps configuration files
- **Launch Parameter Overrides**: Quickly modify several parameters without editing your config files:
  - Threads
  - GPU Layers
  - Tensor Split
  - Context Size
  - FlashAttention
- **Create Launcher Scripts**: Generate batch (.bat) or shell (.sh) scripts for any configuration
- **Integrated nvidia-smi**: Option to automatically launch nvidia-smi alongside KoboldCPP
- **Persistent Settings**: Remembers your paths and selected configuration (json files save to your windows user profile folder, e.g.: C:\Users\User123)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/koboldcpp-launcher.git
   cd koboldcpp-launcher
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the launcher:
   ```bash
   python koboldcpp_launcher.py
   ```

### Prerequisites

- Python 3.6+ (if running from source)
- KoboldCPP installed and configured

### Building Standalone Executables

```bash
# Windows
pyinstaller --onefile --windowed --icon=icon.ico koboldcpp_launcher.py

# Linux/Mac
pyinstaller --onefile --windowed koboldcpp_launcher.py
```

## Usage

1. **Set KoboldCPP Executable**: Browse to select your KoboldCPP executable
2. **Set Configuration Directory**: Select the folder containing your .kcpps files
3. **Select a Configuration**: Choose from the available configuration files
4. **Set Overrides** (Optional): Modify any launch parameters as needed
5. **Launch KoboldCPP**: Click "Launch KoboldCPP" to start the program with your selected configuration

### Creating Launcher Scripts

Click "Create Launcher File" to generate a standalone script that includes all your current settings and overrides. This allows you to quickly launch specific configurations without opening the launcher.

## Troubleshooting

- **KoboldCPP Won't Launch**: 
  - Ensure paths to the executable and config files are correct
  - Check the error message display for specific issues
  - Verify that your configuration file is valid JSON

- **Config Files Not Showing**:
  - Make sure your configuration files have the `.kcpps` extension
  - Refresh the list after adding new files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [KoboldCPP](https://github.com/LostRuins/koboldcpp) - The popular local LLM backend