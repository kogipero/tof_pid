# README

This repository performs PID analysis for the ePIC Barrel ToF. It contains code designed for a Python + ROOT environment. Below you will find the directory structure, configuration files, execution instructions, and additional explanations.

---

## Directory Structure and File Descriptions

```
.
├── config
│   ├── file_path.yaml        
│   ├── branch_name.yaml      
│   └── execute_config.yaml   
│── src
│   │── matching_mc_and_track.py
│   │── matching_mc_and_track_plotter.py
│   │── matching_tof_and_track.py
│   │── matching_tof_and_track_plotter.py
│   │── mc_plotter.py
│   │── mc_reader.py
│   │── tof_pid_performance_manager.py
│   │── tof_pid_performance_plotter.py
│   │── tof_plotter.py
│   │── tof_reader.py
│   │── track_plotter.py
│   │── track_reader.py
│   └── utility_function.py
├── helper_function.py        
└── analyze_script.py         

```


---

## Required Environment

1. **Python Environment**
   - Python 3
   - PyROOT
   - A YAML library (e.g., PyYAML)

2. **ROOT Environment**
   - ROOT must be installed and properly configured

---

## Execution Instructions

1. **Edit `execute_config.yaml`**  
   In the file `config/execute_config.yaml`, modify the following items as needed:
   - **Number of Events:** (e.g., `SELECTED_EVENTS: 10000`)
   - **Output ROOT Filename:** (e.g., `output_name: test.root`)
   - **Output Directory Name:** (e.g., `directory_name: test`)
   - **Input File for Analysis:**  
     Adjust the key `analysis_event_type` as needed. Ensure that the file path specified in `file_path.yaml` corresponds correctly.

2. **Run the Analysis Code**  
   Execute the main script by running:
   ```bash
   python analyze_script.py --rootfile output.root


