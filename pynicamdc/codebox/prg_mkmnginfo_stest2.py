import numpy as np
import toml 
from mod_adm import prc_num_val, nmax_dmd
from mod_io_param import rgnlen, I_SW, I_NE, I_NW, output_fname
from mod_stdio import dmd_data

def prg_mkmnginfo_final_version():
    """
    This function replicates the behavior of the given Fortran code, focusing on domain decomposition,
    process assignment, and data structure population.
    """

    # Domain Decomposition Logic for rgn_tab
    rgn_tab = {}
    for d in range(1, nmax_dmd + 1):
        for i in range(1, rgnlen + 1):
            for j in range(1, rgnlen + 1):
                l = (rgnlen * rgnlen) * (d - 1) + rgnlen * (j - 1) + i
                k = dmd_data[I_SW][d - 1]  # Fetching from demand data
                rgn_tab[l] = {
                    'd': d,
                    'i': i,
                    'j': j,
                    'k': k
                }
                
    # Logic for populating prc_tab, rgn_link_info, and proc_info
    prc_tab, rgn_link_info, proc_info = {}, {}, {}
    for d in range(1, nmax_dmd + 1):
        prc_num = (d - 1) % prc_num_val + 1  
        prc_data = dmd_data[I_SW][d - 1] + d  
        prc_tab[prc_num] = {
            'id': prc_num,
            'data': prc_data
        }
        rgn_link_info[d] = {
            'id': d,
            'data': dmd_data[I_NE][d - 1]
        }
        proc_info[d] = {
            'id': d,
            'data': dmd_data[I_NW][d - 1]
        }
    
    # File Writing Logic with error handling
    try:
        with open(output_fname, 'w') as file:
            for key, value in rgn_tab.items():
                file.write(f"&RGN_INFO {key}\\n")
                for subkey, subvalue in value.items():
                    file.write(f" {subkey} = {subvalue}\\n")
                file.write("/\\n\\n")
            
            for key, value in prc_tab.items():
                file.write(f"&PRC_INFO {key}\\n")
                for subkey, subvalue in value.items():
                    file.write(f" {subkey} = {subvalue}\\n")
                file.write("/\\n\\n")

            for key, value in rgn_link_info.items():
                file.write(f"&RGN_LINK_INFO {key}\\n")
                for subkey, subvalue in value.items():
                    file.write(f" {subkey} = {subvalue}\\n")
                file.write("/\\n\\n")

            for key, value in proc_info.items():
                file.write(f"&PROC_INFO {key}\\n")
                for subkey, subvalue in value.items():
                    file.write(f" {subkey} = {subvalue}\\n")
                file.write("/\\n\\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

# Running the main function to test the final version of the logic
prg_mkmnginfo_final_version()
