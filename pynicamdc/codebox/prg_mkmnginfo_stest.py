def prg_mkmnginfo():
    # Variable Declarations translated from Fortran
    fid = 10
    rlevel = 1  # region division level
    prc_num = 40  # process number
    HGRID_SYSTEM = 'ICO'  # grid system (default ico)
    MAPPING_TYPE = ''  # mapping method
    output_fname = 'rl01-prc40.info'  # output region-management filename
    
    # Constants and Data Generation Logic translated from Fortran
    I_SW, I_NW, I_NE, I_SE = range(4)
    nmax_dmd = 10
    dmd_data = {
        I_SW: [6, 10, 9, 8, 7, 7, 8, 9, 10, 6],
        I_NW: [5, 1, 2, 3, 4, 5, 4, 3, 2, 1],
        I_NE: [2, 3, 4, 5, 1, 1, 2, 3, 4, 5],
        I_SE: [10, 9, 8, 7, 6, 10, 9, 8, 7, 6]
    }
    rgnlen = 2 ** rlevel
    all_rgn = nmax_dmd * rgnlen * rgnlen
    
    # Nested Loop Translation and Internal Calculations
    rgn_tab = {}
    for d in range(1, nmax_dmd + 1):
        for i in range(1, rgnlen + 1):
            for j in range(1, rgnlen + 1):
                l = (rgnlen * rgnlen) * (d - 1) + rgnlen * (j - 1) + i
                k = dmd_data[I_SW][d - 1]  # Placeholder logic
                rgn_tab[l] = {
                    'd': d,
                    'i': i,
                    'j': j,
                    'k': k
                }
                
    # Placeholder logic for populating other data structures like prc_tab
    prc_tab, rgn_link_info, proc_info = {}, {}, {}
    prc_num = 0
    for d in range(1, nmax_dmd + 1):
        prc_num += 1
        prc_tab[prc_num] = {
            'id': prc_num,
            'data': dmd_data[I_SW][d - 1]
        }
        rgn_link_info[d] = {
            'id': d,
            'data': dmd_data[I_NE][d - 1]
        }
        proc_info[d] = {
            'id': d,
            'data': dmd_data[I_NW][d - 1]
        }
    
    # File Writing Logic (unchanged)
    with open(output_fname, 'w') as file:
        for key, value in rgn_tab.items():
            file.write(f"&RGN_INFO {key}\n")
            file.write(f"{value}\n")
            file.write("/\n\n")
        
        for key, value in prc_tab.items():
            file.write(f"&PRC_INFO {key}\n")
            file.write(f"{value}\n")
            file.write("/\n\n")

        for key, value in rgn_link_info.items():
            file.write(f"&RGN_LINK_INFO {key}\n")
            file.write(f"{value}\n")
            file.write("/\n\n")

        for key, value in proc_info.items():
            file.write(f"&PROC_INFO {key}\n")
            file.write(f"{value}\n")
            file.write("/\n\n")

# Running the main function to test the translation of nested loops and internal calculations
prg_mkmnginfo()
