import toml
import numpy as np


def _default_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    else:
        return MPI.COMM_WORLD


class Admin:

    def __init__(self):
        # Basic definition & information ======

        # Identifier for single computation or parallel computation
        self.adm_single_prc = 0
        self.adm_multi_prc  = 1

        # Identifiers of directions of region edges
        self.adm_sw = 0 #1
        self.adm_nw = 1 #2
        self.adm_ne = 2 #3
        self.adm_se = 3 #4

        # Identifiers of directions of region vertices
        self.adm_w = 0 #1
        self.adm_n = 1 #2
        self.adm_e = 2 #3
        self.adm_s = 3 #4

        # Identifier of triangle element (i-axis-side or j-axis side)
        self.adm_ti = 1
        self.adm_tj = 2

        # Identifier of line element (i-axis-side, ij-axis side, or j-axis side)
        self.adm_ai  = 1
        self.adm_aij = 2
        self.adm_aj  = 3

        # Identifier of 1 variable
        self.adm_knone = 1
        self.adm_vnone = 1

        # Identifier of poles (north pole or south pole)
        self.adm_npl = 0 #1
        self.adm_spl = 1 #2

        # Fist colomn on the table for region and direction
        self.adm_rid = 0 #1
        self.adm_dir = 1 #2

        #
        self.adm_vmiss = 1.e0

        # Information for processes ======

        # Communication world for NICAM
        self.adm_mpi_alive = False

        # Master process
        self.adm_prc_run_master = 1  # ???

        # Process ID which manages the pole regions.
        #integer adm_prc_pl

        # Process ID which have the pole regions.
        #integer adm_prc_npl
        #integer adm_prc_spl
        #integer adm_prc_nspl(adm_npl:adm_spl)

        #logical adm_have_pl

        # Information for processes-region relationship ======

        # Maximum number of regions managed by 1 process.
        self.prc_rgn_nmax = 2560

        # Regin managing file name
        #character adm_rgnmngfname

        # Number of regions mangeged by each process
        #integer adm_prc_rnum(:)

        # Table of regions managed by each process
        # integer adm_prc_tab(:,:)

        # Table of edge link information
        # integer adm_rgn_etab(:,:,:)

        # Table of process ID from region ID
        # integer adm_rgn2prc(:)

        # Maximum number of vertex linkage
        # integer adm_vlink_nmax

        # Table of n-vertex-link(?) at the region vertex
        # integer adm_rgn_vnum(:,:)

        # Table of vertex link information
        # integer adm_rgn_vtab(:,:,:,:)

        # Table of vertex link information for poles
        # integer adm_rgn_vtab_pl(:,:,:)

        # Region ID (reguler) of north pole management
        # integer, public, save :: adm_rgnid_npl_mng
        # integer, public, save :: adm_rgnid_spl_mng

        #====== Information for regions ======

        # Region division level
        # integer adm_rlevel

        # Total number of regular regions managed by all process
        # integer adm_rgn_nmax

        # Maximum number of pole regions
        self.adm_rgn_nmax_pl = 2

        # Local region number
        # integer adm_lall

        # Local region number for poles
        self.adm_lall_pl = self.adm_rgn_nmax_pl

        # Present Local region number ! 2010.4.26 M.Satoh
        # integer adm_l_me

        #logical adm_have_sgp(:) # region have singlar point?

        #====== Grid resolution informations  ======

        # Grid division level
        # integer adm_glevel

        # Horizontal grid numbers
        # integer adm_gmin
        # integer adm_gmax
        # integer adm_gall_1d
        # integer adm_gall

        # grid number of inner region in the diamond
        # integer adm_gall_in

        # Identifiers of grid points around poles.
        self.adm_gslf_pl = 1
        self.adm_gmin_pl = 2
        # integer adm_gmax_pl     # [mod] S.Iga 100607
        # integer adm_gall_pl     # [mod] S.Iga 100607

        # Vertica grid numbers
        # integer adm_vlayer
        # integer adm_kmin
        # integer adm_kmax
        # integer adm_kall

        #======  List vector for 1-dimensional array in the horiz. dir. ======

        # Identifiers of grid points around a grid point
        self.adm_gij_nmax = 7
        self.adm_giojo = 0 #1
        self.adm_gipjo = 1 #2
        self.adm_gipjp = 2 #3
        self.adm_giojp = 3 #4
        self.adm_gimjo = 4 #5
        self.adm_gimjm = 5 #6
        self.adm_giojm = 6 #7

        self.adm_nxyz = 3 # dimension of the spacial vector

        #=========== For New Grid (XTMS) start    <= S.Iga100607

        # Horizontal Grid type
        self.adm_hgrid_system = 'ICO' # icosahedral

        # Number of lines at each pole (maybe this is identical to ADM_VLINK_NMAX)
        self.adm_xtms_k=-1 # default

        # Number of segment for MLCP
        self.adm_xtms_mlcp_s= 1

        # XTMS LEVEL (it is conveniently defined especially for mod_oprt)
        self.adm_xtms_level = 0 # original icosahedral (NICAM)

        #=========== For New Grid (XTMS) end    S.Iga100607 =>
        self.adm_debug = True #False

        # Private parameters & variables
        self._gdummy = 1 # Horizontal dummy(halo) cell
        self._kdummy = 1 # Vertical   dummy(halo) cell
        self._nmax_dmd = -999 # number of diamond


    def adm_proc_init(self):
        # Run type (single or multi processes)
        self._adm_run_type = _default_mpi_comm()

        if self._adm_run_type is not None:
            self.adm_mpi_alive = True

            # Communication world for NICAM
            self.adm_comm_world = self._adm_run_type

            # Total number of process
            self.adm_prc_all = self.adm_comm_world.Get_size()

            # My process ID
            self.my_rank = self.adm_comm_world.Get_rank()
        else:
            self.adm_prc_all = 1
            self.my_rank = 0

        self.adm_prc_me = self.my_rank + 1
        self.adm_prc_pl = 1


    def adm_setup(self, param_fname, msg_base="msg"):
        from share.misc import misc_make_idstr #, misc_get_available_fid

        msg = msg_base

        #--- open message file
        fname = misc_make_idstr(msg, 'pe', self.adm_prc_me)

        # check if a file already exists 
        # adm_log_fid = misc_get_available_fid()
        self.adm_log_fid = open(fname, 'wt')

        self.adm_log_fid.write('############################################################\n')
        self.adm_log_fid.write('#                                                          #\n')
        self.adm_log_fid.write('#   NICAM : Nonhydrostatic ICosahedal Atmospheric Model    #\n')
        self.adm_log_fid.write('#                                                          #\n')
        self.adm_log_fid.write('############################################################\n')

        #--- open control file
        try:
            self.adm_ctl_fid = toml.load(param_fname)
        except FileNotFoundError:
            print('xxx Cannot open parameter control file: ', param_fname)
            self.adm_proc_stop()
 
        #--- read parameters
        self.adm_log_fid.write('\n+++ Module[adm]/Category[common share]\n')
 
        try:
            _admparam = self.adm_ctl_fid['admparam']
        except KeyError:
            print('xxx ADMPARAM is not specified in control file! STOP.')
            self.adm_log_fid.write('xxx ADMPARAM is not specified in control file! STOP.\n')
            self.adm_proc_stop()

        for v in ('glevel', 'rlevel', 'vlayer', 'rgnmngfname'):
            try:
                setattr(self, f'adm_{v}', _admparam[v])
            except KeyError:
                print('xxx No appropriate names in ADMPARAM. STOP.\n')
                self.adm_log_fid.write('xxx No appropriate names in ADMPARAM. STOP.\n')
                self.adm_proc_stop()

        if self.adm_hgrid_system == 'ICO':
            self.adm_xtms_level = 0
            self.adm_xtms_k = 5
            self.nmax_dmd = 10
        else:
            print('xxx Name of adm_hgrid_system is wrong. STOP.')  # REMOVE ???
            self.adm_log_fid.write('xxx Name of adm_hgrid_system is wrong. STOP.\n')
            self.adm_proc_stop()
 
        self.adm_vlink_nmax = self.adm_xtms_k
        self.adm_gmax_pl = self.adm_vlink_nmax + 1
        self.adm_gall_pl = self.adm_vlink_nmax + 1

        # ERROR if Glevel & Rlevel are not defined
        if self.adm_glevel < 1:
            print(f'xxx Glevel is not appropriate, STOP. GL= {self.adm_glevel}')
            self.adm_log_fid.write(f'xxx Glevel is not appropriate, STOP. GL= {self.adm_glevel}\n')
            self.adm_proc_stop()

        if self.adm_rlevel < 0:
            print(f'xxx Rlevel is not appropriate, STOP. RL= {self.adm_rlevel}')
            self.adm_log_fid.write(f'xxx Rlevel is not appropriate, STOP. RL= {self.adm_rlevel}\n')
            self.adm_proc_stop()
 
        rgn_nmax = 2 ** self.adm_rlevel
        self.adm_rgn_nmax = rgn_nmax * rgn_nmax * self.nmax_dmd
    
        self._input_mnginfo()  #!!!!
    
        self.adm_prc_npl = self.adm_prc_pl
        self.adm_prc_spl = self.adm_prc_pl

        self.adm_prc_nspl = np.zeros((2,), dtype=int)
        self.adm_prc_nspl[self.adm_npl] = self.adm_prc_npl
        self.adm_prc_nspl[self.adm_spl] = self.adm_prc_spl

        nmax = 2 ** (self.adm_glevel - self.adm_rlevel)
        self.adm_gmin = self._gdummy # + 1
        self.adm_gmax = self._gdummy + nmax
        self.adm_gall_1d = self._gdummy + nmax + self._gdummy
        self.adm_gall = self.adm_gall_1d * self.adm_gall_1d
    
        self.adm_gall_in = ( nmax + self._gdummy ) * ( nmax + self._gdummy ) #--- inner grid number (e.g., 33x33 for gl05)

        if self.adm_vlayer == 1:
            self.adm_kmin = 0 #1
            self.adm_kmax = 0 #1
            self.adm_kall = 1
        else:
            self.adm_kmin = self._kdummy # + 1
            self.adm_kmax = self._kdummy + self.adm_vlayer
            self.adm_kall = self._kdummy + self.adm_vlayer + self._kdummy

        # !!! check dimensionality 
        self.adm_lall = self.adm_prc_rnum[self.adm_prc_me - 1]

        self.adm_have_sgp = np.zeros((self.adm_lall,), dtype=bool)

        # !!! check dimensionality
        for l in range(self.adm_lall):
            rgnid = self.adm_prc_tab[l, self.adm_prc_me - 1] # 1 --> 0
            if self.adm_rgn_vnum[self.adm_w, rgnid - 1] == 3:
                self.adm_have_sgp[l] = True
 
        if self.adm_prc_me == self.adm_prc_pl:
            self.adm_have_pl = True
        else:
            self.adm_have_pl = False
 
        # 2010.4.26 M.Satoh; 2010.5.11 M.Satoh
        # ADM_l_me: this spans from 1 to ADM_lall, if effective.
        # Otherwise, ADM_l_me = 0 should be set. see mod_history
        self.adm_l_me = 0

        #--- make suffix for list-vector loop.
        self.adm_mk_suffix()
        self._output_info()


    def _input_mnginfo(self):
        # num_of_rgn #--- number of region

        # rgnid                        #--- region ID
        sw = -np.ones((2,), dtype=int) #--- south-west region info
        nw = -np.ones((2,), dtype=int) #--- nouth-west region info
        ne = -np.ones((2,), dtype=int) #--- nouth-east region info
        se = -np.ones((2,), dtype=int) #--- south-east region info

        # num_of_proc                #--- number of run-processes
        # peid                       #--- process ID
        # num_of_mng                 #--- number of regions be managed
        mng_rgnid = -np.ones((self.prc_rgn_nmax,), dtype=int) #--- managed region ID

        self.adm_log_fid.write('\n')
        self.adm_log_fid.write('+++ Module[mnginfo]/Category[common share]\n')

        try:
            with open(self.adm_rgnmngfname, "r") as f:
                self.adm_rgn_fid = toml.load(f)
                self.num_of_rgn = self.adm_rgn_fid['rgn_info']['num_of_rgn']
                if self.num_of_rgn != self.adm_rgn_nmax:
                    self.adm_log_fid.write('xxx No match for region number! STOP.\n')
                    self.adm_log_fid.write(f'xxx ADM_rgn_nmax= {self.adm_rgn_nmax} num_of_rgn={self.num_of_rgn}\n')
                    self.adm_proc_stop()

                self.adm_rgn_etab = np.zeros((2, 4, self.adm_rgn_nmax), dtype=int)

                for l in range(self.adm_rgn_nmax):
                    rgnid = int(self.adm_rgn_fid['rgn_link_info'][f'link{l+1}']['rgnid']) - 1
                    self.adm_rgn_etab[:, self.adm_sw, rgnid] = self.adm_rgn_fid['rgn_link_info'][f'link{l+1}']['sw']
                    self.adm_rgn_etab[:, self.adm_nw, rgnid] = self.adm_rgn_fid['rgn_link_info'][f'link{l+1}']['nw']
                    self.adm_rgn_etab[:, self.adm_ne, rgnid] = self.adm_rgn_fid['rgn_link_info'][f'link{l+1}']['ne']
                    self.adm_rgn_etab[:, self.adm_se, rgnid] = self.adm_rgn_fid['rgn_link_info'][f'link{l+1}']['se']

                num_of_proc = self.adm_rgn_fid['proc_info']['num_of_proc']
                if self.adm_prc_all != num_of_proc:
                    self.adm_log_fid.write('Msg : Sub[ADM_input_mngtab]/Mod[admin]\n')
                    self.adm_log_fid.write(' xxx No match for  process number! STOP.\n')
                    self.adm_log_fid.write(f' xxx ADM_prc_all= {self.adm_prc_all} num_of_proc= {num_of_proc}\n')
                    self.adm_proc_stop()

                self.adm_prc_rnum = np.zeros((self.adm_prc_all,), dtype=int)
                self.adm_prc_tab = -np.ones((self.prc_rgn_nmax, self.adm_prc_all), dtype=int) # [Fix] 11/06/30  T.Seiki, fill undefined value
                self.adm_rgn2prc = np.zeros((self.adm_rgn_nmax,), dtype=int)

                for m in range(self.adm_prc_all):
                    peid = int(self.adm_rgn_fid['rgn_mng_info'][f'mng{m+1}']['peid'])
                    num_of_mng = int(self.adm_rgn_fid['rgn_mng_info'][f'mng{m+1}']['num_of_mng'])
                    mng_rgnid = self.adm_rgn_fid['rgn_mng_info'][f'mng{m+1}']['mng_rgnid']

                    self.adm_prc_rnum[m] = num_of_mng
                    self.adm_prc_tab[:, peid-1] = mng_rgnid + [-1,] * (self.prc_rgn_nmax - len(mng_rgnid))
                    for n in range(num_of_mng):
                        self.adm_rgn2prc[mng_rgnid[n] - 1] = peid

        except FileNotFoundError:
            print('xxx mnginfo file is not found! STOP. : ', self.adm_rgnmngfname)
            self.adm_log_fid.write('xxx mnginfo file is not found! STOP. ', self.adm_rgnmngfname)
            self.adm_proc_stop()

        self.setup_vtab()
 

    def setup_vtab(self):

        self.adm_rgn_vnum = np.zeros((4, self.adm_rgn_nmax), dtype=int)
        self.adm_rgn_vtab = np.zeros((2, 4, self.adm_rgn_nmax, self.adm_vlink_nmax), dtype=int)
        self.adm_rgn_vtab_pl = np.zeros((2, self.adm_rgn_nmax_pl, self.adm_vlink_nmax), dtype=int)

        for l in range(self.adm_rgn_nmax):
            for k in range(self.adm_w, self.adm_s + 1):
                vnum, nrid, nvid = self.set_vinfo(l, k)
                #if self.adm_prc_me == 1: print(k+1, l+1, vnum)
                self.adm_rgn_vnum[k, l] = vnum
                self.adm_rgn_vtab[self.adm_rid, k, l, :] = nrid[:]
                self.adm_rgn_vtab[self.adm_dir, k, l, :] = nvid[:]

        for l in range(self.adm_rgn_nmax):
            if self.adm_rgn_vnum[self.adm_n, l] == self.adm_vlink_nmax:
                ll = l
                break

        self.adm_rgnid_npl_mng = ll + 1

        for v in range(self.adm_vlink_nmax):
            self.adm_rgn_vtab_pl[self.adm_rid, self.adm_npl,v] = self.adm_rgn_vtab[self.adm_rid, self.adm_n, ll, v]
            self.adm_rgn_vtab_pl[self.adm_dir, self.adm_npl,v] = self.adm_rgn_vtab[self.adm_dir, self.adm_n, ll, v]

        for l in range(self.adm_rgn_nmax):
            if self.adm_rgn_vnum[self.adm_s, l] == self.adm_vlink_nmax :
                ll = l
                break

        self.adm_rgnid_spl_mng = ll + 1

        for v in range(self.adm_vlink_nmax):
            self.adm_rgn_vtab_pl[self.adm_rid, self.adm_spl, v] = self.adm_rgn_vtab[self.adm_rid, self.adm_s, ll, v]
            self.adm_rgn_vtab_pl[self.adm_dir, self.adm_spl, v] = self.adm_rgn_vtab[self.adm_dir, self.adm_s, ll, v]
 

    def set_vinfo(self, rgnid, vertid):
        vert_num = 0

        rid = rgnid
        eid = vertid

        eid_dict = {self.adm_w: self.adm_sw,
                    self.adm_n: self.adm_nw,
                    self.adm_e: self.adm_ne,
                    self.adm_s: self.adm_se}
        eid = eid_dict[vertid]

        nrgnid = -np.ones((self.adm_vlink_nmax,), dtype=int)
        nvertid = -np.ones((self.adm_vlink_nmax,), dtype=int)

        while True:
            rid_new = self.adm_rgn_etab[self.adm_rid, eid, rid]
            eid_new = self.adm_rgn_etab[self.adm_dir, eid, rid] - 1
            #if self.adm_prc_me == 1:
            #    print('set_vinfo   ', rid+1, eid+1, rid_new, eid_new)
            if eid_new == 0: eid_new = 4
            rid = rid_new - 1
            eid = eid_new - 1

            vert_num += 1

            nrgnid[vert_num-1] = rid + 1
            nvertid[vert_num-1] = eid + 1
    
            if rid == rgnid: break

        return vert_num, nrgnid, nvertid


    def _output_info(self):

        self.adm_log_fid.write('\n====== Process management info. ======\n')
        self.adm_log_fid.write(f'--- Total number of process           : {self.adm_prc_all:7d}\n')
        self.adm_log_fid.write(f'--- My Process rank                   : {self.adm_prc_me:7d}\n')
        self.adm_log_fid.write('====== Region/Grid topology info. ======\n')
        self.adm_log_fid.write(f'--- Grid sysytem                      : {self.adm_hgrid_system.strip()}\n')
        self.adm_log_fid.write(f'--- #  of diamond                     : {self.nmax_dmd:7d}\n')
        self.adm_log_fid.write('====== Region management info. ======\n')
        self.adm_log_fid.write(f'--- Region level (RL)                 : {self.adm_rlevel:7d}\n')
        self.adm_log_fid.write(f'--- Total number of region            : {self.adm_rgn_nmax:7d} ( {2**self.adm_rlevel:4d} x {2**self.adm_rlevel:4d} x {self.nmax_dmd:4d} )\n')
        self.adm_log_fid.write(f'--- #  of region per process          : {self.adm_lall:7d}\n')
        self.adm_log_fid.write('--- ID of region in my process        : \n')
        self.adm_log_fid.write(', '.join([str(x) for x in self.adm_prc_tab[:self.adm_lall, self.adm_prc_me - 1]]))

        self.adm_log_fid.write(f'\n--- Region ID, contains north pole    : {self.adm_rgnid_npl_mng:7d}\n')
        self.adm_log_fid.write(f'--- Region ID, contains south pole    : {self.adm_rgnid_spl_mng:7d}\n')
        self.adm_log_fid.write(f'--- Process rank, managing north pole : {self.adm_prc_npl:7d}\n')
        self.adm_log_fid.write(f'--- Process rank, managing south pole : {self.adm_prc_spl:7d}\n')
        self.adm_log_fid.write('====== Grid management info. ======\n')
        self.adm_log_fid.write(f'--- Grid level (GL)                   : {self.adm_glevel}\n')
        self.adm_log_fid.write(f'--- Total number of grid (horizontal) : {4**(self.adm_glevel - self.adm_rlevel)*self.adm_rgn_nmax:7d} ({2**(self.adm_glevel - self.adm_rlevel):4d} x {2**(self.adm_glevel - self.adm_rlevel):4d} x {self.adm_rgn_nmax:7d})\n')
        self.adm_log_fid.write(f'--- Number of vertical layer          : {self.adm_kmax - self.adm_kmin:7d}\n')

        if self.adm_debug:
            self.adm_log_fid.write('====== Horizontal grid info. ======\n')
            self.adm_log_fid.write(f'adm_gmin = {self.adm_gmin}\n'+
                                f'adm_gmax = {self.adm_gmax}\n'+
                                f'adm_gall_1d = {self.adm_gall_1d}\n'+
                                f'adm_gall = {self.adm_gall}\n')
            self.adm_log_fid.write(f'adm_ioojoo_nmax = {self.adm_ioojoo_nmax}\n'+\
                                f'adm_ioojmo_nmax = {self.adm_ioojmo_nmax}\n'+\
                                f'adm_ioojop_nmax = {self.adm_ioojop_nmax}\n'+\
                                f'adm_ioojmp_nmax = {self.adm_ioojmp_nmax}\n'+\
                                f'adm_imojoo_nmax = {self.adm_imojoo_nmax}\n'+\
                                f'adm_imojmo_nmax = {self.adm_imojmo_nmax}\n'+\
                                f'adm_imojop_nmax = {self.adm_imojop_nmax}\n'+\
                                f'adm_imojmp_nmax = {self.adm_imojmp_nmax}\n'+\
                                f'adm_iopjoo_nmax = {self.adm_iopjoo_nmax}\n'+\
                                f'adm_iopjmo_nmax = {self.adm_iopjmo_nmax}\n'+\
                                f'adm_iopjop_nmax = {self.adm_iopjop_nmax}\n'+\
                                f'adm_iopjmp_nmax = {self.adm_iopjmp_nmax}\n'+\
                                f'adm_impjoo_nmax = {self.adm_impjoo_nmax}\n'+\
                                f'adm_impjmo_nmax = {self.adm_impjmo_nmax}\n'+\
                                f'adm_impjop_nmax = {self.adm_impjop_nmax}\n'+\
                                f'adm_impjmp_nmax = {self.adm_impjmp_nmax}\n')
            #print(self.adm_ioojmo[:10])

            for n in range(self.adm_lall):
                rgnid = self.adm_prc_tab[n, self.adm_prc_me - 1] # self.adm_prc_me or self.adm_prc_me - 1  ????
                self.adm_log_fid.write(f' --- Link information for region {rgnid}\n')
                self.adm_log_fid.write('     < edge link >   --- ( rgnid , edgid )\n')
                for k in range(self.adm_sw, self.adm_se+1):
                    self.adm_log_fid.write(f'     ({rgnid} , {k+1}) -> ({self.adm_rgn_etab[self.adm_rid, k, rgnid - 1]} , {self.adm_rgn_etab[self.adm_dir, k, rgnid - 1]})\n')

                self.adm_log_fid.write('     < vertex link > --- ( rgnid , edgid )')
                for k in range(self.adm_w, self.adm_s+1):
                    self.adm_log_fid.write(f'\n     ({rgnid} , {k+1}) : {self.adm_rgn_vnum[k, rgnid - 1]} point link\n')
                    for m in range(self.adm_rgn_vnum[k, rgnid - 1]):
                        self.adm_log_fid.write(f'        -> ( {self.adm_rgn_vtab[self.adm_rid, k, rgnid - 1, m]} ,  {self.adm_rgn_vtab[self.adm_dir, k, rgnid - 1, m]})\n')

            self.adm_log_fid.write(' --- Table of corresponding between region ID and process ID\n')
            self.adm_log_fid.write('    region ID :  process ID\n')
            for n in range(self.adm_rgn_nmax):
                self.adm_log_fid.write(f'{n+1:13d} {self.adm_rgn2prc[n]:14d}\n')


    def adm_mk_suffix(self):
        # List vectors

        gall_in = self.adm_gmax - self.adm_gmin #+ 1

        #--- ADM_IooJoo
        self.adm_ioojoo_nmax = ( gall_in ) * ( gall_in )
        self.adm_ioojoo = np.zeros((self.adm_ioojoo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax):
            for i in range(self.adm_gmin, self.adm_gmax):
                self.adm_ioojoo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_ioojoo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_ioojoo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_ioojoo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_ioojoo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_ioojoo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_ioojoo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
 
        #--- ADM_IooJmo
        self.adm_ioojmo_nmax = ( gall_in ) * ( gall_in+1 )
        self.adm_ioojmo = np.zeros((self.adm_ioojmo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax):
            for i in range(self.adm_gmin, self.adm_gmax):
                self.adm_ioojmo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_ioojmo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_ioojmo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_ioojmo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_ioojmo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_ioojmo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_ioojmo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1

        #--- ADM_IooJop
        self.adm_ioojop_nmax = ( gall_in ) * ( gall_in+1 )
        self.adm_ioojop = np.zeros((self.adm_ioojop_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax+1):
            for i in range(self.adm_gmin, self.adm_gmax):
                self.adm_ioojop[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_ioojop[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_ioojop[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_ioojop[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_ioojop[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_ioojop[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_ioojop[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_IooJmp
        self.adm_ioojmp_nmax = ( gall_in ) * ( gall_in+2 )
        self.adm_ioojmp = np.zeros((self.adm_ioojmp_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax+1):
            for i in range(self.adm_gmin, self.adm_gmax):
                self.adm_ioojmp[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_ioojmp[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_ioojmp[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_ioojmp[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_ioojmp[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_ioojmp[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_ioojmp[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_ImoJoo
        self.adm_imojoo_nmax = ( gall_in+1 ) * ( gall_in )
        self.adm_imojoo = np.zeros((self.adm_imojoo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax):
            for i in range(self.adm_gmin-1, self.adm_gmax):
                self.adm_imojoo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_imojoo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_imojoo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_imojoo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_imojoo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_imojoo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_imojoo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1

        #--- ADM_ImoJmo
        self.adm_imojmo_nmax = ( gall_in+1 ) * ( gall_in+1 )
        self.adm_imojmo = np.zeros((self.adm_imojmo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax):
            for i in range(self.adm_gmin-1, self.adm_gmax):
                self.adm_imojmo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_imojmo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_imojmo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_imojmo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_imojmo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_imojmo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_imojmo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_ImoJop
        self.adm_imojop_nmax = ( gall_in+1 ) * ( gall_in+1 )
        self.adm_imojop = np.zeros((self.adm_imojop_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax+1):
            for i in range(self.adm_gmin-1, self.adm_gmax):
                self.adm_imojop[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_imojop[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_imojop[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_imojop[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_imojop[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_imojop[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_imojop[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_ImoJmp
        self.adm_imojmp_nmax = ( gall_in+1 ) * ( gall_in+2 )
        self.adm_imojmp = np.zeros((self.adm_imojmp_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax+1):
            for i in range(self.adm_gmin-1, self.adm_gmax):
                self.adm_imojmp[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_imojmp[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_imojmp[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_imojmp[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_imojmp[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_imojmp[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_imojmp[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_IopJoo
        self.adm_iopjoo_nmax = ( gall_in+1 ) * ( gall_in )
        self.adm_iopjoo = np.zeros((self.adm_iopjoo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax):
            for i in range(self.adm_gmin, self.adm_gmax+1):
                self.adm_iopjoo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_iopjoo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_iopjoo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_iopjoo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_iopjoo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_iopjoo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_iopjoo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_IopJmo
        self.adm_iopjmo_nmax = ( gall_in+1 ) * ( gall_in+1 )
        self.adm_iopjmo = np.zeros((self.adm_iopjmo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax):
            for i in range(self.adm_gmin, self.adm_gmax+1):
                self.adm_iopjmo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_iopjmo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_iopjmo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_iopjmo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_iopjmo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_iopjmo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_iopjmo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_IopJop
        self.adm_iopjop_nmax = ( gall_in+1 ) * ( gall_in+1 )
        self.adm_iopjop = np.zeros((self.adm_iopjop_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax+1):
            for i in range(self.adm_gmin, self.adm_gmax+1):
                self.adm_iopjop[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_iopjop[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_iopjop[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_iopjop[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_iopjop[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_iopjop[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_iopjop[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1

        #--- ADM_IopJmp
        self.adm_iopjmp_nmax = ( gall_in+1 ) * ( gall_in+2 )
        self.adm_iopjmp = np.zeros((self.adm_iopjmp_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax+1):
            for i in range(self.adm_gmin, self.adm_gmax+1):
                self.adm_iopjmp[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_iopjmp[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_iopjmp[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_iopjmp[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_iopjmp[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_iopjmp[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_iopjmp[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1

        #--- ADM_ImpJoo
        self.adm_impjoo_nmax = ( gall_in+2 ) * ( gall_in )
        self.adm_impjoo = np.zeros((self.adm_impjoo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax):
            for i in range(self.adm_gmin-1, self.adm_gmax+1):
                self.adm_impjoo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_impjoo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_impjoo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_impjoo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_impjoo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_impjoo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_impjoo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_ImpJmo
        self.adm_impjmo_nmax = ( gall_in+2 ) * ( gall_in+1 )
        self.adm_impjmo = np.zeros((self.adm_impjmo_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax):
            for i in range(self.adm_gmin-1, self.adm_gmax+1):
                self.adm_impjmo[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_impjmo[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_impjmo[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_impjmo[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_impjmo[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_impjmo[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_impjmo[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_ImpJop
        self.adm_impjop_nmax = ( gall_in+2 ) * ( gall_in+1 )
        self.adm_impjop = np.zeros((self.adm_impjop_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin, self.adm_gmax+1):
            for i in range(self.adm_gmin-1, self.adm_gmax+1):
                self.adm_impjop[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_impjop[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_impjop[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_impjop[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_impjop[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_impjop[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_impjop[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1
    
        #--- ADM_ImpJmp
        self.adm_impjmp_nmax = ( gall_in+2 ) * ( gall_in+2 )
        self.adm_impjmp = np.zeros((self.adm_impjmp_nmax, self.adm_gij_nmax), dtype=int)
        n = 0
        for j in range(self.adm_gmin-1, self.adm_gmax+1):
            for i in range(self.adm_gmin-1, self.adm_gmax+1):
                self.adm_impjmp[n, self.adm_giojo] =  self.suf(i  ,j  )
                self.adm_impjmp[n, self.adm_gipjo] =  self.suf(i+1,j  )
                self.adm_impjmp[n, self.adm_gipjp] =  self.suf(i+1,j+1)
                self.adm_impjmp[n, self.adm_giojp] =  self.suf(i  ,j+1)
                self.adm_impjmp[n, self.adm_gimjo] =  self.suf(i-1,j  )
                self.adm_impjmp[n, self.adm_gimjm] =  self.suf(i-1,j-1)
                self.adm_impjmp[n, self.adm_giojm] =  self.suf(i  ,j-1)
                n += 1

    def suf(self, i, j):
        #return self.adm_gall_1d * (j-1) + i
        return self.adm_gall_1d * j + i
 
    def adm_mpitime(self):
        from time import process_time
        #if self.adm_mpi_alive:
        #    return real(MPI_WTIME(), kind=8)
        #else:
        return process_time()

    def adm_proc_stop(self):

        # flush 1kbyte
        self.adm_log_fid.write("                                " * 32)
    
        self.adm_log_fid.write("+++ Abort MPI")
        if self.adm_prc_me == self.adm_prc_run_master:
            print("+++ Abort MPI")
    
        self.adm_log_fid.close()
    
        # Abort MPI
        self.adm_comm_world.Abort() 


    def adm_proc_finish(self):
 
        if self._adm_run_type is not None:
            from mpi4py import MPI
            print("------------")
            print("+++ finalize MPI") 

            self.adm_comm_world.barrier()
            MPI.Finalize() #???

            print("*** MPI is peacefully finalized") 
        else:
            print("------------")
            print("+++ stop serial process.") 

        self.adm_log_fid.close()
