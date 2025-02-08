!-------------------------------------------------------------------------------
!
!+  Program showvlayer
!
!-------------------------------------------------------------------------------
program prg_showvlayer
  !-----------------------------------------------------------------------------
  !
  !++ Description: 
  !       This program shows the vertical grid system as ascii dump.
  ! 
  !++ Current Corresponding Author : H.Tomita
  ! 
  !++ History: 
  !      Version   Date       Comment 
  !      -----------------------------------------------------------------------
  !      0.00      04-02-17   Imported from igdc-4.34
  !      -----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------------
  !
  !++ Used modules ( shared )
  !
  !=============================================================================
  integer, parameter :: kdum=1
  integer, parameter :: fid=11
  integer :: num_of_layer
  character(len=256) :: infname = ''

  namelist / showvlayer_cnf / &
       infname                  !--- input file name

  real(8),allocatable :: z_c(:)
  real(8),allocatable :: z_h(:)

  integer :: kmin
  integer :: kmax
  integer :: kall

  open(fid,file='showvlayer.cnf',status='old',form='formatted')
  read(fid,nml=showvlayer_cnf)
  close(fid)

  !
  call input_layer(infname)

  !=============================================================================
contains
  !-----------------------------------------------------------------------------
  subroutine input_layer( infname )
    implicit none

    character(len=*) :: infname
    integer :: fid=10
    integer :: k
    open(fid,file=trim(infname),status='old',form='unformatted')
    read(fid) num_of_layer

    kmin=kdum+1
    kmax=kdum+num_of_layer
    kall=kdum+num_of_layer+kdum
    allocate(z_c(kall))
    allocate(z_h(kall)) 

    read(fid) z_c
    read(fid) z_h
    close(fid)
    
    do k = 1, kall
       write(*,*) k, z_h(k), z_c(k)
    enddo
  end subroutine input_layer
  !-----------------------------------------------------------------------------
end program prg_showvlayer
!-------------------------------------------------------------------------------
