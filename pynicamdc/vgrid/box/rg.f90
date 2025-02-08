program rg
  implicit none

  character(LEN=1) :: c1
  character(LEN=4) :: c4
  integer :: a, i
  real :: r
  
  open(10,file='vgrid40.dat',form='formatted')

  read(10,*) c1
  write(6,*) c1

    read(10,*) c1
  write(6,*) c1

    read(10,*) c1
  write(6,*) c1

    read(10,*) c1
  write(6,*) c1

    read(10,*) c1
  write(6,*) c1

    read(10,*) c1
  write(6,*) c1

  
  read(10,'(I4)') a 
  write(6,*) a

  do i=1,40
     read(10,'(f12.3)') r
     write(6,*) r
  enddo

  close(10)

end program rg
