module timing_module
    use, intrinsic :: iso_c_binding
    implicit none

    ! Define constants for clock_gettime
    integer(c_int), parameter :: CLOCK_REALTIME = 0

    type, bind(c) :: timespec
        integer(c_long) :: tv_sec  ! seconds
        integer(c_long) :: tv_nsec ! nanoseconds
    end type timespec

    interface
        function clock_gettime(clk_id, tp) bind(C, name="clock_gettime")
            import :: c_int, timespec
            integer(c_int) :: clock_gettime
            integer(c_int), value :: clk_id
            type(timespec) :: tp
        end function clock_gettime
    end interface

contains

    function get_time() result(time)
        type(timespec) :: tp
        real(8) :: time
        integer(c_int) :: ierr

        ! Get current time with nanosecond precision
        ierr = clock_gettime(CLOCK_REALTIME, tp)
        if (ierr /= 0) then
            print *, "Error in clock_gettime"
            stop
        end if

        ! Convert time to seconds
        time = real(tp%tv_sec, kind=8) + real(tp%tv_nsec, kind=8) * 1.0d-9
    end function get_time

    function time_elapsed(start_time) result(elapsed_time)
        real(8), intent(in) :: start_time
        real(8) :: elapsed_time

        ! Calculate elapsed time as difference in seconds
        elapsed_time = get_time() - start_time
    end function time_elapsed

end module timing_module

program main
    use timing_module
    implicit none
    real(8) :: start_time, elapsed_time

    ! Capture start time
    start_time = get_time()

    ! Add some delay for demonstration
    call sleep(1)

    ! Calculate elapsed time
    elapsed_time = time_elapsed(start_time)

    ! Output the elapsed time
    print *, "Elapsed time (in seconds): ", elapsed_time

end program main
