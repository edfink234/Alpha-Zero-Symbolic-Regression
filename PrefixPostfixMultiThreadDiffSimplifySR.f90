module function_space
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

    function linspace(min_val, max_val, num_points) result(linspaced)
        implicit none
        real, intent(in) :: min_val   ! Minimum value of the range
        real, intent(in) :: max_val   ! Maximum value of the range
        integer, intent(in) :: num_points ! Number of points
        real, dimension(:), allocatable :: linspaced ! Resulting array
        real :: step
        integer :: i

        ! Allocate the resulting array
        allocate(linspaced(num_points))

        ! Compute the step size
        step = (max_val - min_val) / real(num_points - 1)

        ! Fill the array with linearly spaced values
        do i = 1, num_points
            linspaced(i) = min_val + (i - 1) * step
        end do
    end function linspace

    function trueMod(N, M) result(modulo)
        implicit none
        integer, intent(in) :: N  ! Numerator
        integer, intent(in) :: M  ! Denominator
        integer :: modulo          ! Result of the modulo operation

        ! Calculate true modulo
        modulo = mod(mod(N, M) + M, M)
    end function trueMod

    logical function isInvalid(x)
        implicit none
        real :: x
        real, parameter :: huge_value = 1.0e30 ! Replace with a large value representing infinity

        ! Check for NaN (x is not equal to itself) and infinity (absolute value is too large)
        isInvalid = (x /= x) .or. (abs(x) > huge_value)
    end function isInvalid

    logical function areAllBelow(arr, til)
        implicit none
        real, dimension(:), intent(in) :: arr  ! Input array
        real, intent(in) :: til                ! Threshold value
        integer :: i                           ! Loop index

        areAllBelow = .true.                   ! Assume all values are below initially

        ! Check each element in the array
        do i = 1, size(arr)
            if (arr(i) >= til) then
                areAllBelow = .false.          ! If any value is above or equal to the threshold
                exit                           ! Exit loop early
            end if
        end do
    end function areAllBelow

    function Variance(vec) result(var)
        real, dimension(:), intent(in) :: vec
        real :: var
        real :: mean_val
        integer :: n, i

        n = size(vec)
        mean_val = sum(vec) / real(n)
        var = 0.0

        do i = 1, n
            var = var + (vec(i) - mean_val)**2
        end do

        var = var / real(n)
    end function Variance

    logical function areAllSimilar(arr, tol)
        implicit none
        real, dimension(:), intent(in) :: arr  ! Input array
        real, intent(in) :: tol                ! Tolerance value
        integer :: i, j                        ! Loop indices

        areAllSimilar = .true.                 ! Assume all values are similar initially

        ! Compare each pair of elements
        do i = 1, size(arr) - 1
            do j = i + 1, size(arr)
                if (abs(arr(i) - arr(j)) > tol) then
                    areAllSimilar = .false.    ! If difference exceeds tolerance
                    exit                       ! Exit early
                end if
            end do
            if (.not. areAllSimilar) exit      ! Exit outer loop if already false
        end do
    end function areAllSimilar


end module function_space

program main
    use function_space
    implicit none
    real(8) :: start_time, elapsed_time
    real :: Val
    logical(4) :: resultVal
    real, dimension(:), allocatable :: result
    integer :: i

    ! Capture start time
    start_time = get_time()

    ! Call the linspace function
    result = linspace(0.0, 10.0, 5)

    ! Print the result
    do i = 1, size(result)
        print *, result(i)
    end do

    print *, "Variance =", Variance(result)

    print *, "All below 1e5?", areAllBelow(result, 1e5)

    print *, "Are all the same?", areAllSimilar(result, 1e-5)

    ! Deallocate the array
    deallocate(result)

    i = trueMod(-7, 5)

    print *, "trueMod(-7, 5) = ", i  ! Expected output: 3

    ! Example values to test
    Val = 1e31         ! NaN
    resultVal = isInvalid(Val)
    print *, "Value: ", Val, " isInvalid: ", resultVal

    Val = (1.0 / 1e-40)         ! Infinity
    resultVal = isInvalid(Val)
    print *, "Value: ", Val, " isInvalid: ", resultVal

    ! Calculate elapsed time
    elapsed_time = time_elapsed(start_time)

    ! Output the elapsed time
    print *, "Elapsed time (in seconds): ", elapsed_time

end program main

!gfortran PrefixPostfixMultiThreadDiffSimplifySR.f90 -o PrefixPostfixMultiThreadDiffSimplifySR


