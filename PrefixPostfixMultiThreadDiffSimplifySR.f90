module function_space
    use, intrinsic :: iso_c_binding
    implicit none

    ! Define constants for clock_gettime
    integer(c_int), parameter :: CLOCK_REALTIME = 0
    character(len=10), parameter :: unary_operators(13) = &
    ["cos       ", "~         ", "sin       ", "log       ", "ln        ", "asin      ", "arcsin    ", &
     "acos      ", "arccos    ", "exp       ", "sech      ", "tanh      ", "sqrt      "]
    character(len=*), parameter :: binary_operators(5) = ["+", "-", "*", "/", "^"]

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

    logical function is_unary(token) result(isUnary)
        implicit none
        character(len=*), intent(in) :: token
        integer :: i
        isUnary = .false.
        do i = 1, size(unary_operators)
            if (trim(token) == trim(unary_operators(i))) then
                isUnary = .true.
                exit
            end if
        end do
    end function is_unary

    logical function is_binary(token) result(isBinary)
        implicit none
        character(len=*), intent(in) :: token
        integer :: i
        isBinary = .false.
        do i = 1, size(binary_operators)
            if (token == binary_operators(i)) then
                isBinary = .true.
                exit
            end if
        end do
    end function is_binary

    logical function is_const(token) result(isConst)
        implicit none
        character(len=*), intent(in) :: token
        isConst = .not. is_unary(token) .and. .not. is_binary(token)
    end function is_const

    logical function isFloat(s)
        implicit none
        character(len=*), intent(in) :: s
        integer :: i, len, state
        logical :: has_digits

        ! Define states
        integer, parameter :: START = 0, INT = 1, FRAC = 2, EXP = 3, EXP_NUM = 4

        ! Initialization
        len = len_trim(s)
        state = START
        has_digits = .false.

        do i = 1, len
            select case (state)
                case (START)
                    if (s(i:i) == '+' .or. s(i:i) == '-') then
                        state = INT
                    else if (s(i:i) >= '0' .and. s(i:i) <= '9') then
                        state = INT
                        has_digits = .true.
                    else if (s(i:i) == '.') then
                        state = FRAC
                    else
                        isFloat = .false.
                        return
                    end if
                case (INT)
                    if (s(i:i) >= '0' .and. s(i:i) <= '9') then
                        has_digits = .true.
                    else if (s(i:i) == '.') then
                        state = FRAC
                    else if (s(i:i) == 'e' .or. s(i:i) == 'E') then
                        state = EXP
                    else
                        isFloat = .false.
                        return
                    end if
                case (FRAC)
                    if (s(i:i) >= '0' .and. s(i:i) <= '9') then
                        has_digits = .true.
                    else if (s(i:i) == 'e' .or. s(i:i) == 'E') then
                        state = EXP
                    else
                        isFloat = .false.
                        return
                    end if
                case (EXP)
                    if (s(i:i) == '+' .or. s(i:i) == '-') then
                        state = EXP_NUM
                    else if (s(i:i) >= '0' .and. s(i:i) <= '9') then
                        state = EXP_NUM
                    else
                        isFloat = .false.
                        return
                    end if
                case (EXP_NUM)
                    if (s(i:i) < '0' .or. s(i:i) > '9') then
                        isFloat = .false.
                        return
                    end if
            end select
        end do

        isFloat = has_digits .and. (state == INT .or. state == FRAC .or. state == EXP_NUM)
    end function isFloat

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

    function simplifyString(x) result(simplified)
        implicit none
        character(len=*), intent(in) :: x
        character(len=len(x)) :: simplified
        integer :: dotPos, i
        logical :: allZerosAfterDot

        ! Handle the case "-0" -> "0"
        if (len(x) == 2 .and. x(1:1) == '-' .and. x(2:2) == '0') then
            simplified = "0"
            return
        end if

        ! Find the position of the decimal point
        dotPos = index(x, '.')

        ! If there is no decimal point, return the input as is
        if (dotPos == 0) then
            simplified = x
            return
        end if

        ! Check if all characters after the decimal point are '0'
        allZerosAfterDot = .true.
        do i = dotPos + 1, len(x)
            if (x(i:i) /= '0') then
                allZerosAfterDot = .false.
                exit
            end if
        end do

        ! If there are non-zero characters after the decimal, return the input as is
        if (.not. allZerosAfterDot) then
            simplified = x
            return
        end if

        ! Extract the part before the decimal point
        simplified = x(1:dotPos - 1)

        ! Handle the case "-0.0000" -> "0"
        if (len(simplified) == 2 .and. simplified(1:1) == '-' .and. simplified(2:2) == '0') then
            simplified = "0"
        end if

    end function simplifyString

    subroutine print_container(c)
        implicit none
        character(len=*), dimension(:), intent(in) :: c
        integer :: i

        do i = 1, size(c)
            write(*, '(A,1X)', advance="no") trim(c(i))
        end do
        write(*, *)  ! Newline after printing all elements
    end subroutine print_container

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
    character(len=100) :: string_result
    character(len=15) :: token

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

    ! Test cases
    string_result = simplifyString("-0")
    print *, "Input: '-0', Output: ", trim(string_result), " Expected: '0'"

    string_result = simplifyString("123.000000")
    print *, "Input: '123.000000', Output: ", trim(string_result), " Expected: '123'"

    string_result = simplifyString("0.1")
    print *, "Input: '0.1', Output: ", trim(string_result), " Expected: '0.1'"

    string_result = simplifyString("123.004500")
    print *, "Input: '123.004500', Output: ", trim(string_result), " Expected: '123.004500'"

    string_result = simplifyString("0.000000")
    print *, "Input: '0.000000', Output: ", trim(string_result), " Expected: '0'"

    string_result = simplifyString("-123.0045")
    print *, "Input: '-123.0045', Output: ", trim(string_result), " Expected: '-123.0045'"



    ! Test cases for is_unary
    print *, "Testing is_unary function:"
    do i = 1, size(unary_operators)
        token = trim(unary_operators(i))  ! Use a valid unary operator
        resultVal = is_unary(token)
        print *, "Input:", token, "-> is_unary:", resultVal
    end do

    token = "invalid"
    resultVal = is_unary(token)
    print *, "Input: 'invalid' -> is_unary:", resultVal

    ! Test cases for is_binary
    print *, "Testing is_binary function:"
    do i = 1, size(binary_operators)
        token = trim(binary_operators(i))  ! Use a valid binary operator
        resultVal = is_binary(token)
        print *, "Input:", token, "-> is_binary:", resultVal
    end do

    token = "amp"
    resultVal = is_binary(token)
    print *, "Input: 'amp' -> is_binary:", resultVal

    print *, "Testing is_const function:"
    print *, "Input: 'cos' -> is_const:", is_const("cos")       ! Expected: F
    print *, "Input: '+'   -> is_const:", is_const("+")         ! Expected: F
    print *, "Input: '123' -> is_const:", is_const("123")       ! Expected: T
    print *, "Input: 'log' -> is_const:", is_const("log")       ! Expected: F
    print *, "Input: 'xyz' -> is_const:", is_const("xyz")       ! Expected: T

    print *, isFloat("123.45")
    print *, isFloat("1.2e3")
    print *, isFloat("abc")
    print *, isFloat(".5")

    call print_container(unary_operators)
    call print_container(binary_operators)

    ! Calculate elapsed time
    elapsed_time = time_elapsed(start_time)

    ! Output the elapsed time
    print *, "Elapsed time (in seconds): ", elapsed_time

end program main

!gfortran PrefixPostfixMultiThreadDiffSimplifySR.f90 -o PrefixPostfixMultiThreadDiffSimplifySR


