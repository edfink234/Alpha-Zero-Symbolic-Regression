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

    function create_linspace_matrix(rows, cols, min_vec, max_vec) result(mat)
        implicit none
        integer, intent(in) :: rows, cols
        real(4), intent(in) :: min_vec(cols), max_vec(cols)
        real(4) :: mat(rows, cols)
        integer :: row, col

        do col = 1, cols
            do row = 1, rows
                mat(row, col) = min_vec(col) + (max_vec(col) - min_vec(col)) * (row - 1) / real(rows - 1, 8)
            end do
        end do

    end function create_linspace_matrix

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

    FUNCTION isConstant(vec, sz, tolerance) RESULT(res)
        IMPLICIT NONE
        INTEGER, INTENT(IN) :: sz
        REAL(4), INTENT(IN) :: vec(sz)
        REAL(4), INTENT(IN), OPTIONAL :: tolerance
        REAL(4) :: tol, mean_val, var
        INTEGER :: i
        LOGICAL :: res
    
        ! Set default tolerance if not provided
        tol = 1.0E-5
        IF (PRESENT(tolerance)) THEN
            tol = tolerance
        END IF

        ! A vector with 0 or 1 element is trivially constant
        IF (sz <= 1) THEN
            res = .TRUE.
            RETURN
        END IF

        ! Check for NaN or Inf values
        DO i = 1, sz-2
            IF (isInvalid(vec(i))) THEN
                res = .TRUE.
                RETURN
            END IF
        END DO

        ! Compute variance
        mean_val = SUM(vec) / REAL(sz)
        var = SUM((vec - mean_val)**2) / REAL(sz)

        ! Check if variance is within tolerance
        res = (var <= tol)
    END FUNCTION isConstant

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

    function createMeshgridVectors(rows, cols, min_vec, max_vec) result(matrix)
        implicit none
        integer, intent(in) :: rows, cols
        real, intent(in) :: min_vec(cols), max_vec(cols)
        real, allocatable :: matrix(:,:)
        real, allocatable :: linspaces(:,:)
        integer, allocatable :: repeat_count(:)
        integer :: total_combinations, col, i, j, repeat, index, num_repeats

        ! Compute total number of combinations
        total_combinations = rows**cols

        ! Allocate matrix for output
        allocate(matrix(total_combinations, cols))
        allocate(linspaces(rows, cols))
        allocate(repeat_count(cols))

        ! Generate linspaces
        do col = 1, cols
            linspaces(:, col) = linspace(min_vec(col), max_vec(col), rows)
        end do

        ! Compute repeat_count for each column
        repeat_count(cols) = 1
        do col = cols - 1, 1, -1
            repeat_count(col) = repeat_count(col + 1) * rows
        end do

        ! Fill the matrix
        do col = 1, cols
            num_repeats = total_combinations / (repeat_count(col) * rows)
            index = 1
            do repeat = 1, num_repeats
                do i = 1, rows
                    do j = 1, repeat_count(col)
                        matrix(index, col) = linspaces(i, col)
                        index = index + 1
                    end do
                end do
            end do
        end do

        ! Clean up
        deallocate(linspaces, repeat_count)

    end function createMeshgridVectors

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

    subroutine print_range_of_container(c, low, up)
        character(len=*), dimension(:), intent(in) :: c
        integer, intent(in) :: low, up
        integer :: i

        do i = low, up
            write(*, '(A)', advance='no') trim(c(i))
            write(*, '(A)', advance='no') ' '
        end do
        write(*, *)  ! Print newline
    end subroutine print_range_of_container

    RECURSIVE SUBROUTINE GB(z, ind, individual, expression_type)
        IMPLICIT NONE
        INTEGER, INTENT(INOUT) :: ind
        INTEGER, INTENT(IN) :: z
        CHARACTER(LEN=*), DIMENSION(:), INTENT(IN) :: individual
        CHARACTER(LEN=*), INTENT(IN) :: expression_type

        INTEGER :: remaining_z

        remaining_z = z
        DO WHILE (remaining_z > 0)
           ! Update `ind` based on the expression type
           IF (TRIM(expression_type) == "prefix") THEN
              ind = ind + 1
           ELSE
              ind = ind - 1
           END IF

           ! Check if the current token is unary or binary
           IF (is_unary(individual(ind))) THEN
              CALL GB(1, ind, individual, expression_type)
           ELSE IF (is_binary(individual(ind))) THEN
              CALL GB(2, ind, individual, expression_type)
           END IF

           remaining_z = remaining_z - 1
        END DO
    END SUBROUTINE GB

    integer function GR(i, individual, expression_type) result(gr_value)
        implicit none
        integer, intent(in)    :: i
        character(len=*), dimension(:), intent(in) :: individual
        character(len=*), intent(in) :: expression_type

        integer :: start, ptr_lgb

        start   = i
        ptr_lgb = start

        ! If current symbol is unary, parse 1 operand; if binary, parse 2 operands
        if (is_unary(individual(i))) then
            call GB(1, ptr_lgb, individual, expression_type)
        else if (is_binary(individual(i))) then
            call GB(2, ptr_lgb, individual, expression_type)
        end if

        ! For prefix, final difference is (ptr_lgb - i)
        ! For postfix, final difference is (i - ptr_lgb)
        if (expression_type == 'prefix') then
            gr_value = ptr_lgb - i
        else if (expression_type == 'postfix') then
            gr_value = i - ptr_lgb
        else
            gr_value = 0
        end if

    end function GR

    subroutine setPrefixGR(prefix, grasp)
        implicit none
        character(len=*), dimension(:), intent(in)  :: prefix
        integer, allocatable, dimension(:), intent(out) :: grasp

        integer :: k, n

        n = size(prefix)
        allocate(grasp(n))

        do k = 1, n
            grasp(k) = GR(k, prefix, 'prefix')
        end do
    end subroutine setPrefixGR

    subroutine setPostfixGR(postfix, grasp)
        implicit none
        character(len=*), dimension(:), intent(in)  :: postfix
        integer, allocatable, dimension(:), intent(out) :: grasp

        integer :: k, n

        n = size(postfix)
        allocate(grasp(n))

        do k = 1, n
            grasp(k) = GR(k, postfix, 'postfix')
        end do
    end subroutine setPostfixGR

    function areExpressionRangesEqual(start_idx_1, start_idx_2, num_steps, expression) result(is_equal)
        implicit none
        integer, intent(in) :: start_idx_1, start_idx_2, num_steps
        character(len=*), intent(in) :: expression(:)
        logical :: is_equal
        integer :: i, j, stop_idx_1

        stop_idx_1 = start_idx_1 + num_steps - 1
        is_equal = .true.

        do i = start_idx_1, stop_idx_1
            j = start_idx_2 + (i - start_idx_1)
            if (expression(i) /= expression(j)) then
                is_equal = .false.
                return
            end if
        end do
    end function areExpressionRangesEqual

    function trueMod(N, M) result(modulo)
        implicit none
        integer, intent(in) :: N  ! Numerator
        integer, intent(in) :: M  ! Denominator
        integer :: modulo          ! Result of the modulo operation

        ! Calculate true modulo
        modulo = mod(mod(N, M) + M, M)
    end function trueMod

    function MSE(actual) result(mse_value)
        implicit none
        real, intent(in) :: actual(:)
        real :: mse_value

        mse_value = sum(actual**2)
    end function MSE

    function loss_func(actual) result(loss_value)
        implicit none
        real, intent(in) :: actual(:)
        real :: loss_value

        loss_value = 1.0/(1.0 + MSE(actual))
    end function loss_func

    function MSE_actual_predicted(actual, predicted, n) result(mse_value)
        implicit none
        integer, intent(in) :: n
        real, intent(in) :: actual(n), predicted(n)
        real :: mse_value

        ! Compute Mean Squared Error (MSE)
        mse_value = sum((actual - predicted) ** 2) / dble(n)

    end function MSE_actual_predicted

    function loss_func_actual_predicted(actual, predicted, n) result(loss) 
        implicit none
        integer, intent(in) :: n
        real, intent(in) :: actual(n), predicted(n)
        real :: loss

        ! Compute Mean Squared Error (MSE)
        real :: mse
        mse = sum((actual - predicted) ** 2) / n

        ! Compute the loss function
        loss = 1.0d0 / (1.0d0 + mse)

    end function loss_func_actual_predicted

    function MSE_MATRIX(actual, n) result(mse)
        implicit none
        integer, intent(in) :: n  ! Number of vectors
        real, intent(in) :: actual(:,:)  ! 2D array where each column is a vector
        real :: mse
        integer :: i

        mse = 0.0
        do i = 1, n
            mse = mse + sum(actual(:, i) ** 2)
        end do
    end function MSE_MATRIX

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

    real function sum_row(row)
        real, intent(in) :: row(:)
        sum_row = sum(row)
    end function sum_row


end module function_space

module matrix_utils
    implicit none

    interface
        real function row_function(row)
            real, intent(in) :: row(:)
        end function row_function
    end interface

contains

    function add_column_with_lambda(matrix, lambda) result(new_matrix)
        real, intent(in) :: matrix(:, :)
        procedure(row_function) :: lambda
        real, allocatable :: new_matrix(:, :)
        integer :: rows, cols, i

        rows = size(matrix, 1)
        cols = size(matrix, 2)

        ! Allocate new matrix with an additional column
        allocate(new_matrix(rows, cols + 1))

        ! Copy the original matrix into the new matrix (without the last column)
        new_matrix(:, 1:cols) = matrix

        ! Apply the lambda function to each row and store the result in the last column
        do i = 1, rows
            new_matrix(i, cols + 1) = lambda(matrix(i, :))
        end do
    end function add_column_with_lambda

end module matrix_utils

program main
    use function_space
    use matrix_utils
    implicit none
    real(8) :: start_time, elapsed_time
    real :: Val
    logical(4) :: resultVal
    real, dimension(:), allocatable :: result
    integer :: i, j, low, up, rows, cols
    character(len=100) :: string_result
    character(len=15) :: token
    CHARACTER(LEN=15), DIMENSION(:), ALLOCATABLE :: individual
    CHARACTER(LEN=10) :: expression_type
    INTEGER :: ind, z
    real(4), allocatable :: mat(:, :), min_vec(:), max_vec(:)
    real :: matrix(3, 2)
    real, allocatable :: new_matrix(:, :)
    integer, allocatable, dimension(:) :: grasp

    ! Capture start time
    start_time = get_time()
    print *, "Start time =", start_time

    ! Call the linspace function
    result = linspace(0.0, 10.0, 5)

    ! Print the result
    print *, "Result ="
    do i = 1, size(result)
        print *, result(i)
    end do

    print *, "Variance =", Variance(result)

    print *, "All below 1e5?", areAllBelow(result, 1e5)

    print *, "Are all the same?", areAllSimilar(result, 1e-5)

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

    ! Define the range to print
    low = 2
    up = 4

    ! Call the subroutine to print the specified range
    call print_range_of_container(unary_operators, low, up)
    call print_range_of_container(binary_operators, low, up)

    ! Test Case 1: Prefix expression
    PRINT *, "Running Test Case 1: Prefix Expression"
    ALLOCATE(individual(7))
    individual = (/"+   ", "tanh", "cos ", "x   ", "^   ", "0   ", "x   "/)
    expression_type = "prefix"
    ind = 1  ! Initial index (Fortran is 1-based)
    z = 2    ! Start with a single expression

    CALL GB(z, ind, individual, expression_type)
    PRINT *, "Final index (ind) after processing: ", ind
    DEALLOCATE(individual)

    ! Test Case 2: Postfix expression
    PRINT *, "Running Test Case 2: Postfix Expression"
    ALLOCATE(individual(9))
    individual = (/"x   ", "x   ", "+   ", "cos ", "cos ", "sin ", "tanh", "0   ", "-   " /)
    expression_type = "postfix"
    ind = 4  ! Initial index for postfix (processing starts at the end)
    z = 1    ! Start with a single expression

    CALL GB(z, ind, individual, expression_type)
    PRINT *, "Final index (ind) after processing: ", ind
    DEALLOCATE(individual)
    ALLOCATE(individual(15))
    individual = (/"x   ", "x   ", "+   ", "cos ", "cos ", "sin ", "tanh", "x   ", "x   ", "+   ", "cos ", "cos ", "sin ", "tanh", "-   " /)

    PRINT *, "Expression: "
    call print_container(individual)
    PRINT *, "areExpressionRangesEqual(1, 8, 6, individual) = ", areExpressionRangesEqual(1, 8, 6, individual)
    DEALLOCATE(individual)

    rows = 5
    cols = 3
    allocate(min_vec(cols), max_vec(cols), mat(rows, cols))

    min_vec = (/ 0.0d0, 1.0d0, 2.0d0 /)
    max_vec = (/ 10.0d0, 5.0d0, 8.0d0 /)

    mat = create_linspace_matrix(rows, cols, min_vec, max_vec)

    ! Print matrix
    print *, "mat = "
    do i = 1, rows
        write(*, '(3F10.3)') (mat(i, j), j = 1, cols)
    end do

    print *, "MSE(mat) = ", MSE_MATRIX(mat, rows)

    print *, "isConstant(min_vec) = ", isConstant(min_vec, 3)
    do i = 1, size(min_vec)
        min_vec(i) = 3
    end do
    print *, "isConstant(min_vec) = ", isConstant(min_vec, 3)

    min_vec = (/ 0.0d0, 1.0d0, 2.0d0 /)
    max_vec = (/ 10.0d0, 5.0d0, 8.0d0 /)
    mat = createMeshgridVectors(10, 3, min_vec, max_vec);
    print *, "createMeshgridVectors mat = "
    do i = 1, size(mat, 1)
        print *, mat(i, :)
    end do

    deallocate(min_vec, max_vec, mat)

    print *, "MSE(result): ", MSE(result)
    print *, "loss_func(result): ", loss_func(result)
    ! Deallocate the array
    deallocate(result)

    ! Initialize the matrix with some values
    matrix = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])

    ! Print the original matrix
    print *, "Original Matrix:"
    do i = 1, size(matrix, 1)
        print *, matrix(i, :)
    end do

    ! Call the function to add a new column
    new_matrix = add_column_with_lambda(matrix, sum_row)

    ! Print the new matrix with the added column
    print *, "New Matrix with Added Column:"
    do i = 1, size(new_matrix, 1)
        print *, new_matrix(i, :)
    end do

    print *, "new_matrix(0, :)"
    print *, new_matrix(0, :)
    print *, "new_matrix(1, :)"
    print *, new_matrix(1, :)
    ! (1^2 + 1^2 + 1^2)/3 = 3/3 = 1
    print *, "MSE_actual_predicted(new_matrix(0, :), new_matrix(1, :), 3):", MSE_actual_predicted(new_matrix(0, :), new_matrix(1, :), 3)
    print *, "loss_func_actual_predicted(new_matrix(0, :), new_matrix(1, :), 3):", loss_func_actual_predicted(new_matrix(0, :), new_matrix(1, :), 3)

    ALLOCATE(individual(12))

    individual = [ &
        '+  ', &
        '-  ', '+  ', 'x  ', 'y  ', 'z  ', &
        'cos', '-  ', '+  ', 'x  ', 'y  ', 'z  ' &
    ]
    allocate(grasp(size(individual)))
    call setPrefixGR(individual, grasp)

    print *, ""
    print *, "=== Test: Prefix expression (+ - + x y z cos - + x y z) ==="
    do i = 1, 12
        print *, "Element: ", individual(i), "  GR(", i, "): ", GR(i, individual, 'prefix'), " grasp(", i, "): ", grasp(i)
    end do

    individual = [&
        'x  ', 'y  ', '+  ', 'z  ', '-  ', &
        'x  ', 'y  ', '+  ', 'z  ', '-  ', 'cos', &
        '+  ' &
    ]

    DEALLOCATE(grasp)
    allocate(grasp(size(individual)))
    call setPostfixGR(individual, grasp)
    print *, ""
    print *, "=== Test: Postfix expression (x y + z - x y + z - cos +) ==="
    do i = 1, 12
        print *, "Element: ", individual(i), "  GR(i): ", GR(i, individual, 'postfix'), " grasp(", i, "): ", grasp(i)
    end do

    DEALLOCATE(individual)

    ! Calculate elapsed time
    elapsed_time = time_elapsed(start_time)
    print *, "End time =", get_time()

    ! Output the elapsed time
    print *, "Elapsed time (in seconds): ", elapsed_time

end program main

!gfortran PrefixPostfixMultiThreadDiffSimplifySR.f90 -o PrefixPostfixMultiThreadDiffSimplifySR
