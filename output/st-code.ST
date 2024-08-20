FUNCTION_BLOCK MAIN
VAR
    MotorEnabled : BOOL;
    MotorRunning : BOOL;

    StartButtonPressed : BOOL;
    StopButtonPressed : BOOL;
END_VAR
BEGIN
    IF StartButtonPressed AND NOT MotorRunning THEN
        MotorEnabled := TRUE;
    END_IF;
    
    IF StopButtonPressed THEN
        MotorEnabled := FALSE;
    END_IF;

    IF MotorEnabled THEN
        MotorEnabled := TRUE;
    END_IF;

END_FUNCTION_BLOCK