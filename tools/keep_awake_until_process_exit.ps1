param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId
)

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class ExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$EsContinuous = [uint32]2147483648
$EsContinuousSystemRequired = [uint32]2147483649

try {
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        [void][ExecutionState]::SetThreadExecutionState($EsContinuousSystemRequired)
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][ExecutionState]::SetThreadExecutionState($EsContinuous)
}
