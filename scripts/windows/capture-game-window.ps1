[CmdletBinding()]
param([Parameter(Mandatory=$true)][int]$GameProcessId, [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
if (!('D4RWindowCapture' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class D4RWindowCapture {
    public delegate bool Callback(IntPtr window, IntPtr data);
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool EnumWindows(Callback callback, IntPtr data);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr window, StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
}
'@
}
$windows = [Collections.Generic.List[object]]::new()
$previousDpi=[D4RWindowCapture]::SetThreadDpiAwarenessContext([IntPtr](-4))
try {
$callback = [D4RWindowCapture+Callback]{ param($window,$data)
    [uint32]$owner = 0
    [void][D4RWindowCapture]::GetWindowThreadProcessId($window,[ref]$owner)
    if ($owner -eq $GameProcessId) {
        $rect = [D4RWindowCapture+Rect]::new()
        [void][D4RWindowCapture]::GetWindowRect($window,[ref]$rect)
        $title = [Text.StringBuilder]::new(512)
        [void][D4RWindowCapture]::GetWindowText($window,$title,512)
        $windows.Add([pscustomobject]@{handle=$window; title=$title.ToString(); visible=[D4RWindowCapture]::IsWindowVisible($window); rect=$rect;
            area=($rect.Right-$rect.Left)*($rect.Bottom-$rect.Top)})
    }
    return $true
}
[void][D4RWindowCapture]::EnumWindows($callback,[IntPtr]::Zero)
$windows | Select-Object handle,title,visible,area | Format-Table
$chosen = $windows | Where-Object { $_.visible -and $_.area -gt 4096 } | Sort-Object area -Descending | Select-Object -First 1
if (!$chosen) { throw "No visible game window for PID $GameProcessId" }
if ([D4RWindowCapture]::GetForegroundWindow() -ne $chosen.handle) {
    throw 'Game window is not foreground; capture refused to avoid including unrelated overlapping windows.'
}
$rect = $chosen.rect
$bitmap = [Drawing.Bitmap]::new($rect.Right-$rect.Left,$rect.Bottom-$rect.Top)
$graphics = [Drawing.Graphics]::FromImage($bitmap)
try {
    $graphics.CopyFromScreen($rect.Left,$rect.Top,0,0,$bitmap.Size)
    $bitmap.Save([IO.Path]::GetFullPath($OutputPath),[Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose(); $bitmap.Dispose()
}
Write-Host "Captured game window: $OutputPath"
} finally {
    if ($previousDpi -ne [IntPtr]::Zero) { [void][D4RWindowCapture]::SetThreadDpiAwarenessContext($previousDpi) }
}
