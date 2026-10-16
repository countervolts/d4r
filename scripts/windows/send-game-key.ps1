[CmdletBinding()]
param([Parameter(Mandatory=$true)][int]$GameProcessId,
    [ValidateSet('Enter','Escape','Up','Down','Tab','Space')][string]$Key='Enter')
$ErrorActionPreference='Stop'
$process=Get-Process -Id $GameProcessId -ErrorAction Stop
if ($process.ProcessName -eq 'GhostOfTsushima') {
    $expected='C:\Program Files (x86)\Steam\steamapps\common\Ghost of Tsushima DIRECTOR''S CUT\GhostOfTsushima.exe'
    if ($process.Path -ine $expected) { throw 'Ghost of Tsushima process path does not match the authorized Steam installation.' }
} elseif ($process.ProcessName -ne 'SHProto-Win64-Shipping') {
    throw 'This helper only accepts the authorized Ghost of Tsushima or Silent Hill 2 diagnostic process.'
}
if (!('D4RGameKeys' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class D4RGameKeys {
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr key, IntPtr flags);
    [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint kind);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
'@
}
$window=$process.MainWindowHandle
if ($window -eq [IntPtr]::Zero) { throw 'Game window missing on this desktop.' }
if ([D4RGameKeys]::GetForegroundWindow() -ne $window) { throw 'Target game is not foreground; key was not sent.' }
$codes=@{Enter=13; Escape=27; Up=38; Down=40; Tab=9; Space=32}
$code=$codes[$Key]; $scan=[D4RGameKeys]::MapVirtualKey($code,0)
$flags=[long](1 -bor ($scan -shl 16))
if ($Key -in @('Up','Down')) { $flags=$flags -bor 0x01000000 }
if (![D4RGameKeys]::PostMessage($window,0x100,[IntPtr]$code,[IntPtr]$flags)) { throw 'Game key-down post failed.' }
if (![D4RGameKeys]::PostMessage($window,0x101,[IntPtr]$code,[IntPtr]($flags -bor 0xc0000000L))) { throw 'Game key-up post failed.' }
Write-Host "Sent $Key only to diagnostic game PID $GameProcessId."
