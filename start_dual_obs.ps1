param(
    [string]$Obs1Exe = "",
    [string]$Obs2Exe = "",
    [switch]$NoPortable,
    [string]$Obs1Args = "",
    [string]$Obs2Args = ""
)

$ErrorActionPreference = "Stop"

function Resolve-ObsExePath {
    param(
        [string]$InputPath
    )
    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        return $null
    }
    $p = $InputPath.Trim('"')
    if (-not [System.IO.Path]::IsPathRooted($p)) {
        $p = Join-Path -Path $PSScriptRoot -ChildPath $p
    }
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        throw "OBS executable not found: $p"
    }
    return (Resolve-Path -LiteralPath $p).Path
}

function Split-Args {
    param(
        [string]$Raw
    )
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return @()
    }
    return [System.Management.Automation.PSParser]::Tokenize($Raw, [ref]$null) |
        Where-Object { $_.Type -eq "CommandArgument" -or $_.Type -eq "String" } |
        ForEach-Object { $_.Content }
}

function Start-ObsInstance {
    param(
        [string]$ExePath,
        [bool]$PortableEnabled,
        [string]$ExtraArgsRaw
    )
    $args = @("--multi")
    if ($PortableEnabled) {
        $args += "--portable"
    }
    $args += Split-Args -Raw $ExtraArgsRaw

    $wd = Split-Path -Path $ExePath -Parent
    $proc = Start-Process -FilePath $ExePath -ArgumentList $args -WorkingDirectory $wd -PassThru
    return [PSCustomObject]@{
        Exe  = $ExePath
        Pid  = $proc.Id
        Args = ($args -join " ")
    }
}

$exe1 = Resolve-ObsExePath -InputPath $Obs1Exe
$exe2 = Resolve-ObsExePath -InputPath $Obs2Exe

if (-not $exe1 -or -not $exe2) {
    $found = Get-ChildItem -Path $PSScriptRoot -Recurse -File -Filter "obs64.exe" |
        Sort-Object FullName |
        Select-Object -ExpandProperty FullName

    if ($found.Count -lt 2) {
        Write-Host "Need two OBS executables under this folder." -ForegroundColor Red
        Write-Host "Place two OBS folders in root, or pass explicit paths:" -ForegroundColor Yellow
        Write-Host "  .\start_dual_obs.ps1 -Obs1Exe '.\obs-a\bin\64bit\obs64.exe' -Obs2Exe '.\obs-b\bin\64bit\obs64.exe'" -ForegroundColor Yellow
        exit 1
    }

    if (-not $exe1) { $exe1 = $found[0] }
    if (-not $exe2) { $exe2 = $found[1] }
}

if ($exe1 -eq $exe2) {
    Write-Host "Obs1Exe and Obs2Exe resolved to the same path. Provide two different OBS installs." -ForegroundColor Red
    exit 1
}

$portableEnabled = -not $NoPortable.IsPresent

Write-Host "[dual-obs] launching instance #1:" -ForegroundColor Cyan
Write-Host "  exe: $exe1"
Write-Host "[dual-obs] launching instance #2:" -ForegroundColor Cyan
Write-Host "  exe: $exe2"
Write-Host "  portable: $portableEnabled"

$r1 = Start-ObsInstance -ExePath $exe1 -PortableEnabled $portableEnabled -ExtraArgsRaw $Obs1Args
$r2 = Start-ObsInstance -ExePath $exe2 -PortableEnabled $portableEnabled -ExtraArgsRaw $Obs2Args

Write-Host ""
Write-Host "[dual-obs] started" -ForegroundColor Green
Write-Host "  OBS1 PID=$($r1.Pid) args=$($r1.Args)"
Write-Host "  OBS2 PID=$($r2.Pid) args=$($r2.Args)"
Write-Host ""
Write-Host "Tips:"
Write-Host "  1) Add per-instance profile/scene args:"
Write-Host "     -Obs1Args '--profile OBS1 --collection LiveA'"
Write-Host "     -Obs2Args '--profile OBS2 --collection LiveB'"
Write-Host "  2) If you want shared (non-portable) OBS config, add -NoPortable."
