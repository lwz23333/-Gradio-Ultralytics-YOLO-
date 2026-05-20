$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$env:BYSJ_HOST = "127.0.0.1"
$env:BYSJ_PORT = "7860"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = "localhost,127.0.0.1"

$PythonCandidates = @(
    "D:\python\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

if (-not $PythonCandidates) {
    throw "未找到 Python。请先安装 Python 3.11，或把 Python 加入 PATH。"
}

$Python = $PythonCandidates[0]
$Url = "http://127.0.0.1:$($env:BYSJ_PORT)"
$Stdout = Join-Path $ProjectRoot "_app_server_oneclick_stdout.log"
$Stderr = Join-Path $ProjectRoot "_app_server_oneclick_stderr.log"

function Test-ProjectPort {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect("127.0.0.1", [int]$env:BYSJ_PORT, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne(500)
        if ($ok) {
            $client.EndConnect($async)
            $client.Close()
            return $true
        }
        $client.Close()
        return $false
    } catch {
        return $false
    }
}

if (Test-ProjectPort) {
    Start-Process $Url
    Write-Host "项目已经在运行：$Url"
    return
}

Write-Host "正在启动项目：$Url"
Start-Process -FilePath $Python `
    -ArgumentList @("app_server.py") `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Test-ProjectPort) {
        Start-Process $Url
        Write-Host "启动成功：$Url"
        return
    }
}

Write-Host "启动未确认，请查看日志："
Write-Host $Stdout
Write-Host $Stderr
