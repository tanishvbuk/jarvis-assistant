# How much CPU and memory the HUD window and the Jarvis process use, averaged over 15 seconds. Run it with the window idle:
#   powershell -File scripts/measure_hud_cpu.ps1
# Reference (15 Oct 2026, a test laptop, 16 logical cores): full HUD window 42-54% of one core and ~760 MB at standby (87% before the idle fixes); Jarvis core ~12%.
$all = Get-CimInstance Win32_Process
$roots = $all | Where-Object { $_.CommandLine -match 'jarvis\.hud\.window' } | Select-Object -ExpandProperty ProcessId
$jarvis = $all | Where-Object { $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match '-m jarvis$' -or ($_.CommandLine -match '-m jarvis"?\s*$') } | Select-Object -ExpandProperty ProcessId
function Tree($ids) {
  $set = New-Object System.Collections.Generic.HashSet[int]
  $queue = [System.Collections.Generic.Queue[int]]::new()
  foreach ($i in $ids) { [void]$set.Add($i); $queue.Enqueue($i) }
  while ($queue.Count) { $p = $queue.Dequeue(); foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $p })) { if ($set.Add([int]$c.ProcessId)) { $queue.Enqueue([int]$c.ProcessId) } } }
  return $set
}
$hud = Tree $roots
$core = $all | Where-Object { $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match ' -m jarvis' -and $_.CommandLine -notmatch 'hud\.window' } | Select-Object -ExpandProperty ProcessId
function Snap($ids) { $t = 0.0; $ws = 0; foreach ($i in $ids) { $p = Get-Process -Id $i -ErrorAction SilentlyContinue; if ($p) { $t += $p.TotalProcessorTime.TotalSeconds; $ws += $p.WorkingSet64 } } return @($t, $ws) }
$h0 = Snap $hud; $c0 = Snap $core
Start-Sleep -Seconds 15
$h1 = Snap $hud; $c1 = Snap $core
$cores = [Environment]::ProcessorCount
"HUD window tree ($($hud.Count) processes): {0:N1}% of one core, {1:N0} MB" -f (($h1[0]-$h0[0])/15*100), ($h1[1]/1MB)
"Jarvis core ($($core.Count) processes): {0:N1}% of one core, {1:N0} MB" -f (($c1[0]-$c0[0])/15*100), ($c1[1]/1MB)
"logical cores: $cores"
