param(
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [switch]$ExportPdf
)

$ErrorActionPreference = 'Stop'
$manifest = (Resolve-Path -LiteralPath $ManifestPath).Path
$base = Split-Path -Parent $manifest
$items = @((Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json).slides)
$output = [System.IO.Path]::GetFullPath($OutputPath)
$pdf = [System.IO.Path]::ChangeExtension($output, '.pdf')
if ($items.Count -eq 0) { throw 'The manifest must contain at least one slide.' }
if ([System.IO.Path]::GetExtension($output) -ne '.pptx') { throw 'OutputPath must end with .pptx.' }
if ((Test-Path -LiteralPath $output) -or ($ExportPdf -and (Test-Path -LiteralPath $pdf))) {
    throw 'Output already exists. Choose a new output filename.'
}
$images = @($items | ForEach-Object {
    $image = $_.image
    if (-not [System.IO.Path]::IsPathRooted($image)) { $image = Join-Path $base $image }
    (Resolve-Path -LiteralPath $image).Path
})
$outDir = Split-Path -Parent $output
$preview = Join-Path (Join-Path $outDir '.preview') ([System.IO.Path]::GetFileNameWithoutExtension($output))
New-Item -ItemType Directory -Force -Path $outDir, $preview | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Add(0)
    $deck.PageSetup.SlideWidth = 960
    $deck.PageSetup.SlideHeight = 540
    for ($i = 0; $i -lt $images.Count; $i++) {
        $slide = $deck.Slides.Add($i + 1, 12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.Solid()
        $slide.Background.Fill.ForeColor.RGB = 16777215
        $picture = $slide.Shapes.AddPicture($images[$i], 0, -1, 0, 0, -1, -1)
        $picture.LockAspectRatio = -1
        $picture.Width = 960
        if ($picture.Height -gt 540) { $picture.Height = 540 }
        $picture.Left = (960 - $picture.Width) / 2
        $picture.Top = (540 - $picture.Height) / 2
        if ($items[$i].notes) {
            $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $items[$i].notes
        }
    }
    $deck.SaveAs($output, 24)
    if ($ExportPdf) { $deck.SaveAs($pdf, 32) }
    for ($i = 1; $i -le $images.Count; $i++) {
        $slidePath = Join-Path $preview ('slide_{0:D2}.png' -f $i)
        $deck.Slides.Item($i).Export($slidePath, 'PNG', 1920, 1080)
    }
    Write-Output "Created $($deck.Slides.Count) image slides: $output"
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if ($app.Presentations.Count -eq 0) { $app.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
