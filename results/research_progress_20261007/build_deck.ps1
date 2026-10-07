$ErrorActionPreference = 'Stop'
$out = $PSScriptRoot
$check = Join-Path $out '.build'
New-Item -ItemType Directory -Force -Path $check | Out-Null
$images = @(
    '01_background.png',
    '02_surrogate_method_final.png',
    '03_grouping_method.png',
    '04_surrogate_results.png',
    '05_grouping_results.png'
)
$notes = @(
    'Sources: README.md; project structural design and evaluation code. Building illustration is schematic.',
    'Sources: src/analysis/surrogate_data.py; surrogate_model.py; evaluator.py; src/optimization/parallel_ga.py; grouped_workflow.py. All diagrams are schematic.',
    'Sources: src/structure/section_grouping/discovery.py; application.py; features.py. Grouping illustration is schematic.',
    'Sources: benchmark.json; results/surrogate_data/engineering_230k_group_v2_tail/sigma_reduced_v2.json; results/surrogate_data/engineering_200k_group/delta_single_group_v1.json. Independent models; warm single-design timing, not end-to-end GA speedup.',
    'Source: docs/section_grouping_optimization.md, section 5. Existing recorded cases, not rerun for this deck. Same topology comparison; mesh size 0.5 m.'
)
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Add(0)
    $deck.PageSetup.SlideWidth = 960
    $deck.PageSetup.SlideHeight = 540
    for ($i = 0; $i -lt $images.Count; $i++) {
        $slide = $deck.Slides.Add($i + 1, 12)
        $picture = $slide.Shapes.AddPicture((Join-Path $out $images[$i]), 0, -1, 0, 0, -1, -1)
        $picture.LockAspectRatio = -1
        $picture.Width = 960
        if ($picture.Height -gt 540) { $picture.Height = 540 }
        $picture.Left = (960 - $picture.Width) / 2
        $picture.Top = (540 - $picture.Height) / 2
        $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $notes[$i]
    }
    $deck.SaveAs((Join-Path $out 'research_progress.pptx'), 24)
    $deck.SaveAs((Join-Path $out 'research_progress.pdf'), 32)
    for ($i = 1; $i -le $images.Count; $i++) {
        $deck.Slides.Item($i).Export((Join-Path $check "slide_$i.png"), 'PNG', 1920, 1080)
    }
    Write-Output "Created $($deck.Slides.Count) image slides, PPTX, PDF and rendered previews."
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if ($app.Presentations.Count -eq 0) { $app.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
