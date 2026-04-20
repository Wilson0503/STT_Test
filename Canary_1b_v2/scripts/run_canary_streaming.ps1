param(
    [Parameter(Mandatory = $true)]
    [string]$NeMoRoot,

    [Parameter(Mandatory = $true)]
    [string]$Manifest,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$PretrainedName = "nvidia/canary-1b-v2",
    [double]$LeftContextSecs = 10.0,
    [double]$ChunkSecs = 1.0,
    [double]$RightContextSecs = 0.5,
    [int]$BatchSize = 8,
    [ValidateSet("waitk", "alignatt")]
    [string]$StreamingPolicy = "alignatt",
    [string]$SourceLang = "en",
    [string]$TargetLang = "en",
    [ValidateSet("asr", "s2t_translation")]
    [string]$Task = "asr",
    [ValidateSet("yes", "no")]
    [string]$Pnc = "yes"
)

$scriptPath = Join-Path $NeMoRoot "examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py"
if (-not (Test-Path $scriptPath)) {
    throw "Cannot find NeMo streaming script: $scriptPath"
}

if (-not (Test-Path $Manifest)) {
    throw "Manifest file does not exist: $Manifest"
}

$cmd = @(
    "python",
    $scriptPath,
    "pretrained_name=$PretrainedName",
    "dataset_manifest=$Manifest",
    "output_filename=$Output",
    "left_context_secs=$LeftContextSecs",
    "chunk_secs=$ChunkSecs",
    "right_context_secs=$RightContextSecs",
    "batch_size=$BatchSize",
    "decoding.streaming_policy=$StreamingPolicy",
    "decoding.alignatt_thr=8",
    "decoding.waitk_lagging=2",
    "decoding.exclude_sink_frames=8",
    "decoding.xatt_scores_layer=-2",
    "decoding.hallucinations_detector=True",
    "+prompt.pnc=$Pnc",
    "+prompt.task=$Task",
    "+prompt.source_lang=$SourceLang",
    "+prompt.target_lang=$TargetLang"
)

Write-Host "Running NeMo Canary streaming inference..."
Write-Host ($cmd -join " `n")

& $cmd[0] $cmd[1..($cmd.Length - 1)]
