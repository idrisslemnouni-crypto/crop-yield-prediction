param([string]$Repository = 'idrisslemnouni-crypto/crop-yield-prediction')
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $TaskRoot
if ($Repository -ne 'idrisslemnouni-crypto/crop-yield-prediction') {
    throw 'Unexpected repository destination. Review the script before changing the target.'
}
gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Connect using gh auth login, then rerun this script.' }
$TaskLogin = gh api user --jq .login
if ($LASTEXITCODE -ne 0 -or $TaskLogin -ne 'idrisslemnouni-crypto') { throw 'Authenticated GitHub account does not match the intended owner.' }
$TaskChanges = git status --porcelain
if ($TaskChanges) { throw 'Commit reviewed changes before publishing.' }
$TaskRemotes = git remote
if ($TaskRemotes) { throw 'Remote already configured. Review its destination instead of rerunning repository creation.' }
gh repo create $Repository --public --source . --remote origin --description 'Chronological US maize yield benchmark with real data, honest generalization analysis, SHAP and FastAPI'
if ($LASTEXITCODE -ne 0) { throw 'Repository creation failed; no push attempted.' }
git push -u origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed. Review authentication and retry the push.' }
gh repo edit $Repository --default-branch main --add-topic agriculture --add-topic machine-learning --add-topic xgboost --add-topic crop-yield --add-topic time-series --add-topic fastapi
if ($LASTEXITCODE -ne 0) { throw 'Code pushed, but repository metadata update failed.' }
Write-Output "Published https://github.com/$Repository"
