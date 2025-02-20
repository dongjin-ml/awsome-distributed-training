$Env:CONDA_EXE = "/fsx/ubuntu/awsome-distributed-training/3.test_cases/21.efficient-gpu-training/miniconda3/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/fsx/ubuntu/awsome-distributed-training/3.test_cases/21.efficient-gpu-training/miniconda3"
$Env:_CONDA_EXE = "/fsx/ubuntu/awsome-distributed-training/3.test_cases/21.efficient-gpu-training/miniconda3/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs