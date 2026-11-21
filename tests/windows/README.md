# Validation Windows x64

Le test utilise les sources de `zml_win` au commit `0bb81d1957fb6438faddd1751566ad73eed97914`. Le `HEAD` du fork est passé à `ed3be5bffb1e84cdb9daa50b36959de996dfdeff` pendant le travail. Le diff entre les deux commits se limite à `README.md` et `docs/windows-x64.md`. Le rejeu contrôle cette équivalence et refuse un changement dans les sources ou la configuration.

## Commandes PowerShell

Préparer les plugins, Git Bash et l’ICD AMD selon [le guide du fork](https://github.com/MrMybal/zml_win/blob/windows-x64/docs/windows-x64.md). Depuis le dépôt des portages, avec PowerShell 7.

    $fork = (Resolve-Path ..\zml_win).Path
    python .\scripts\build.py --engine $fork --with-fork -- --jobs=12 --then pwsh -NoProfile -File .\tests\windows\validate.ps1 -Fork $fork
    if ($LASTEXITCODE) { throw 'Validation Windows en échec' }

Le script compile les références du fork et les portages sous `_Temp`, lance les deux validations, puis supprime les fichiers de compilation. Les journaux et le CSV sont écrits dans `validation-output\windows-x64`. Passer `-ForkOnly` ou `-ConsumerOnly` à `validate.ps1` pour limiter les exécutions.

Pour vérifier les fichiers distribuables dans un dossier autonome, copier l’en-tête et les binaires et créer l’archive d’import à partir de la table d’exports PE. L’exemple utilise les outils LLVM x64 de Visual Studio 2022.

    python .\scripts\build.py --engine $fork -- --jobs=12
    if ($LASTEXITCODE) { throw 'Compilation en échec' }
    $dist = (New-Item -ItemType Directory -Force .\validation-output\windows-x64\bin).FullName
    $bin = '.\validation-output\bin'
    Copy-Item .\include\zml_lib.h, "$bin\zml_lib.dll", "$bin\windows_consumer.exe", "$bin\zml_lib_cpp.exe", "$bin\libraries\realesrgan\realesrgan_cli.exe" -Destination $dist
    Copy-Item .\libraries\realesrgan\third_party\LICENSE-*.txt -Destination $dist
    Copy-Item "$fork\LICENSE" -Destination "$dist\LICENSE-ZML.txt"
    $llvm = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin'
    & "$llvm\llvm-dlltool.exe" -m i386:x86-64 -D zml_lib.dll -d .\src\zml_lib.def -l "$dist\zml_lib_import.lib"
    if ($LASTEXITCODE) { throw 'Création de l’archive d’import en échec' }

Pour tester ces binaires, passer `-BinDir $dist -ConsumerOnly -Logs .\validation-output\windows-x64\distribution` à `validate.ps1` dans la commande de validation ci-dessus.

Le script lance un nouveau processus par backend. Il sélectionne `HIP_VISIBLE_DEVICES=0`, `IREE_PJRT_ROCM_TARGET=gfx1036` et le manifeste ICD AMD sous `u0420529.inf_amd64_*`. Le test CUDA sélectionne `sm_86` pour la RTX 3090. Les chemins exacts sont disponibles dans le CSV. Le consommateur vérifie le nom PJRT, le périphérique et la résidence de chaque sortie. Il exécute dix fois chaque modèle chargé, décharge, vérifie l’erreur sans modèle, recharge et teste à nouveau. Une DLL PJRT manquante est testée dans un autre processus.

## Résultats

| Backend forcé | Plateforme et périphérique | Direct dans ZML | Via le consommateur C++ | DLL PJRT et dépendances spécifiques |
| --- | --- | --- | --- | --- |
| CPU | `cpu`, `cpu:0`, Ryzen 9 7900X | 42, dix fois [7, 0], MNIST 7/2/1 | Réussi, rechargement inclus | `pjrt_cpu.dll` |
| CUDA | `iree_cuda`, NVIDIA GeForce RTX 3090 | 42, dix fois [7, 0], MNIST 7/2/1 | Réussi, rechargement inclus | `iree_pjrt_cuda_dylib.dll`, `IREECompiler.dll`, `nvcuda.dll` |
| ROCm/HIP | `iree_rocm`, AMD Radeon Graphics gfx1036 | 42, dix fois [7, 0], MNIST 7/2/1 | Réussi, rechargement inclus | `iree_pjrt_rocm_dylib.dll`, `IREECompiler.dll`, `amdhip64.dll` |
| Vulkan | `iree_vulkan`, AMD Radeon Graphics, pilote AMD 25.10.30.02 | 42, dix fois [7, 0], MNIST 7/2/1 | Réussi, rechargement inclus | `iree_pjrt_vulkan_dylib.dll`, `IREECompiler.dll`, `vulkan-1.dll`, ICD AMD |

Les quatre tests de DLL absente ont reçu `ZML_LIB_BACKEND_ERROR`, `PluginLoadFailed` et un handle nul. L’inspection PE a confirmé `IMAGE_FILE_MACHINE_AMD64`, les 12 exports publics et l’import de `zml_lib.dll` par le consommateur. Les plugins IREE utilisent l’UCRT et les runtimes Visual C++ x64. Les journaux PE recensent les imports statiques. Le chargeur de ZML résout les plugins et leurs dépendances à l’exécution.

| Journaux générés dans `validation-output\windows-x64` | Contenu |
| --- | --- |
| `fork-build.log` | Compilation du fork |
| `consumer-build-final.log` | Compilation Bazel du dépôt des portages |
| `distribution\fork-native-smoke-<backend>.log` et `distribution\fork-mnist-<backend>-<indice>.log` | Exécutions directes du fork |
| `distribution\consumer-<backend>.log` et `distribution\consumer-missing-<backend>.log` | Exécutions ABI C et DLL absente |
| `distribution\consumer-pe-*.log`, `distribution\pjrt-pe-*.log` et `distribution\consumer-sha256.log` | En-têtes PE, imports, exports et empreintes |
| `distribution\matrix.csv` et `distribution\repositories.log` | Statuts, chemins des plugins et commits |

## Real‑ESRGAN

Avec des poids synthétiques déterministes, les deux variantes ont produit la sortie FP32 attendue et le PNG attendu via le CLI et l’ABI C sur CPU, ROCm/HIP et Vulkan. Le test sur CUDA échoue dans IREE lors de `cuModuleLoadDataEx` avec `CUDA_ERROR_INVALID_CONTEXT`. L’erreur se reproduit sur anime_6B via l’ABI C sans export d’activations. CUDA a réussi les tests d’addition, du petit modèle et de MNIST ci-dessus.

    python .\scripts\build.py -- --jobs=12 --then python .\tests\windows\rrdb_fixture.py --bin '{bin}' --backend rocm --out .\validation-output\windows-x64\rrdb-replay-rocm

Configurer les variables du backend comme indiqué dans le guide du fork et utiliser un nouveau dossier à chaque rejeu. Les journaux de cette exécution sont `rrdb-bazel-cpu.log`, `rrdb-bazel-rocm.log`, `rrdb-bazel-vulkan.log`, `rrdb-fixture-cuda.log` et `rrdb-bazel-cuda-anime.log`. La comparaison avec les checkpoints officiels reste à exécuter avec les commandes du [port Real‑ESRGAN](../../libraries/realesrgan/README.md).
