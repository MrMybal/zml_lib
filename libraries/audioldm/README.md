# AudioLDM

Le port AudioLDM-S Full v2 regroupe le [conditionnement CLAP](../laion_clap/README.md), l’UNet FP32 à 690 tenseurs, DDIM déterministe, le VAE et le [vocodeur HiFi-GAN](../hifigan/README.md). Le générateur prend les identifiants de tokens du prompt et un bruit initial FP32 `[1,8,T,16]`. Le profil requiert un nombre de trames latentes multiple de huit. Il produit un signal mono FP32 à 16 kHz.

    $py = (Join-Path $env:COMFYUI_ROOT '.venv\Scripts\python.exe')
    $env:ZML_CPU_PJRT_DLL = (Resolve-Path '..\zml_win\bazel-win-tools\pjrt-cpu\pjrt_cpu.dll').Path
    $cpu = @('--@zml//platforms:cuda=false', '--@zml//platforms:rocm=false', '--@zml//platforms:vulkan=false')
    $targets = @('--target', '//libraries/audioldm:audio_decode_probe',
                 '--target', '//libraries/audioldm:audio_generate_probe',
                 '--target', '//libraries/laion_clap:clap_text_probe')
    python .\scripts\build.py @targets -- --jobs=8 --host_compilation_mode=fastbuild @cpu --then $py -B .\scripts\validate_generation_parts.py --part audioldm --part clap_text --part audioldm_generate --cytools $env:CYTOOLS_ROOT

Le test du décodeur donne 12 trames log-mel et 1 952 échantillons sur `cpu:0` avec les poids locaux. Les erreurs absolues maximales contre PyTorch FP32 sont de `4.76837158e-6` pour le log-mel et de `2.53785402e-8` pour le signal. DDIM passe le test Zig Windows de quatre pas contre Diffusers.

La génération depuis le prompt de test produit 5 152 échantillons sur `cpu:0` après deux pas DDIM avec guidage. Les erreurs absolues maximales contre PyTorch FP32 sont de `1.23977661e-5` sur le latent débruité et de `2.15694308e-6` sur le signal.

Sous Windows x64 sur RTX 3090, le checkpoint officiel a donné les mêmes 5 152 échantillons avec un écart absolu maximal de `1,20e-5` sur le signal et de `1,98e-5` sur le latent. Le prétraitement texte, le décodeur VAE et le vocodeur ont aussi passé leurs essais CUDA séparés. Ajouter `--audio-model $model --backend cuda --expected-device 'RTX 3090'` à la commande de validation avec les cibles CUDA et les DLL IREE du fork. [Checkpoint officiel épinglé](https://huggingface.co/cvssp/audioldm-s-full-v2/tree/feeb3d14203495a4b6ac0893cbdedb2159b4819c)

Le [préparateur local](prepare.py) extrait les 114 tenseurs du décodeur VAE et les 203 tenseurs de CLAP texte depuis le dossier du modèle. L’UNet et le vocodeur réutilisent leurs SafeTensors d’origine. Les probes écrivent les buffers FP32 little endian. [Référence AudioLDM](https://huggingface.co/docs/diffusers/api/pipelines/audioldm)
