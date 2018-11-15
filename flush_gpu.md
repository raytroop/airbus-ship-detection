- Apart from `nvidia-smi`, on Linux you can check which processes might be using the GPU using the command
    ```bash
    sudo fuser -v /dev/nvidia*
    ```
(this will list processes that have NVIDIA GPU device nodes open).

```bash
raytroop@myserver:~/challenges$ sudo fuser -v /dev/nvidia*
[sudo] password for raytroop:
Cannot stat file /proc/2674/fd/1023: Permission denied
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        root       1025 F...m Xorg
                     raytroop   1839 F...m compiz
                     raytroop   6943 F...m python
                     raytroop   7952 F...m chrome
                     raytroop   7953 F...m code
                     raytroop   7959 F...m chrome
/dev/nvidiactl:      root       1025 F...m Xorg
                     raytroop   1839 F...m compiz
                     raytroop   6943 F...m python
                     raytroop   7952 F...m chrome
                     raytroop   7953 F...m code
                     raytroop   7959 F...m chrome
/dev/nvidia-modeset: root       1025 F.... Xorg
                     raytroop   1839 F.... compiz
                     raytroop   7952 F.... chrome
                     raytroop   7953 F.... code
                     raytroop   7959 F.... chrome
/dev/nvidia-uvm:     raytroop   6943 F.... python
raytroop@myserver:~/challenges$ kill -9 6943 6943 6943
raytroop@myserver:~/challenges$ nvidia-smi
Tue Nov 13 09:45:09 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.73       Driver Version: 410.73       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0  On |                  N/A |
| 37%   48C    P0    59W / 250W |    379MiB / 11175MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1025      G   /usr/lib/xorg/Xorg                           245MiB |
|    0      1839      G   compiz                                        65MiB |
|    0      7952      G   ...quest-channel-token=4796196089802321374    47MiB |
|    0      7953      G   ...-token=1A3907DB8319268B6AAB495C27B2745D    16MiB |
+-----------------------------------------------------------------------------+
raytroop@myserver:~/challenges$ sudo fuser -v /dev/nvidia*
Cannot stat file /proc/2674/fd/1023: Permission denied
                     USER        PID ACCESS COMMAND
/dev/nvidia0:        root       1025 F...m Xorg
                     raytroop   1839 F...m compiz
                     raytroop   7952 F...m chrome
                     raytroop   7953 F...m code
                     raytroop   7959 F...m chrome
/dev/nvidiactl:      root       1025 F...m Xorg
                     raytroop   1839 F...m compiz
                     raytroop   7952 F...m chrome
                     raytroop   7953 F...m code
                     raytroop   7959 F...m chrome
/dev/nvidia-modeset: root       1025 F.... Xorg
                     raytroop   1839 F.... compiz
                     raytroop   7952 F.... chrome
                     raytroop   7953 F.... code
                     raytroop   7959 F.... chrome
```
<br>

- `nvidia-smi` to reset the GPUs
    ```bash
    sudo nvidia-smi --gpu-reset -i 0
    ```
<br>

- credits:
[11 GB of GPU RAM used, and no process listed by nvidia-smi](https://devtalk.nvidia.com/default/topic/958159/cuda-programming-and-performance/11-gb-of-gpu-ram-used-and-no-process-listed-by-nvidia-smi/)