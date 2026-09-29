# Result tables — new evaluation protocol

Held-out test questions only, 50 per group. NLL is the mean over answer tokens 
under teacher forcing. Deltas are *left minus reference*: positive means the left 
model assigns **less** probability to the correct answer. KL is KL(twin ‖ model), 
twin always on the left, over the full 100,352-token vocabulary. SD is the sample 
standard deviation across the 50 paired question-level values.


## Ancient Rome


### Table A — erasure models versus the twin (Ancient Rome)

| Method | Question group | N | Mean ΔNLL(M−T) | SD ΔNLL(M−T) | Mean KL(T‖M) | SD KL(T‖M) |
|---|---|---:|---:|---:|---:|---:|
| EMBER | Target | 50 | +5.2140 | 3.6207 | 6.1283 | 2.8684 |
| RMU | Target | 50 | -0.3030 | 0.9767 | 1.1427 | 0.6165 |
| SNMF | Target | 50 | -0.8773 | 0.9535 | 0.8347 | 0.6286 |
| EMBER | Neighboring | 50 | +2.7337 | 4.5080 | 2.8004 | 4.0917 |
| RMU | Neighboring | 50 | +0.8474 | 0.9410 | 0.6275 | 0.4065 |
| SNMF | Neighboring | 50 | -0.1256 | 0.6118 | 0.2418 | 0.1465 |
| EMBER | Unrelated | 50 | +0.5666 | 2.0365 | 0.9083 | 1.8385 |
| RMU | Unrelated | 50 | +0.1376 | 0.6404 | 0.4553 | 0.2577 |
| SNMF | Unrelated | 50 | -0.0205 | 0.5134 | 0.3448 | 0.2084 |

### Table B — erasure models versus the full model (Ancient Rome)

| Method | Question group | N | Mean ΔNLL(M−F) | SD ΔNLL(M−F) |
|---|---|---:|---:|---:|
| EMBER | Target | 50 | +6.2298 | 3.7752 |
| RMU | Target | 50 | +0.7128 | 0.9830 |
| SNMF | Target | 50 | +0.1385 | 0.2000 |
| EMBER | Neighboring | 50 | +2.9049 | 4.6861 |
| RMU | Neighboring | 50 | +1.0186 | 0.9417 |
| SNMF | Neighboring | 50 | +0.0456 | 0.1472 |
| EMBER | Unrelated | 50 | +0.5971 | 1.9520 |
| RMU | Unrelated | 50 | +0.1681 | 0.4285 |
| SNMF | Unrelated | 50 | +0.0101 | 0.0696 |

### Table C — twin versus full (Ancient Rome)

| Question group | N | Mean ΔNLL(T−F) | SD ΔNLL(T−F) | Mean KL(T‖F) | SD KL(T‖F) |
|---|---:|---:|---:|---:|---:|
| Target | 50 | +1.0158 | 1.0015 | 0.9551 | 0.7248 |
| Neighboring | 50 | +0.1712 | 0.6702 | 0.2531 | 0.1619 |
| Unrelated | 50 | +0.0306 | 0.5090 | 0.3429 | 0.2106 |

## Baseball


### Table A — erasure models versus the twin (Baseball)

| Method | Question group | N | Mean ΔNLL(M−T) | SD ΔNLL(M−T) | Mean KL(T‖M) | SD KL(T‖M) |
|---|---|---:|---:|---:|---:|---:|
| EMBER | Target | 50 | +1.1369 | 2.3385 | 1.7150 | 1.5169 |
| RMU | Target | 50 | -0.3224 | 1.0185 | 0.6792 | 0.4669 |
| SNMF | Target | 50 | -0.3028 | 1.0538 | 0.6909 | 0.4765 |
| EMBER | Neighboring | 50 | -0.0033 | 0.8479 | 0.7327 | 0.7020 |
| RMU | Neighboring | 50 | -0.1270 | 0.7779 | 0.5490 | 0.2939 |
| SNMF | Neighboring | 50 | -0.1782 | 0.8807 | 0.5537 | 0.3078 |
| EMBER | Unrelated | 50 | +0.0188 | 0.5655 | 0.3920 | 0.2680 |
| RMU | Unrelated | 50 | +0.0042 | 0.5435 | 0.3733 | 0.2694 |
| SNMF | Unrelated | 50 | +0.0457 | 0.5240 | 0.3884 | 0.2818 |

### Table B — erasure models versus the full model (Baseball)

| Method | Question group | N | Mean ΔNLL(M−F) | SD ΔNLL(M−F) |
|---|---|---:|---:|---:|
| EMBER | Target | 50 | +1.5136 | 2.1340 |
| RMU | Target | 50 | +0.0543 | 0.1334 |
| SNMF | Target | 50 | +0.0739 | 0.2965 |
| EMBER | Neighboring | 50 | +0.0893 | 0.7420 |
| RMU | Neighboring | 50 | -0.0345 | 0.1545 |
| SNMF | Neighboring | 50 | -0.0856 | 0.3165 |
| EMBER | Unrelated | 50 | +0.0039 | 0.0927 |
| RMU | Unrelated | 50 | -0.0107 | 0.0421 |
| SNMF | Unrelated | 50 | +0.0308 | 0.1407 |

### Table C — twin versus full (Baseball)

| Question group | N | Mean ΔNLL(T−F) | SD ΔNLL(T−F) | Mean KL(T‖F) | SD KL(T‖F) |
|---|---:|---:|---:|---:|---:|
| Target | 50 | +0.3767 | 1.0777 | 0.7070 | 0.4950 |
| Neighboring | 50 | +0.0925 | 0.7472 | 0.5507 | 0.3093 |
| Unrelated | 50 | -0.0149 | 0.5537 | 0.3734 | 0.2692 |

## Artificial Intelligence


### Table A — erasure models versus the twin (Artificial Intelligence)

| Method | Question group | N | Mean ΔNLL(M−T) | SD ΔNLL(M−T) | Mean KL(T‖M) | SD KL(T‖M) |
|---|---|---:|---:|---:|---:|---:|
| EMBER | Target | 50 | +1.6249 | 2.5066 | 1.9635 | 2.0930 |
| RMU | Target | 50 | -0.5729 | 1.1590 | 0.5789 | 0.3645 |
| SNMF | Target | 50 | -0.5144 | 1.1325 | 0.5882 | 0.3699 |
| EMBER | Neighboring | 50 | +0.2966 | 1.2837 | 0.4385 | 0.3400 |
| RMU | Neighboring | 50 | +0.1417 | 0.7709 | 0.3566 | 0.1785 |
| SNMF | Neighboring | 50 | +0.1089 | 0.8123 | 0.3646 | 0.1874 |
| EMBER | Unrelated | 50 | -0.1501 | 0.6954 | 0.3925 | 0.2970 |
| RMU | Unrelated | 50 | -0.1290 | 0.6871 | 0.3917 | 0.2950 |
| SNMF | Unrelated | 50 | -0.1268 | 0.6775 | 0.3934 | 0.2909 |

### Table B — erasure models versus the full model (Artificial Intelligence)

| Method | Question group | N | Mean ΔNLL(M−F) | SD ΔNLL(M−F) |
|---|---|---:|---:|---:|
| EMBER | Target | 50 | +2.2203 | 2.6618 |
| RMU | Target | 50 | +0.0225 | 0.1078 |
| SNMF | Target | 50 | +0.0810 | 0.1816 |
| EMBER | Neighboring | 50 | +0.1572 | 0.8779 |
| RMU | Neighboring | 50 | +0.0024 | 0.1422 |
| SNMF | Neighboring | 50 | -0.0305 | 0.3212 |
| EMBER | Unrelated | 50 | -0.0045 | 0.0319 |
| RMU | Unrelated | 50 | +0.0166 | 0.0625 |
| SNMF | Unrelated | 50 | +0.0188 | 0.1130 |

### Table C — twin versus full (Artificial Intelligence)

| Question group | N | Mean ΔNLL(T−F) | SD ΔNLL(T−F) | Mean KL(T‖F) | SD KL(T‖F) |
|---|---:|---:|---:|---:|---:|
| Target | 50 | +0.5954 | 1.1796 | 0.5932 | 0.3799 |
| Neighboring | 50 | -0.1394 | 0.7690 | 0.3501 | 0.1732 |
| Unrelated | 50 | +0.1456 | 0.6879 | 0.3918 | 0.2967 |
