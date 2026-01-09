## hw_emu perf check (prop32 experiment)

Goal: `tmp_work_32b/` (32-bit edge prop `ap_fixed<32,16>` stored in `edge_props`) within 5% cycles of baseline `tmp_work/` (12-bit packed weight) on a bigger graph.

Test graph:
- `gen_random_graph.py 2048 4096` (script default `undirected=1`, so `graph.txt` has 8192 edges/lines)

Metric:
- Vitis hw_emu `profile_kernels.csv` **Compute Unit Running Time (us)**, converted to cycles @ 250MHz (`cycles = us * 250`).

Baseline (packed weight):
- Run: `tmp_work/.run/278934`
- Profile: `tmp_work/.run/278934/hw_emu/device0/binary_0/behav_waveform/xsim/profile_kernels.csv`
- `graphyflow_big_*` avg: `579.824 us` → `144,956.0 cycles`
- `graphyflow_little_*` avg: `579.824 us` → `144,956.0 cycles`

Prop32 (128-bit edges, 4 edges/512b word):
- Run: `tmp_work_32b/.run/501379`
- Profile: `tmp_work_32b/.run/501379/hw_emu/device0/binary_0/behav_waveform/xsim/profile_kernels.csv`
- `graphyflow_big_*` avg: `593.954 us` → `148,488.5 cycles`
- `graphyflow_little_*` avg: `593.954 us` → `148,488.5 cycles`

Delta vs baseline:
- Big kernels: `+2.437%`
- Little kernels: `+2.437%`

---

## 10× edges (2048 nodes, 40960 edges input)

Note: `gen_random_graph.py` defaults to `undirected=1`, so `graph.txt` has `2 * edges` lines:
- `python3 gen_random_graph.py 2048 40960 --have_weight` → `81920` lines

Baseline (packed weight):
- Run: `tmp_work/.run/602690`
- Profile: `tmp_work/.run/602690/hw_emu/device0/binary_0/behav_waveform/xsim/profile_kernels.csv`
- `graphyflow_big_*` avg: `918.559 us` → `229,639.8 cycles`
- `graphyflow_little_*` avg: `918.559 us` → `229,639.8 cycles`

Prop32 (128-bit edges, 4 edges/512b word):
- Run: `tmp_work_32b/.run/545916`
- Profile: `tmp_work_32b/.run/545916/hw_emu/device0/binary_0/behav_waveform/xsim/profile_kernels.csv`
- `graphyflow_big_*` avg: `976.054 us` → `244,013.5 cycles`
- `graphyflow_little_*` avg: `976.054 us` → `244,013.5 cycles`

Delta vs baseline:
- Big kernels: `+6.259%`
- Little kernels: `+6.259%`
