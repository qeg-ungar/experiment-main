"""
Batch version of SPAD_ODMR_hermesread.ipynb.
Processes all 10 gradient-tracking iterations with a fixed QM dataset.
Outputs one .npz + _header.txt per iteration into a fresh
<spad_dir>/processed/<dataset_tag>_<run_timestamp>/ directory created per run.
"""
import sys
import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
spad_dir   = Path(r'C:\Users\SPUD1\Documents\experiment_workspace\SPAD data')
hermes_dir = spad_dir / 'hermes'  # sub-directory holding this run's .hrm files
qm_dir     = Path(r'C:\Users\SPUD1\Documents\experiment_workspace\qua-libs'
                  r'\Quantum-Control-Applications'
                  r'\Optically addressable spin qubits\NV2_array_SPAD')

# ── fixed experiment settings ─────────────────────────────────────────────────
qm_file        = r'Data\2026-07-13\#456_pulsed_odmr_gradient_tracking_202335'
spad_background = 'spc3_snap_20260714-190431-366426'   # None to skip
USE_LO_FREQUENCY = False

# ── HRM headers to process ────────────────────────────────────────────────────
AUTO_DISCOVER_HEADERS = False  # set True to auto-populate HRM_HEADERS by scanning hermes_dir

def discover_hrm_headers(hermes_dir, glob_pattern='ODMR_gradient_tracking_*.hrm'):
    """Scan hermes_dir for .hrm files and return one header string per
    acquisition, in chronological order.

    Multi-part acquisitions share a common prefix with a trailing part number
    (e.g. '..._iter001.hrm', '..._iter0012.hrm', '..._iter0013.hrm', ...) —
    the same convention the main loop uses (`re.sub(r'\\d+$', '', header)`) to
    find all part-files for a given header. This groups files the same way
    and keeps the lowest-numbered part of each group as its representative
    header, matching how entries in HRM_HEADERS are named manually.
    """
    hermes_dir = Path(hermes_dir)
    groups = {}
    for fp in hermes_dir.glob(glob_pattern):
        m = re.match(r'^(.*?)(\d+)$', fp.stem)
        if not m:
            continue
        prefix, num = m.group(1), int(m.group(2))
        if prefix not in groups or num < groups[prefix][0]:
            groups[prefix] = (num, fp.stem)
    return [stem for _, stem in sorted(groups.values(), key=lambda v: v[1])]

def _confirm_hrm_headers(headers):
    print(f'Discovered {len(headers)} header(s) in {hermes_dir}:')
    for h in headers:
        print(f'  {h}')
    reply = input('Proceed with this list? [y/N] ').strip().lower()
    if reply != 'y':
        sys.exit('Aborted: header list not confirmed.')

if AUTO_DISCOVER_HEADERS:
    HRM_HEADERS = discover_hrm_headers(hermes_dir)
    _confirm_hrm_headers(HRM_HEADERS)
else:
    HRM_HEADERS = [
        'ODMR_gradient_tracking_20260714-191740-766361_iter001',
        'ODMR_gradient_tracking_20260714-195116-747156_iter002',
        'ODMR_gradient_tracking_20260714-202442-893661_iter003',
        'ODMR_gradient_tracking_20260714-205809-289204_iter004',
        'ODMR_gradient_tracking_20260714-213135-406745_iter005',
        'ODMR_gradient_tracking_20260714-220501-397736_iter006',
        'ODMR_gradient_tracking_20260714-223827-532943_iter007',
        'ODMR_gradient_tracking_20260714-231153-852613_iter008',
        'ODMR_gradient_tracking_20260714-234520-085945_iter009',
        'ODMR_gradient_tracking_20260715-001846-467627_iter010'
    ]

# ── sys.path setup ────────────────────────────────────────────────────────────
WORKSPACE = Path(r'C:\Users\SPUD1\Documents\experiment_workspace')
sys.path.insert(0, str(WORKSPACE / 'experiment-main'))

HERMES_DIR = (WORKSPACE / 'qudi-iqo-modules' / 'src' / 'qudi' / 'hardware'
              / 'camera' / 'Hermes')
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))

from Hermes import Hermes
from qua_tools_nv2.dataset import DatasetReader

# ── load QM dataset once (fixed for all iterations) ──────────────────────────
reader  = DatasetReader(str(qm_dir))
qm_path = Path(qm_dir) / qm_file
ds      = reader.resolve_dataset(str(reader.resolve_dataset(str(qm_path)).folder))
data    = reader.load(ds)

if_hz = np.asarray(data['IF_frequencies'], dtype=float)
cfg   = data.get('config') or {}
if USE_LO_FREQUENCY:
    lo_hz  = float(cfg['elements']['NV']['mixInputs']['lo_frequency'])
    f_vec  = lo_hz + if_hz
else:
    f_vec  = if_hz

n_avg       = int(data.get('n_avg', 1))
n_iteration = int(data.get('iteration', n_avg - 1)) + 1
F           = len(f_vec)
frames_per_rep = 2 * F

print(f'QM dataset    : {ds.folder}')
print(f'n_avg         : {n_avg}   n_iteration: {n_iteration}')
print(f'len(f_vec)    : {F}   range: {f_vec.min()/1e6:.1f}–{f_vec.max()/1e6:.1f} MHz')
print()

# ── load background once ──────────────────────────────────────────────────────
bg_hrm_path = spad_dir / 'hermes'  # background snaps live in the default hermes folder, independent of hermes_dir
DATA_OFFSET = 1032  # 8-byte signature + 1024-byte header

if spad_background is not None:
    bg_path = bg_hrm_path / f'{spad_background}.hrm'
    _, header_bg = Hermes.ReadHermesDataFile(str(bg_path))

    if header_bg.CoarseGate_C1_ON:
        integration_time_s_bg = header_bg.CoarseGate_C1_stopPos - header_bg.CoarseGate_C1_startPos
    else:
        integration_time_s_bg = header_bg.SummedFrames * header_bg.HwIntTime

    dtype_bg   = np.uint8 if header_bg.bit_x_pix == 8 else np.uint16
    n_rows_bg  = header_bg.N_rows
    n_cols_bg  = header_bg.N_cols
    raw_bg     = np.memmap(str(bg_path), dtype=dtype_bg, mode='r', offset=DATA_OFFSET)
    n_frames_bg = raw_bg.size // (n_rows_bg * n_cols_bg)
    frames_bg   = raw_bg[:n_frames_bg * n_rows_bg * n_cols_bg].reshape(
        n_frames_bg, n_rows_bg, n_cols_bg
    )
    bg_avg_cps  = frames_bg.mean(axis=0).astype(np.float64) / integration_time_s_bg
    del raw_bg, frames_bg
    print(f'Background loaded: {bg_path.name}  ({bg_avg_cps.mean():.1f} cps avg)')
else:
    bg_avg_cps  = None
    header_bg   = None
    print('No background — skipping subtraction.')
print()

# ── helpers ───────────────────────────────────────────────────────────────────
def _to_builtin(v):
    if isinstance(v, np.generic):   return v.item()
    if isinstance(v, np.ndarray):   return v.tolist()
    if isinstance(v, Path):         return str(v)
    if isinstance(v, dict):         return {str(k): _to_builtin(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)): return [_to_builtin(x) for x in v]
    try:
        json.dumps(v); return v
    except (TypeError, OverflowError): return str(v)

def _obj_to_dict(obj):
    try:
        raw = vars(obj)
        if raw: return {k: _to_builtin(v) for k, v in raw.items()}
    except TypeError:
        pass
    out = {}
    for name in dir(obj):
        if name.startswith('_'): continue
        try:
            val = getattr(obj, name)
        except Exception: continue
        if callable(val): continue
        out[name] = _to_builtin(val)
    return out

# ── main loop ─────────────────────────────────────────────────────────────────
dataset_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(qm_file).name)

run_id  = datetime.now().strftime('%Y%m%d-%H%M%S')
out_dir = spad_dir / 'processed' / f'{dataset_tag}_{run_id}'
out_dir.mkdir(parents=True, exist_ok=True)
print(f'Output directory : {out_dir}\n')

for hrm_file_header in HRM_HEADERS:
    print(f'---  {hrm_file_header}  ---')

    base_name  = f'{hrm_file_header}_{dataset_tag}_avg'
    array_path = out_dir / f'{base_name}.npz'

    # discover part-files
    _prefix  = re.sub(r'\d+$', '', hrm_file_header)
    _part_re = re.compile(r'^' + re.escape(_prefix) + r'(\d+)\.hrm$', re.IGNORECASE)
    hrm_files = sorted(
        (p for p in hermes_dir.glob(f'{_prefix}*.hrm') if _part_re.match(p.name)),
        key=lambda p: int(_part_re.match(p.name).group(1))
    )
    if not hrm_files:
        print(f'  ERROR: no .hrm files found, skipping.\n')
        continue
    print(f'  {len(hrm_files)} part-file(s) found')

    # read header from first file
    _, header = Hermes.ReadHermesDataFile(str(hrm_files[0]))
    n_rows = header.N_rows
    n_cols = header.N_cols
    dtype  = np.uint8 if header.bit_x_pix == 8 else np.uint16

    if header.CoarseGate_C1_ON:
        integration_time_s = header.CoarseGate_C1_stopPos - header.CoarseGate_C1_startPos
    else:
        integration_time_s = header.SummedFrames * header.HwIntTime

    # accumulate
    sig_sum = np.zeros((F, n_rows, n_cols), dtype=np.float64)
    ref_sum = np.zeros((F, n_rows, n_cols), dtype=np.float64)
    n_complete_reps = 0
    carry = None

    for fp in hrm_files:
        raw = np.memmap(str(fp), dtype=dtype, mode='r', offset=DATA_OFFSET)
        n_frames_file = raw.size // (n_rows * n_cols)
        frames = raw[:n_frames_file * n_rows * n_cols].reshape(n_frames_file, n_rows, n_cols)

        if carry is not None and carry.shape[0] > 0:
            gap = frames_per_rep - (carry.shape[0] % frames_per_rep)
            if gap < frames_per_rep and gap <= n_frames_file:
                combined = np.concatenate([carry, np.array(frames[:gap])], axis=0)
                n_reps_c = combined.shape[0] // frames_per_rep
                if n_reps_c > 0:
                    rv = combined[:n_reps_c * frames_per_rep].reshape(n_reps_c, F, 2, n_rows, n_cols)
                    sig_sum += rv[:, :, 0].sum(axis=0, dtype=np.float64)
                    ref_sum += rv[:, :, 1].sum(axis=0, dtype=np.float64)
                    n_complete_reps += n_reps_c
                start_idx = gap
            else:
                carry = np.concatenate([carry, np.array(frames)], axis=0)
                del raw, frames
                continue
            carry = None
        else:
            start_idx = 0

        n_remaining   = n_frames_file - start_idx
        n_complete_here = (n_remaining // frames_per_rep) * frames_per_rep
        if n_complete_here > 0:
            batch = frames[start_idx : start_idx + n_complete_here]
            rv = batch.reshape(-1, F, 2, n_rows, n_cols)
            sig_sum += rv[:, :, 0].sum(axis=0, dtype=np.float64)
            ref_sum += rv[:, :, 1].sum(axis=0, dtype=np.float64)
            n_complete_reps += n_complete_here // frames_per_rep

        leftover_start = start_idx + n_complete_here
        carry = np.array(frames[leftover_start:]) if leftover_start < n_frames_file else None
        del raw, frames

    dropped = carry.shape[0] if carry is not None else 0
    print(f'  complete reps: {n_complete_reps}  ({n_complete_reps/n_iteration*100:.1f}%)  '
          f'dropped frames: {dropped}')

    sig_avg_cps = sig_sum / n_complete_reps / integration_time_s
    ref_avg_cps = ref_sum / n_complete_reps / integration_time_s

    if bg_avg_cps is not None:
        sig_avg_cps_bgsub = sig_avg_cps - bg_avg_cps
        ref_avg_cps_bgsub = ref_avg_cps - bg_avg_cps
    else:
        sig_avg_cps_bgsub = sig_avg_cps
        ref_avg_cps_bgsub = ref_avg_cps

    # export
    meta_path = out_dir / f'{base_name}_header.txt'

    np.savez_compressed(
        array_path,
        sig_avg_cps_bgsub=sig_avg_cps_bgsub,
        ref_avg_cps_bgsub=ref_avg_cps_bgsub,
        f_vec=f_vec,
        n_complete_reps=n_complete_reps,
        bg_avg_cps=bg_avg_cps if spad_background is not None else None,
        bg_filename=str(bg_hrm_path / f'{spad_background}.hrm') if spad_background is not None else None,
    )

    metadata_export = {
        'export': {
            'array_file': str(array_path),
            'array_shapes': {
                'sig_avg_cps_bgsub': list(sig_avg_cps_bgsub.shape),
                'ref_avg_cps_bgsub': list(ref_avg_cps_bgsub.shape),
                'f_vec': list(np.shape(f_vec)),
                'bg_avg_cps': list(np.shape(bg_avg_cps)) if spad_background is not None else None,
            },
        },
        'run_info': {
            'qm_file': str(ds.folder),
            'n_complete_reps': int(n_complete_reps),
        },
        'hrm_data': {
            'hermes measurement metadata': _obj_to_dict(header),
            'integration_time_s': integration_time_s,
            'hermes background metadata': _obj_to_dict(header_bg) if spad_background is not None else None,
        },
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write('Hermes ODMR processed export metadata\n')
        f.write('=' * 36 + '\n\n')
        f.write(json.dumps(metadata_export, indent=2))

    print(f'  saved -> {array_path.name}')
    print()

print('Done.')
