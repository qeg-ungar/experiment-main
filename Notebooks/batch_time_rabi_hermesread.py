"""
Batch version of SPAD_time_rabi_hermesread.ipynb.
Processes all tracking iterations for a fixed Rabi QM dataset.
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
hermes_dir = spad_dir / 'hermes temp'  # sub-directory holding this run's .hrm files
qm_dir     = Path(r'C:\Users\SPUD1\Documents\experiment_workspace\qua-libs'
                  r'\Quantum-Control-Applications'
                  r'\Optically addressable spin qubits\NV2_array_SPAD')

# ── fixed experiment settings ─────────────────────────────────────────────────
qm_file        = r'Data\2026-07-19\#583_time_rabi_gradient_tracking_145838'
spad_background = 'spc3_snap_20260719-142535-427299'   # set to stem name (no .hrm) to subtract a background snap

# ── HRM headers to process ────────────────────────────────────────────────────
AUTO_DISCOVER_HEADERS = True  # set True to auto-populate HRM_HEADERS by scanning hermes_dir

def discover_hrm_headers(hermes_dir, glob_pattern='time_rabi_tracking_*.hrm'):
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
        'time_rabi_tracking_20260717-125219-367413_iter001',
        'time_rabi_tracking_20260717-132401-665945_iter002',
        'time_rabi_tracking_20260717-135543-829128_iter003',
        'time_rabi_tracking_20260717-142726-762305_iter004',
        'time_rabi_tracking_20260717-145908-915647_iter005',
        'time_rabi_tracking_20260717-153050-820680_iter006',
        'time_rabi_tracking_20260717-160232-609725_iter007',
        'time_rabi_tracking_20260717-163414-212408_iter008',
        'time_rabi_tracking_20260717-170556-594024_iter009',
        'time_rabi_tracking_20260717-173738-845329_iter010',
        'time_rabi_tracking_20260717-180921-097126_iter011',
        'time_rabi_tracking_20260717-184102-755556_iter012',
        'time_rabi_tracking_20260717-191244-541227_iter013',
        'time_rabi_tracking_20260717-194426-320697_iter014',
        'time_rabi_tracking_20260717-201608-311726_iter015',
        'time_rabi_tracking_20260717-204750-193910_iter016',
        'time_rabi_tracking_20260717-211933-126223_iter017',
        'time_rabi_tracking_20260717-215115-601480_iter018',
        'time_rabi_tracking_20260717-222257-954885_iter019',
        'time_rabi_tracking_20260717-225439-607465_iter020',
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

t_vec = np.asarray(data['t_vec'], dtype=float)
n_avg           = int(data.get('n_avg', 1))
n_iteration     = int(data.get('iteration', n_avg - 1)) + 1
F               = len(t_vec)
frames_per_rep  = 2 * F

print(f'QM dataset      : {ds.folder}')
print(f'n_avg           : {n_avg}   n_iteration: {n_iteration}')
print(f'len(pulse_dur)  : {F}   range: {t_vec.min():.1f}–{t_vec.max():.1f} ns')
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

        n_remaining    = n_frames_file - start_idx
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
        t_vec=t_vec,
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
                't_vec': list(np.shape(t_vec)),
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
        f.write('Hermes time-Rabi processed export metadata\n')
        f.write('=' * 42 + '\n\n')
        f.write(json.dumps(metadata_export, indent=2))

    print(f'  saved -> {array_path.name}')
    print()

print('Done.')
