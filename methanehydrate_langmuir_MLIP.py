"""
Langmuir Constant Calculation — EMPTY sI Clathrate Hydrate | MACE-ANI-CC
=========================================================================

Physical picture
----------------
The NPT trajectory (extxyz, from mace_prod_npt.py) contains only water
(O and H).  To represent the correct infinite-lattice host when computing
C_i for one cavity type, every cavity of the OTHER type must be occupied
by methane (the blocker).  Blocker CH4 molecules are inserted at fixed
mapping-file cavity centres with independent random orientations assigned
ONCE per frame.

MACE energy decomposition
--------------------------
Since MACE is a many-body potential, pair energies are not accessible.
The probe-water interaction is isolated via four system energies:

  Computed ONCE per frame:
    U0 = E(water + blocker CH4)          [frame constant]
    U3 = E(blocker CH4 only)             [frame constant]

  Computed per accepted insertion:
    U1 = E(water + blocker CH4 + probe)
    U2 = E(blocker CH4 + probe)

    DeltaU = (U1 - U0) - (U2 - U3)      [probe-water interaction only]

Why blockers must be full CH4 (C + 4H)
---------------------------------------
(U2 - U3) is designed to equal the probe-blocker guest-guest term so it
cancels from DeltaU, leaving only probe-water.  For this cancellation to
be correct, U2 and U3 must describe the SAME physical blocker system.
With bare C atoms the cancellation is spurious; with full CH4 (C+4H,
random orientations) it is exact -- matching the fully-occupied MACE
code which uses real C+4H molecules from the trajectory as blockers.

Spatial rejection  (minimum-image PBC, before any MACE call)
  (a) C(probe)-O(water)   < cutoff_CO  --> BF = 0  (hard-core guard)
  (b) C(probe)-C(blocker) < cutoff_CC  --> BF = 0  (cavity-region guard)
  Rejected insertions contribute 0 to the numerator but are counted in
  the denominator  (consistent Widom formula convention).

Langmuir constant:
  C = [V_box / (N_target x k_B x T)] x <exp(-beta DeltaU)>

To switch between small and large cavities change TARGET_CAVITY below.
  TARGET_CAVITY = 'small'  -->  probe samples 512 cages
                                blockers fill 48 large (51262) cavities
  TARGET_CAVITY = 'large'  -->  probe samples 51262 cages
                                blockers fill 16 small (512) cavities

Output files (same names / format as LJ code):
  langmuir_results_MC_empty_{cavity}.txt
  langmuir_results_MC_empty_{cavity}_data.csv
  langmuir_analysis_{cavity}_MC_empty.png
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import numpy as np
from datetime import datetime
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# ==================== DUAL LOGGER ====================

class DualLogger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


log_filename = "langmuir_mace_off24_empty_large_3.4484_widom.log"
sys.stdout = DualLogger(log_filename)
sys.stderr = sys.stdout

print("=== Langmuir Constant — EMPTY Hydrate (MACE-ANI-CC + ASE extxyz Traj) ===")
print(f"Start time: {datetime.now()}")
print(f"Log file  : {log_filename}\n")


# ==================== MAIN CALCULATOR CLASS ====================

class MACELangmuirCalculatorEmpty:
    """
    Langmuir constant calculator for an EMPTY sI clathrate hydrate.
    Energy engine: MACE-ANI-CC.
    Trajectory  : extxyz NPT production run (water only, O + H).
    Mapping file: Takeuchi 2x2x2 cavity centres (64 entries).
    """

    # 2x2x2 sI supercell cavity counts
    N_SMALL = 16   # small (512)   cages
    N_LARGE = 48   # large (51262) cages

    def __init__(self, model_path, temperature=273.0, seed=42, device='cuda'):
        if seed is not None:
            np.random.seed(seed)
            print(f"Random seed set to: {seed}")

        self.temperature = temperature
        self.kB_SI  = 1.380649e-23   # J/K
        self.kB_eV  = 8.617333e-5    # eV/K

        self.small_centres = None    # (16, 3) Angstrom — loaded once
        self.large_centres = None    # (48, 3) Angstrom — loaded once

        # Spatial rejection cutoffs (Angstrom)
        self.cutoff_CO = 3.4484    # C(probe)-O(water):   hard-core guard
        self.cutoff_CC = 3.73   # C(probe)-C(blocker): cavity-region guard
                                # (= CH4 sigma, same as OpenMM rejection_dist)

        self.ch_bond = 1.09     # C-H bond length for building CH4 (Angstrom)

        print("="*80)
        print("FORCE FIELD / CALCULATOR PARAMETERS")
        print("="*80)
        print(f"Temperature    : {self.temperature} K")
        print(f"Rejection C-O  : {self.cutoff_CO} A  (probe-water hard core)")
        print(f"Rejection C-C  : {self.cutoff_CC} A  (same as LJ code, CH4 sigma)")
        print(f"\nLoading MACE model: {model_path}")
        self.calc = MACECalculator(
            model_paths   = model_path,
            device        = device,
            default_dtype = "float64"
        )
        print(f"MACE calculator ready on {device}")
        print("="*80 + "\n")

    # =========================================================================
    # UTILITY
    # =========================================================================

    @staticmethod
    def apply_pbc(delta, box):
        """Minimum-image convention.  delta: (...,3), box: (3,) array"""
        return delta - box * np.round(delta / box)

    def _make_ch4(self, center_pos, rotation=None):
        # Proper tetrahedral vertices on unit sphere, then scale by ch_bond
        s2 = np.sqrt(2.0)
        s6 = np.sqrt(6.0)
        h_ref = np.array([
            [ 0.0,          0.0,    1.0      ],   # top
            [ 2*s2/3,       0.0,   -1.0/3.0  ],   # front
            [  -s2/3,  s6/3,       -1.0/3.0  ],   # back-left
            [  -s2/3, -s6/3,       -1.0/3.0  ],   # back-right
        ]) * self.ch_bond
        # Verify: dot product between any two rows = -1/3  →  angle = 109.47 deg
        if rotation is None:
            rotation = Rotation.random()
        h_pos = rotation.apply(h_ref) + center_pos
        return Atoms(
            symbols   = ['C', 'H', 'H', 'H', 'H'],
            positions = np.vstack([center_pos, h_pos])
        )

    def _compute_probe_reference_energy(self):
        """
        Energy of a single CH4 molecule in vacuum.
        Computed once at startup. Used to correct the MACE absolute
        energy reference: delta_U = U_full - U_water - U_probe_vac
        """
        # Place probe at origin in a large non-periodic cell
        # Non-periodic so no image interactions
        probe = self._make_ch4(np.array([0.0, 0.0, 0.0]))
        probe.set_cell([50.0, 50.0, 50.0])
        probe.set_pbc(False)
        probe.calc = self.calc
        U_probe_vac = probe.get_potential_energy()
        print(f"U_probe_vacuum (isolated CH4) : {U_probe_vac:.6f} eV")
        return U_probe_vac

    # =========================================================================
    # PHASE 1: LOAD CAVITY CENTRES  (called once)
    # =========================================================================

    def load_cavity_centres(self, mapping_file):
        """
        Read Takeuchi 2x2x2 mapping file.
        Format:  ch4_id  cavity_type  x  y  z   (64 lines, Angstrom)
        Identical file format to the LJ/OpenMM code.
        """
        print("="*80)
        print("PHASE 1: LOADING CAVITY CENTRES FROM MAPPING FILE  "
              "(fixed for all frames)")
        print("="*80)
        print(f"  Mapping file : {mapping_file}")

        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        small_list, large_list = [], []
        with open(mapping_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                ctype = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                if ctype == 'small':
                    small_list.append([x, y, z])
                elif ctype == 'large':
                    large_list.append([x, y, z])
                else:
                    print(f"  WARNING: unknown cavity type '{ctype}' -- skipped")

        self.small_centres = np.array(small_list)
        self.large_centres = np.array(large_list)

        print(f"\n  Cavity centres loaded:")
        print(f"    Small cages (512)   : {len(self.small_centres)}")
        print(f"    Large cages (51262) : {len(self.large_centres)}")
        print(f"    Total               : "
              f"{len(self.small_centres) + len(self.large_centres)}")

        if len(self.small_centres) != self.N_SMALL:
            print(f"  WARNING: expected {self.N_SMALL} small cages, "
                  f"got {len(self.small_centres)}")
        if len(self.large_centres) != self.N_LARGE:
            print(f"  WARNING: expected {self.N_LARGE} large cages, "
                  f"got {len(self.large_centres)}")
        print("="*80 + "\n")

    # =========================================================================
    # PHASE 2: READ EXTXYZ TRAJECTORY
    # =========================================================================

    def read_extxyz_trajectory(self, filename, max_frames=None):
        """
        Read ASE extxyz trajectory produced by mace_prod_npt.py.
        Returns list of frame dicts (frame_id, box, positions, symbols).
        """
        print(f"Reading trajectory: {filename}")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        all_atoms = read(filename, index=':', format='extxyz')
        if max_frames is not None:
            all_atoms = all_atoms[:max_frames]

        frames = []
        for idx, atoms in enumerate(all_atoms):
            frames.append({
                'frame_id' : idx,
                'atoms'    : atoms,
                'box'      : atoms.get_cell().lengths(),      # (3,) Angstrom
                'positions': atoms.get_positions().copy(),
                'symbols'  : np.array(atoms.get_chemical_symbols())
            })
            if (idx + 1) % 200 == 0:
                print(f"  Loaded {idx+1} frames...")

        print(f"Total frames loaded: {len(frames)}\n")
        return frames

    # =========================================================================
    # WIDOM INSERTION — ONE FRAME
    # =========================================================================

    def run_widom_insertion_frame(self, frame, target_cavity_type='small',
                                  N_insert=100_000, screen_batch=10_000):
        """
        Widom test-particle insertion for one extxyz frame.
    
        Two-phase design:
          Phase 1 — vectorized NumPy screening over all N_insert positions,
                    processed in batches of screen_batch to control RAM.
                    No Python loop over rejected insertions.
          Phase 2 — MACE loop runs only over the ~1-2% accepted positions.
                    torch.cuda.empty_cache() called once after loop completes.
    
        screen_batch=10_000: peak array (10000, 368, 3) x 4 bytes ~ 44 MB RAM.
        Increase if you have RAM to spare; decrease if RAM is tight.
        """
        import torch
    
        print(f"\n{'='*80}")
        print(f"WIDOM INSERTION  (Frame {frame['frame_id']})")
        print(f"{'='*80}")
    
        box       = frame['box']          # (3,) Angstrom
        #box_L     = box[0]                # scalar for minimum-image PBC
        water_pos = frame['positions']
        water_sym = frame['symbols']
    
        # ── Blocker / target cavity setup ────────────────────────────────────
        if target_cavity_type == 'small':
            other_centres = self.large_centres
            other_type      = 'large' 
            N_target        = self.N_SMALL  # (48, 3) — reject probe landing here
        else:
            other_centres = self.small_centres
            other_type      = 'small'
            N_target        = self.N_LARGE   # (16, 3) — reject probe landing here
    
        print(f"Target cavity  : {target_cavity_type}  (N={N_target})")
        print(f"Blocker cavity : {other_type}  (N={len(other_centres)})")
        print(f"Box (A)        : {box}")
        #print(f"N_insert       : {N_insert:,}  |  screen_batch : {screen_batch:,}")
    
        # U_water: water only, computed ONCE per frame and cached
        water_ase = Atoms(
            symbols   = list(water_sym),
            positions = water_pos.copy(),
            cell      = [box[0], box[1], box[2]],
            pbc       = True
        )
        water_ase.calc = self.calc
        U_water = water_ase.get_potential_energy()
        print(f"U_water (water only) : {U_water:.6f} eV")

        # ── Pre-extract O and blocker-C positions for screening
        water_o_pos = water_pos[water_sym == 'O']   # (N_O, 3)
        print(f"\nScreening arrays:")
        print(f"  Water oxygens  : {len(water_o_pos)}")
        print(f"  Blocker carbons: {len(other_centres)}")
    
        # =========================================================================
        # PHASE 1: vectorized screening — find all accepted positions at once
        # =========================================================================
    
        t_screen           = time.time()
        accepted_positions = []
        n_rejected_water   = 0
        n_rejected_blocker = 0
        n_generated        = 0
    
        while n_generated < N_insert:
            batch        = min(screen_batch, N_insert - n_generated)
            n_generated += batch
    
            # Generate batch of random positions — one vectorized call
            rpos_batch = np.column_stack([
                np.random.uniform(0, box[0], batch),
                np.random.uniform(0, box[1], batch),
                np.random.uniform(0, box[2], batch)
            ])                                                  # (batch, 3)
    
            # ── C-O screening ──────────────────────────────────────────────
            # dO shape: (batch, N_O, 3)
            dO  = rpos_batch[:, None, :] - water_o_pos[None, :, :]
            dO  = self.apply_pbc(dO, box)
            # minimum distance from each probe to any oxygen: (batch,)
            min_dO = np.min(np.linalg.norm(dO, axis=2), axis=1)
            mask_O = min_dO >= self.cutoff_CO                  # True = passed
            n_rejected_water += int(np.sum(~mask_O))
            rpos_O = rpos_batch[mask_O]                        # (M, 3)
    
            if len(rpos_O) == 0:
                continue
    
            # ── C-C screening on O-passed positions only ───────────────────
            # dC shape: (M, N_C, 3)
            dC  = rpos_O[:, None, :] - other_centres[None, :, :]
            dC  = self.apply_pbc(dC, box)
            # minimum distance from each probe to any blocker carbon: (M,)
            min_dC = np.min(np.linalg.norm(dC, axis=2), axis=1)
            mask_C = min_dC >= self.cutoff_CC                  # True = passed
            n_rejected_blocker += int(np.sum(~mask_C))
    
            accepted_positions.append(rpos_O[mask_C])          # (K, 3)
    
        # Combine accepted positions from all batches
        if accepted_positions:
            accepted_positions = np.vstack(accepted_positions) # (K_total, 3)
        else:
            accepted_positions = np.empty((0, 3))
    
        n_accepted_screening = len(accepted_positions)
        screen_time          = time.time() - t_screen
    
        # =========================================================================
        # PHASE 2: MACE loop — runs only over accepted positions
        # =========================================================================
    
        beta   = 1.0 / (self.kB_eV * self.temperature)
        sum_BF = 0.0
        t_mace = time.time()
    
        # Progress interval: 5 updates over the full accepted set
        #print_interval = max(1, n_accepted_screening // 5)
    
        for rpos in accepted_positions:
            probe_ch4 = self._make_ch4(rpos)
        
            sys_full = water_ase.copy()
            sys_full.extend(probe_ch4)
            sys_full.set_pbc(True)
            sys_full.calc = self.calc
            U_full = sys_full.get_potential_energy()
        
            delta_U = U_full - U_water - self.U_probe_vac
            sum_BF += np.exp(-beta * delta_U)
            # Inside the MACE loop, after computing delta_U:
            #if delta_U < -0.10:   # deeper than any LJ minimum
             #   print(f"  WARNING: very large BF, delta_U={delta_U:.4f} eV, pos={rpos}")
    
        # Single cache clear after the entire MACE loop — not inside it
        torch.cuda.empty_cache()
    
        # avg_BF denominator = N_insert (total attempted) — correct Widom convention
        avg_BF   = sum_BF / N_insert
        mace_time = time.time() - t_mace
    
        print(f"=== Frame Summary ===")
        print(f"  Total attempted               : {N_insert:,}")
        print(f"  Passed screening (MACE calls) : {n_accepted_screening} "
              f"({100*n_accepted_screening/N_insert:.2f}%)")
        print(f"  Rejected C-O                  : {n_rejected_water} "
              f"({100*n_rejected_water/N_insert:.2f}%)")
        print(f"  Rejected C-C                  : {n_rejected_blocker} "
              f"({100*n_rejected_blocker/N_insert:.2f}%)")
        print(f"  Average Boltzmann Factor      : {avg_BF:.6e}")
        print(f"  Screening time                : {screen_time:.2f} s")
        print(f"  MACE time                     : {mace_time:.1f} s")
        print(f"=====================")
    
        box_vol_m3 = float(np.prod(box)) * 1e-30   # A^3 → m^3
        return avg_BF, box_vol_m3, N_target

    # =========================================================================
    # FULL TRAJECTORY ANALYSIS
    # =========================================================================

    def run_trajectory_analysis(self, mapping_file, trajectory_file,
                                target_cavity_type='small',
                                N_insert=1_000_000,
                                max_frames=None):
        """
        Complete workflow:
          1. load_cavity_centres()   -- read mapping file once
          2. read_extxyz_trajectory() -- load all production frames
          3. Widom insertion per frame
          4. Global average --> Langmuir constant
        """
        print("="*80)
        print("LANGMUIR CONSTANT — EMPTY HYDRATE + MACE TRAJECTORY")
        print("="*80)
        print(f"  Mapping file       : {mapping_file}")
        print(f"  Production traj    : {trajectory_file}")
        print(f"  Target cavity      : {target_cavity_type}")
        print(f"  Insertions/frame   : {N_insert:,}")
        print(f"  Max frames         : {'ALL' if max_frames is None else max_frames}")
        print("="*80 + "\n")

        # Phase 1
        self.load_cavity_centres(mapping_file)

        # Phase 2
        print("Loading production trajectory...")
        frames = self.read_extxyz_trajectory(trajectory_file, max_frames)
        if not frames:
            raise ValueError("No frames found in trajectory!")
        print(f"Processing {len(frames)} frames\n")

        # Phase 3 — compute isolated probe reference energy ONCE
        print("Computing isolated CH4 reference energy...")
        self.U_probe_vac = self._compute_probe_reference_energy()
        print(f"  U_probe_vac = {self.U_probe_vac:.6f} eV\n")

        # Phase 4: Widom insertion per frame
        print("="*80)
        print("PHASE 2-3: WIDOM INSERTION PER FRAME")
        print("="*80)

        frame_BF          = []
        frame_volumes     = []
        N_target_cavities = None
        start_time        = time.time()

        for idx, frame in enumerate(frames):
            frame_start = time.time()
            print(f"\nFrame {idx+1}/{len(frames)}  "
                  f"(frame_id={frame['frame_id']})")

            avg_BF, V_box, N_target = self.run_widom_insertion_frame(
                frame, target_cavity_type, N_insert
            )
            frame_BF.append(avg_BF)
            frame_volumes.append(V_box)
            if N_target_cavities is None:
                N_target_cavities = N_target

            frame_time = time.time() - frame_start
            print(f"  Result: <BF>={avg_BF:.6e}  "
                  f"V={V_box:.6e} m^3  t={frame_time:.1f}s")

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta     = elapsed / (idx + 1) * (len(frames) - idx - 1)
                print(f"\n=== Progress: {idx+1}/{len(frames)} | "
                      f"ETA: {eta/60:.1f} min ===")

        # Phase 4: global average
        print("\n" + "="*80)
        print("PHASE 4: GLOBAL AVERAGING")
        print("="*80)

        BF_array = np.array(frame_BF)
        V_array  = np.array(frame_volumes)
        mean_BF  = np.mean(BF_array)
        std_BF   = np.std(BF_array)
        sem_BF   = std_BF / np.sqrt(len(BF_array))
        mean_V   = np.mean(V_array)
        std_V    = np.std(V_array)

        print(f"  <BF>    = {mean_BF:.6e} +/- {sem_BF:.6e}")
        print(f"  Std(BF) = {std_BF:.6e}")
        print(f"  <V_box> = {mean_V:.6e} +/- {std_V:.6e} m^3")
        print(f"  N_{target_cavity_type:5s} = {N_target_cavities}")

        # Phase 5: Langmuir constant
        print("\n" + "="*80)
        print("PHASE 5: LANGMUIR CONSTANT CALCULATION")
        print("="*80)

        C_Pa      = (mean_V /
                     (N_target_cavities * self.kB_SI * self.temperature)
                     ) * mean_BF
        C_Pa_sem  = (mean_V /
                     (N_target_cavities * self.kB_SI * self.temperature)
                     ) * sem_BF
        C_bar     = C_Pa     * 1e-5
        C_bar_sem = C_Pa_sem * 1e-5

        print(f"\nFormula: C = (V_box / (N_cav x k_B x T)) x <exp(-beta DeltaU)>")
        print(f"\nLangmuir Constant ({target_cavity_type} cavities):")
        print(f"  C = {C_Pa:.6e} +/- {C_Pa_sem:.6e} Pa^-1")
        print(f"  C = {C_bar:.6e} +/- {C_bar_sem:.6e} bar^-1")
        print(f"\nTotal execution time: "
              f"{(time.time()-start_time)/60:.2f} minutes")
        print("="*80 + "\n")

        self.create_analysis_plots(BF_array, mean_BF, sem_BF,
                                   target_cavity_type)

        return {
            'frame_BF'     : BF_array,
            'frame_volumes': V_array,
            'mean_BF'      : mean_BF,
            'std_BF'       : std_BF,
            'sem_BF'       : sem_BF,
            'mean_V'       : mean_V,
            'std_V'        : std_V,
            'N_target'     : N_target_cavities,
            'C_Pa'         : C_Pa,
            'C_Pa_sem'     : C_Pa_sem,
            'C_bar'        : C_bar,
            'C_bar_sem'    : C_bar_sem,
            'temperature'  : self.temperature,
            'n_frames'     : len(frames),
            'n_insertions' : N_insert,
        }

    # =========================================================================
    # PLOTTING  (same style as LJ code)
    # =========================================================================

    def create_analysis_plots(self, BF_array, mean_BF, sem_BF, cavity_type):
        TITLE_SIZE      = 24
        AXIS_LABEL_SIZE = 20
        TICK_SIZE       = 18
        LEGEND_SIZE     = 16

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))

        ax1.plot(BF_array, color='steelblue', alpha=0.6, linewidth=0.8,
                 label='Frame BF values')
        ax1.axhline(mean_BF, color='r', linestyle='--', linewidth=2.5,
                    label=f'Mean: {mean_BF:.3e}')
        ax1.fill_between(range(len(BF_array)),
                         mean_BF - sem_BF, mean_BF + sem_BF,
                         color='r', alpha=0.15,
                         label=f'+-SEM: {sem_BF:.3e}')
        ax1.set_xlabel('Frame Number',        fontsize=AXIS_LABEL_SIZE)
        ax1.set_ylabel(r'$\langle e^{-\beta\Delta U} \rangle$',
                       fontsize=AXIS_LABEL_SIZE)
        ax1.set_title(
            f'Boltzmann Factor vs Time ({cavity_type} cages — empty hydrate)',
            fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax1.legend(fontsize=LEGEND_SIZE, loc='upper right',
                   framealpha=0.85, edgecolor='grey',
                   borderpad=0.8, labelspacing=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE,
                        width=1.5, length=6)

        ax2.hist(BF_array, bins=50, color='steelblue', edgecolor='black',
                 alpha=0.7, density=True)
        ax2.axvline(mean_BF, color='r', linestyle='--', linewidth=2.5,
                    label=f'Mean: {mean_BF:.3e}')
        ax2.axvline(mean_BF - sem_BF, color='orange',
                    linestyle=':', linewidth=2.0)
        ax2.axvline(mean_BF + sem_BF, color='orange',
                    linestyle=':', linewidth=2.0,
                    label=f'+-SEM: {sem_BF:.3e}')
        ax2.set_xlabel(r'$\langle e^{-\beta\Delta U} \rangle$',
                       fontsize=AXIS_LABEL_SIZE)
        ax2.set_ylabel('Probability Density', fontsize=AXIS_LABEL_SIZE)
        ax2.set_title('Distribution of Boltzmann Factors',
                      fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax2.legend(fontsize=LEGEND_SIZE, loc='upper right',
                   framealpha=0.85, edgecolor='grey',
                   borderpad=0.8, labelspacing=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE,
                        width=1.5, length=6)

        plt.tight_layout(pad=2.5)
        out_plot = f'langmuir_analysis_{cavity_type}_MC_mace_off24_3.4484_widom.png'
        plt.savefig(out_plot, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {out_plot}")
        plt.close()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING LANGMUIR CONSTANT CALCULATION — EMPTY HYDRATE")
    print("="*80 + "\n")

    # ------------------------------------------------------------------
    # USER-CONFIGURABLE PARAMETERS
    # ------------------------------------------------------------------
    TEMPERATURE  = 273.0       # K
    SEED         = 42
    DEVICE       = "cuda"      # 'cuda' or 'cpu'

    MODEL_PATH   = "/home/saman/mlip_model/MACE-OFF24_medium.model"
    MAPPING_FILE = "sI_2x2x2_cavity_mapping.txt"
    TRAJ_FILE    = "mace_off24_prod_npt_traj.extxyz"

    # 'small' --> probe samples 512 cages,   blockers at 48 large-cavity centres
    # 'large' --> probe samples 51262 cages, blockers at 16 small-cavity centres
    TARGET_CAVITY = 'large'

    N_INSERT   = 100000     # Widom insertions per frame
    MAX_FRAMES = None          # None = all frames

    # Literature reference: Ravipati & Punnathanam (2013), 273 K
    # small: 8.5625e-7 Pa^-1,  large: 4.4855e-6 Pa^-1
    C_LITERATURE_Pa = 4.4855e-6   # update to 4.4855e-6 when TARGET_CAVITY='large'

    # ------------------------------------------------------------------
    # Instantiate calculator
    # ------------------------------------------------------------------
    calc = MACELangmuirCalculatorEmpty(
        model_path  = MODEL_PATH,
        temperature = TEMPERATURE,
        seed        = SEED,
        device      = DEVICE,
    )

    # ------------------------------------------------------------------
    # Run full trajectory analysis
    # ------------------------------------------------------------------
    results = calc.run_trajectory_analysis(
        mapping_file       = MAPPING_FILE,
        trajectory_file    = TRAJ_FILE,
        target_cavity_type = TARGET_CAVITY,
        N_insert           = N_INSERT,
        max_frames         = MAX_FRAMES,
    )

    # ------------------------------------------------------------------
    # Literature comparison
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("LITERATURE COMPARISON")
    print("="*80)
    C_calc        = results['C_Pa']
    deviation_pct = (C_calc - C_LITERATURE_Pa) / C_LITERATURE_Pa * 100.0
    print(f"  Literature value : {C_LITERATURE_Pa:.4e} Pa^-1")
    print(f"  Calculated value : {C_calc:.4e} +/- {results['C_Pa_sem']:.4e} Pa^-1")
    print(f"  Deviation        : {deviation_pct:+.2f}%")
    print("="*80 + "\n")

    # ------------------------------------------------------------------
    # Save results  (same format as LJ code)
    # ------------------------------------------------------------------
    #output_file = f'langmuir_results_MC_empty_{TARGET_CAVITY}_ani_noCO.txt'
    #with open(output_file, 'w') as fout:
    #    fout.write("="*80 + "\n")
    #    fout.write("LANGMUIR CONSTANT — EMPTY HYDRATE\n")
    #    fout.write("="*80 + "\n\n")
    #    fout.write(f"Model path         : {MODEL_PATH}\n")
    #    fout.write(f"Mapping file       : {MAPPING_FILE}\n")
    #    fout.write(f"Production traj    : {TRAJ_FILE}\n")
    #    fout.write(f"Temperature        : {results['temperature']} K\n")
    #    fout.write(f"Cavity type        : {TARGET_CAVITY}\n")
    #    fout.write(f"N_cavities         : {results['N_target']}\n")
    #    fout.write(f"Frames processed   : {results['n_frames']}\n")
    #    fout.write(f"Insertions/frame   : {results['n_insertions']}\n\n")
    #    fout.write("Results:\n")
    #    fout.write(f"  <BF> = {results['mean_BF']:.6e} +/- {results['sem_BF']:.6e}\n")
    #    fout.write(f"  C    = {results['C_Pa']:.6e} +/- {results['C_Pa_sem']:.6e} Pa^-1\n")
    #    fout.write(f"  C    = {results['C_bar']:.6e} +/- {results['C_bar_sem']:.6e} bar^-1\n\n")
    #    fout.write(f"Literature : {C_LITERATURE_Pa:.4e} Pa^-1\n")
    #    fout.write(f"Deviation  : {deviation_pct:+.2f}%\n\n")
    #    fout.write("Frame  Boltzmann_Factor  Volume(m3)\n")
    #    for fi, (bf, vol) in enumerate(zip(results['frame_BF'],
    #                                       results['frame_volumes'])):
    #        fout.write(f"{fi:5d}  {bf:.10e}  {vol:.10e}\n")
    #    fout.write("="*80 + "\n")
#
    #np.savetxt(
    #    output_file.replace('.txt', '_data.csv'),
    #    np.column_stack([
    #        np.arange(len(results['frame_BF'])),
    #        results['frame_BF'],
    #        results['frame_volumes']
    #    ]),
    #    header="frame  boltzmann_factor  volume_m3",
    #    fmt='%d %.10e %.10e'
    #)
#
    #print(f"Results saved to : {output_file}")
    #print(f"Data CSV         : {output_file.replace('.txt', '_data.csv')}")
    print(f"\nCalculation complete!")
    print(f"Final: C = {results['C_Pa']:.6e} +/- {results['C_Pa_sem']:.6e} Pa^-1")
    print(f"       C = {results['C_bar']:.6e} +/- {results['C_bar_sem']:.6e} bar^-1")
