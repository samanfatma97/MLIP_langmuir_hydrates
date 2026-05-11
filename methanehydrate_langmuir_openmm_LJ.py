"""
Langmuir Constant Calculation — EMPTY sI Clathrate Hydrate
===========================================================

Physical picture
----------------
The simulation box contains only water (TIP4P/Ice).  To compute the Langmuir
constant for methane in the SMALL cavities we need to represent the physically
correct host environment: an infinite periodic lattice of water cages where
every LARGE cavity is already occupied by a methane molecule.

To achieve this, before building the OpenMM system for each frame, the
large-cavity CH4s are explicitly inserted at the fixed mapping-file positions
into the frame.  OpenMM then applies PBC automatically, making the system
equivalent to an infinite lattice with all large cavities occupied.

Since the cages do not move appreciably during the simulation, these blocker
positions are fixed (taken from the mapping file once) and do not change
frame-to-frame.

Workflow
--------
1. Read mapping file ONCE → store small (16) and large (48) cavity centres (Å).

2. For each production frame:
   a. build_frame_with_blockers(): append 48 large-cavity CH4 atoms at their
      fixed mapping positions to the water-only frame atom list.
   b. Build OpenMM system: water O + H + 48 blocker CH4s (inert mass only).
      CustomNonbondedForce interaction group = probe ↔ water-O only.
      The CH4s sit in the system but do NOT interact with the probe energetically.
   c. Draw a random probe position uniformly from the full box.
   d. MC rejection: if probe is within REJECTION_DISTANCE (3.73 Å) of any
      large-cavity centre (minimum-image PBC) → BF = 0 (rejected).
      Minimum-image is essential here because several large-cavity centres
      sit within 3 Å of a box face (e.g. entries at y=0, z=0, x≈21 Å),
      so their periodic images must be considered for correct rejection.
   e. Otherwise: evaluate probe–water-O LJ energy with OpenMM → BF = exp(−βU).
   f. Frame average BF = sum(BF_i) / N_insert.

3. Langmuir constant:
      C = V_box / (N_small × k_B × T) × <BF>

Mapping file format (unchanged from fully-occupied code):
    ch4_id  cavity_type  x  y  z
    cavity_type: 'small' | 'large'
    x, y, z: Angstroms, Takeuchi 2×2×2 reference coordinates
    Total: 64 entries — 16 small + 48 large

Production trajectory: dump_prod_empty.lammpstrj   (O and H atoms only)
Equilibration file   : dump_eq_empty.lammpstrj     (O and H atoms only,
                       used only for box/atom-count sanity check)
"""

import numpy as np
import time
import sys
import os
from datetime import datetime
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

log_filename = "langmuir_openmm_MC_empty.log"
sys.stdout = DualLogger(log_filename)
sys.stderr = sys.stdout

print("=== Langmuir Constant — EMPTY Hydrate (OpenMM + LAMMPS Traj) ===")
print(f"Start time: {datetime.now()}")
print(f"Log file: {log_filename}\n")


# ==================== MAIN CALCULATOR CLASS ====================

class OpenMMLangmuirCalculatorEmpty:
    """
    Langmuir constant calculator for an EMPTY sI clathrate hydrate.

    Key differences from the fully-occupied (Takeuchi) version
    ----------------------------------------------------------
    - No CH4 atoms exist in the trajectory.
    - Cavity centres are read from the mapping file ONCE before the frame
      loop and kept fixed — cages do not move appreciably during simulation.
    - build_frame_with_blockers(): for each frame, the blocker-type CH4s are
      explicitly inserted at their fixed mapping positions into the frame atom
      list before the OpenMM system is built.  This makes OpenMM see the
      correct infinite-lattice host (water + occupied large cages) when PBC
      is applied.
    - The CH4 blocker atoms are inert mass in OpenMM — the CustomNonbondedForce
      interaction group is probe ↔ water-O only (unchanged from original).
    - MC rejection (minimum-image PBC, 3.73 Å) still handles geometric
      exclusion of the probe from large-cavity regions.
    - No per-frame cavity-to-atom matching required.
    """

    def __init__(self, box_size, temperature=273.0, seed=42):
        if seed is not None:
            np.random.seed(seed)
            print(f"Random seed set to: {seed}")

        self.box_size    = box_size
        self.temperature = temperature
        self.kB_SI       = 1.380649e-23

        # Fixed cavity centre positions loaded once (Å)
        self.small_centres = None   # np.ndarray (16, 3)
        self.large_centres = None   # np.ndarray (48, 3)

        # Force field: TraPPE CH4 + TIP4P/Ice, Lorentz-Berthelot mixing
        self.water_O_epsilon = 0.211032   # kcal/mol
        self.water_O_sigma   = 3.166800   # Angstrom
        self.water_H_charge  = 0.5897     # e
        self.mix_epsilon     = 0.249242   # kcal/mol  (CH4-O mixed, LB)
        self.mix_sigma       = 3.448400   # Angstrom  (CH4-O mixed, LB)
        self.cutoff          = 11.5       # Angstrom
        self.kcal_to_kJ      = 4.184
        self.angstrom_to_nm  = 0.1

        # MC rejection distance: CH4 sigma, same as original code
        self.rejection_dist  = 3.73       # Angstrom

        print("=" * 80)
        print("FORCE FIELD PARAMETERS")
        print("=" * 80)
        print(f"Water (TIP4P/Ice):  O  sigma={self.water_O_sigma:.4f} A"
              f"  eps={self.water_O_epsilon:.6f} kcal/mol")
        print(f"                    H  q={self.water_H_charge:.4f} e (no LJ)")
        print(f"CH4-Water Mixed:       sigma={self.mix_sigma:.4f} A"
              f"  eps={self.mix_epsilon:.6f} kcal/mol")
        print(f"Cutoff           : {self.cutoff} A")
        print(f"Rejection dist   : {self.rejection_dist} A  (CH4 sigma, same as original)")
        print(f"Temperature      : {self.temperature} K")
        print("=" * 80 + "\n")

    # ========================================================================
    # LAMMPS TRAJECTORY READER  (unchanged from original)
    # ========================================================================

    def read_lammps_trajectory(self, filename, max_frames=None):
        """
        Read LAMMPS dump file.
        Expected column order: id mol type element q x y z vx vy vz
        For the empty trajectory only O and H are present; no error if CH4 absent.
        """
        print(f"Reading LAMMPS trajectory: {filename}")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        frames = []
        with open(filename, 'r') as f:
            lines = f.readlines()

        i, frame_count = 0, 0
        while i < len(lines):
            if "ITEM: TIMESTEP" not in lines[i]:
                i += 1
                continue
            timestep = int(lines[i + 1].strip())
            i += 2

            if i >= len(lines) or "ITEM: NUMBER OF ATOMS" not in lines[i]:
                break
            n_atoms = int(lines[i + 1].strip())
            i += 2

            if i >= len(lines) or "ITEM: BOX BOUNDS" not in lines[i]:
                break
            box_bounds = []
            for j in range(3):
                lo, hi = lines[i + 1 + j].split()[:2]
                box_bounds.append([float(lo), float(hi)])
            box_lengths = [b[1] - b[0] for b in box_bounds]
            i += 4

            if i >= len(lines) or "ITEM: ATOMS" not in lines[i]:
                break
            atoms_data = []
            for j in range(n_atoms):
                if i + 1 + j >= len(lines):
                    break
                parts = lines[i + 1 + j].split()
                if len(parts) < 11:
                    continue
                atoms_data.append({
                    'id'     : int(parts[0]),
                    'mol'    : int(parts[1]),
                    'type'   : int(parts[2]),
                    'element': parts[3],
                    'q'      : float(parts[4]),
                    'pos'    : np.array([float(parts[5]),
                                         float(parts[6]),
                                         float(parts[7])]),
                    'vel'    : np.array([float(parts[8]),
                                         float(parts[9]),
                                         float(parts[10])])
                })

            if len(atoms_data) == n_atoms:
                frames.append({
                    'timestep': timestep,
                    'box'     : box_lengths,
                    'atoms'   : atoms_data
                })
                frame_count += 1
                if frame_count % 200 == 0:
                    print(f"  Loaded {frame_count} frames...")
                if max_frames is not None and frame_count >= max_frames:
                    break
            i += n_atoms + 1

        print(f"Total frames loaded: {len(frames)}\n")
        return frames

    # ========================================================================
    # PHASE 1: READ CAVITY CENTRES FROM MAPPING FILE  (called once)
    # ========================================================================

    def load_cavity_centres(self, mapping_file, eq_file=None):
        """
        Read the Takeuchi mapping file and store small/large cavity centres.
        Called ONCE before the frame loop — positions are fixed throughout.

        Mapping file format:
            ch4_id  cavity_type  x  y  z
            cavity_type: 'small' | 'large'
            x, y, z: Angstroms, Takeuchi 2×2×2 reference coordinates
            64 total entries: 16 small + 48 large

        For TARGET_CAVITY = 'small':  blockers = self.large_centres  (48 pts)
        For TARGET_CAVITY = 'large':  blockers = self.small_centres  (16 pts)

        Parameters
        ----------
        mapping_file : str   — path to Takeuchi cavity mapping file
        eq_file      : str   — optional path to LAMMPS equilibration dump
                               (used only for box-size / atom-count info)
        """
        print("=" * 80)
        print("PHASE 1: LOADING CAVITY CENTRES FROM MAPPING FILE  (fixed for all frames)")
        print("=" * 80)
        print(f"  Mapping file : {mapping_file}")

        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        small_list, large_list = [], []
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cavity_type = parts[1]           # 'small' or 'large'
                x, y, z     = float(parts[2]), float(parts[3]), float(parts[4])
                if cavity_type == 'small':
                    small_list.append([x, y, z])
                elif cavity_type == 'large':
                    large_list.append([x, y, z])
                else:
                    print(f"  WARNING: Unknown cavity type '{cavity_type}' — skipped.")

        self.small_centres = np.array(small_list)   # shape (16, 3)  Å
        self.large_centres = np.array(large_list)   # shape (48, 3)  Å

        print(f"\n  Cavity centres loaded:")
        print(f"    Small cages (512)   : {len(self.small_centres)}")
        print(f"    Large cages (51262) : {len(self.large_centres)}")
        print(f"    Total               : "
              f"{len(self.small_centres) + len(self.large_centres)}")

        if len(self.small_centres) != 16:
            print(f"  WARNING: Expected 16 small cages for 2×2×2 sI, "
                  f"got {len(self.small_centres)}")
        if len(self.large_centres) != 48:
            print(f"  WARNING: Expected 48 large cages for 2×2×2 sI, "
                  f"got {len(self.large_centres)}")

        # Optional sanity check on equilibration frame
        if eq_file is not None and os.path.exists(eq_file):
            print(f"\n  Checking equilibration file: {eq_file}")
            eq_frames = self.read_lammps_trajectory(eq_file, max_frames=1)
            if eq_frames:
                box   = np.array(eq_frames[0]['box'])
                atoms = eq_frames[0]['atoms']
                n_O   = sum(1 for a in atoms if a['element'] == 'O')
                n_H   = sum(1 for a in atoms if a['element'] == 'H')
                n_CH4 = sum(1 for a in atoms if a['element'] == 'CH4')
                print(f"  Eq frame box : {box} A")
                print(f"  O atoms      : {n_O}")
                print(f"  H atoms      : {n_H}")
                if n_CH4 > 0:
                    print(f"  WARNING: {n_CH4} CH4 atoms found — "
                          f"expected 0 for empty hydrate!")
                else:
                    print(f"  Confirmed: 0 CH4 atoms in eq frame (empty hydrate OK)")

        print("=" * 80 + "\n")

    # ========================================================================
    # BUILD FRAME WITH BLOCKER CH4s INSERTED
    # ========================================================================

    def build_frame_with_blockers(self, frame, target_cavity_type='small'):
        """
        Insert blocker-type CH4 atoms at their fixed mapping positions into a
        copy of the water-only frame, before building the OpenMM system.

        For target_cavity_type = 'small': insert 48 large-cavity CH4s.
        For target_cavity_type = 'large': insert 16 small-cavity CH4s.

        This makes the OpenMM system represent the physically correct host:
        an infinite periodic lattice (via PBC) of water cages with all
        blocker-type cavities occupied by methane.

        The water-only frame has 368 molecules = 1104 atoms (IDs 1–1104).
        Blocker CH4s are appended with IDs 1105, 1106, ... sequentially.

        Parameters
        ----------
        frame              : dict — water-only frame from read_lammps_trajectory()
        target_cavity_type : str  — 'small' or 'large'

        Returns
        -------
        augmented_frame : dict — same structure as input frame, with blocker
                          CH4 atoms appended to the atoms list
        blocker_centres : np.ndarray (N_blocker, 3) Å — blocker positions
        other_type      : str — cavity type of the inserted blockers
        """
        if self.small_centres is None or self.large_centres is None:
            raise RuntimeError("Call load_cavity_centres() before "
                               "build_frame_with_blockers().")

        if target_cavity_type == 'small':
            blocker_centres = self.large_centres    # (48, 3) Å
            other_type      = 'large'
        else:
            blocker_centres = self.small_centres    # (16, 3) Å
            other_type      = 'small'

        # Water-only frame has 368 molecules = 1104 atoms (IDs 1–1104).
        # Blocker CH4s are appended with IDs starting from 1105 sequentially.
        augmented_atoms = list(frame['atoms'])
        n_water_atoms   = len(augmented_atoms)   # should be 1104

        if n_water_atoms != 1104:
            print(f"  WARNING: Expected 1104 water atoms (368 molecules × 3), "
                  f"got {n_water_atoms}")

        for k, pos in enumerate(blocker_centres):
            augmented_atoms.append({
                'id'     : n_water_atoms + 1 + k,  # 1105, 1106, ... for small target
                'mol'    : n_water_atoms + 1 + k,
                'type'   : 3,                       # CH4 type (LAMMPS convention)
                'element': 'CH4',
                'q'      : 0.0,
                'pos'    : pos.copy(),              # Å, fixed mapping position
                'vel'    : np.zeros(3)
            })

        augmented_frame = {
            'timestep': frame['timestep'],
            'box'     : frame['box'],
            'atoms'   : augmented_atoms
        }

        print(f"  Inserted {len(blocker_centres)} {other_type}-cavity CH4s "
              f"at fixed mapping positions (blocker for {target_cavity_type} "
              f"cavity calculation)")

        return augmented_frame, blocker_centres, other_type

    # ========================================================================
    # WIDOM INSERTION FOR A SINGLE LAMMPS FRAME
    # ========================================================================

    def run_widom_insertion_frame(self, frame, target_cavity_type='small',
                                  N_insert=1_000_000):
        """
        Widom test-particle insertion with MC rejection for one LAMMPS frame.

        Steps:
        1. build_frame_with_blockers(): insert blocker-type CH4s at fixed
           mapping positions into the frame before building the OpenMM system.
           OpenMM PBC then makes this equivalent to an infinite lattice with
           all blocker-type cavities occupied.
        2. Build OpenMM system: water O + H + blocker CH4s (inert mass).
           CustomNonbondedForce interaction group = probe ↔ water-O only.
           Blocker CH4s do NOT interact with the probe energetically.
        3. MC rejection: probe within REJECTION_DISTANCE (3.73 Å) of any
           blocker centre (minimum-image PBC) → BF = 0.
           Minimum-image is essential — several blocker centres sit within
           3 Å of a box face, so their periodic images must be considered.
        4. Accepted insertions: compute probe–water-O LJ energy via OpenMM.

        Parameters
        ----------
        frame              : dict  — water-only frame from read_lammps_trajectory()
        target_cavity_type : str   — 'small' or 'large'
        N_insert           : int   — number of Widom insertion attempts

        Returns
        -------
        avg_BF     : float — frame-average Boltzmann factor
        box_vol_m3 : float — frame box volume in m^3
        N_target   : int   — number of target-type cavity centres
        """
        import openmm
        from openmm import unit

        if self.small_centres is None or self.large_centres is None:
            raise RuntimeError("Call load_cavity_centres() before "
                               "run_widom_insertion_frame().")

        # ---- Step 1: insert blocker CH4s into frame ----
        augmented_frame, blocker_centres, other_type = \
            self.build_frame_with_blockers(frame, target_cavity_type)

        if target_cavity_type == 'small':
            N_target = len(self.small_centres)
        else:
            N_target = len(self.large_centres)

        N_blocker  = len(blocker_centres)
        box_A      = np.array(frame['box'])
        box_nm     = box_A * self.angstrom_to_nm
        blocker_nm = blocker_centres * self.angstrom_to_nm   # (N_blocker, 3)
        reject_nm  = self.rejection_dist * self.angstrom_to_nm

        print(f"  Target cavity  : {target_cavity_type}  (N={N_target})")
        print(f"  Blocker cavity : {other_type}  (N={N_blocker})")
        print(f"  Rejection dist : {self.rejection_dist} A")
        print(f"  Box (A)        : {box_A}")

        # ---- Step 2: Build OpenMM system: water O + H + blocker CH4s ----
        system = openmm.System()
        system.setDefaultPeriodicBoxVectors(
            [box_nm[0], 0, 0] * unit.nanometer,
            [0, box_nm[1], 0] * unit.nanometer,
            [0, 0, box_nm[2]] * unit.nanometer
        )

        water_O_indices = []
        positions       = []
        atom_types      = []

        for atom in augmented_frame['atoms']:
            elem   = atom['element']
            pos_nm = atom['pos'] * self.angstrom_to_nm
            if elem == 'O':
                system.addParticle(15.999 * unit.amu)
                water_O_indices.append(len(positions))
                atom_types.append('O')
                positions.append(pos_nm)
            elif elem == 'H':
                system.addParticle(1.008 * unit.amu)
                atom_types.append('H')
                positions.append(pos_nm)
            elif elem == 'CH4':
                # Blocker CH4: added as inert mass — NOT in interaction group
                system.addParticle(16.043 * unit.amu)
                atom_types.append('CH4_blocker')
                positions.append(pos_nm)

        # Ghost probe particle (TraPPE CH4 mass)
        system.addParticle(16.043 * unit.amu)
        probe_index = len(positions)
        positions.append(np.array([0.0, 0.0, 0.0]))
        atom_types.append('PROBE')

        n_ch4_blockers = sum(1 for t in atom_types if t == 'CH4_blocker')
        print(f"  System: {len(water_O_indices)} water-O  |  "
              f"{n_ch4_blockers} blocker CH4s (inert)  |  "
              f"{probe_index} atoms + 1 probe")

        # ---- Probe <-> Water-O LJ force (CustomNonbondedForce) ----
        mix_sigma_nm = self.mix_sigma   * self.angstrom_to_nm
        mix_eps_kJ   = self.mix_epsilon * self.kcal_to_kJ
        energy_expr  = (f"4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
                        f"epsilon={mix_eps_kJ}; sigma={mix_sigma_nm}")

        custom_force = openmm.CustomNonbondedForce(energy_expr)
        custom_force.setNonbondedMethod(
            openmm.CustomNonbondedForce.CutoffPeriodic)
        custom_force.setCutoffDistance(
            self.cutoff * self.angstrom_to_nm * unit.nanometer)
        custom_force.setUseLongRangeCorrection(True)

        for _ in atom_types:
            custom_force.addParticle([])

        # Only probe ↔ water-O interaction computed
        custom_force.addInteractionGroup({probe_index}, set(water_O_indices))
        system.addForce(custom_force)

        # ---- CUDA context ----
        platform   = openmm.Platform.getPlatformByName('CUDA')
        properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        context    = openmm.Context(system, integrator, platform, properties)

        actual_platform = context.getPlatform().getName()
        print(f"\n=== CUDA VERIFICATION ===")
        print(f"  Platform : {actual_platform}")
        if actual_platform != 'CUDA':
            raise RuntimeError(
                f"CUDA requested but '{actual_platform}' is in use!")
        print(f"  Device   : "
              f"{context.getPlatform().getPropertyValue(context, 'DeviceIndex')}")
        print(f"  Precision: "
              f"{context.getPlatform().getPropertyValue(context, 'Precision')}")
        print(f"=========================\n")

        positions_array = np.array(positions)
        context.setPositions(
            [openmm.Vec3(p[0], p[1], p[2]) for p in positions_array]
            * unit.nanometer
        )

        box_volume_m3 = np.prod(box_nm) * 1e-27   # nm³ → m³

        # ---- Widom insertion loop ----
        sum_BF = 0.0
        kB_kJ  = 0.0083144621          # kJ mol⁻¹ K⁻¹
        beta   = 1.0 / (kB_kJ * self.temperature)
        n_acc  = 0
        n_rej  = 0

        for i in range(N_insert):
            # Uniform random position in box
            random_pos_nm = np.random.uniform(0, box_nm, 3)

            # MC rejection: is probe inside a blocker (large-cavity) region?
            # Minimum-image PBC — no explicit wrapping of blocker_nm required.
            is_rejected = False
            if N_blocker > 0:
                delta = random_pos_nm - blocker_nm       # (N_blocker, 3)
                delta = delta - box_nm * np.round(delta / box_nm)
                if np.min(np.linalg.norm(delta, axis=1)) < reject_nm:
                    is_rejected = True
                    n_rej += 1

            if is_rejected:
                BF_i = 0.0
            else:
                n_acc += 1
                positions_array[probe_index] = random_pos_nm
                context.setPositions(
                    [openmm.Vec3(p[0], p[1], p[2]) for p in positions_array]
                    * unit.nanometer
                )
                state = context.getState(getEnergy=True)
                U_kJ  = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole)
                BF_i  = np.exp(-beta * U_kJ)

            sum_BF += BF_i

            if (i + 1) % 2000 == 0:
                acc_pct = n_acc / (i + 1) * 100
                print(f"  Insertion {i+1}/{N_insert}: "
                      f"<BF>={sum_BF/(i+1):.6e} | "
                      f"Accepted: {acc_pct:.1f}% ({n_acc}) | "
                      f"Rejected: {n_rej}")

        avg_BF   = sum_BF / N_insert
        acc_rate = n_acc / N_insert * 100
        print(f"\n=== Monte Carlo Statistics ===")
        print(f"  Total insertions              : {N_insert}")
        print(f"  Accepted ({target_cavity_type:5s} region) : "
              f"{n_acc} ({acc_rate:.2f}%)")
        print(f"  Rejected ({other_type:5s} blocked) : "
              f"{n_rej} ({100 - acc_rate:.2f}%)")
        print(f"  Average Boltzmann Factor      : {avg_BF:.6e}")
        print(f"==============================\n")

        del context
        del integrator
        return avg_BF, box_volume_m3, N_target

    # ========================================================================
    # FULL TRAJECTORY ANALYSIS
    # ========================================================================

    def run_trajectory_analysis(self,
                                eq_file,
                                mapping_file,
                                trajectory_file,
                                target_cavity_type='small',
                                N_insert=1_000_000,
                                max_frames=None):
        """
        Complete workflow for empty hydrate:
          1. load_cavity_centres() — read mapping file ONCE, fix blocker positions
          2. Load all production frames (O and H only)
          3. Widom insertion per frame
          4. Global average → Langmuir constant
        """
        print("=" * 80)
        print("LANGMUIR CONSTANT — EMPTY HYDRATE + LAMMPS TRAJECTORY")
        print("=" * 80)
        print(f"  Equilibration file : {eq_file}")
        print(f"  Mapping file       : {mapping_file}")
        print(f"  Production traj    : {trajectory_file}")
        print(f"  Target cavity      : {target_cavity_type}")
        print(f"  Insertions/frame   : {N_insert}")
        print("=" * 80 + "\n")

        # ---- Phase 1: cavity centres, fixed for all frames ----
        self.load_cavity_centres(mapping_file, eq_file=eq_file)

        # ---- Phase 2: load production frames ----
        print("Loading production trajectory...")
        frames = self.read_lammps_trajectory(trajectory_file, max_frames)
        if not frames:
            raise ValueError("No frames found in production trajectory!")
        print(f"Processing {len(frames)} frames\n")

        # ---- Phase 3: Widom insertion per frame ----
        print("=" * 80)
        print("PHASE 2-3: WIDOM INSERTION PER FRAME")
        print("=" * 80)

        frame_BF          = []
        frame_volumes     = []
        N_target_cavities = None
        start_time        = time.time()

        for idx, frame in enumerate(frames):
            frame_start = time.time()
            print(f"\nFrame {idx+1}/{len(frames)}  "
                  f"(Timestep={frame['timestep']})")

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

        # ---- Phase 4: global average ----
        print("\n" + "=" * 80)
        print("PHASE 4: GLOBAL AVERAGING")
        print("=" * 80)

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

        # ---- Phase 5: Langmuir constant ----
        print("\n" + "=" * 80)
        print("PHASE 5: LANGMUIR CONSTANT CALCULATION")
        print("=" * 80)

        C_Pa      = (mean_V /
                     (N_target_cavities * self.kB_SI * self.temperature)
                     ) * mean_BF
        C_Pa_sem  = (mean_V /
                     (N_target_cavities * self.kB_SI * self.temperature)
                     ) * sem_BF
        C_bar     = C_Pa     * 1e-5
        C_bar_sem = C_Pa_sem * 1e-5

        print(f"\nFormula: C = (V_box / (N_cav × k_B × T)) × <exp(-βU)>")
        print(f"\nLangmuir Constant ({target_cavity_type} cavities):")
        print(f"  C = {C_Pa:.6e} +/- {C_Pa_sem:.6e} Pa^-1")
        print(f"  C = {C_bar:.6e} +/- {C_bar_sem:.6e} bar^-1")
        print(f"\nTotal execution time: "
              f"{(time.time()-start_time)/60:.2f} minutes")
        print("=" * 80 + "\n")

        self.create_analysis_plots(BF_array, mean_BF, sem_BF, target_cavity_type)

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

    # ========================================================================
    # XYZ WRITER FOR VMD VERIFICATION
    # ========================================================================

    def write_xyz_for_verification(self, traj_file, mapping_file,
                                   target_cavity_type='small',
                                   out_xyz='verify_augmented_frame.xyz'):
        """
        Write an XYZ file of the first frame of the PRODUCTION trajectory
        with blocker CH4s inserted, for visual verification in VMD.

        Element labels used in the XYZ file:
            O  — water oxygen
            H  — water hydrogen
            C  — blocker CH4 (labelled as carbon so VMD colours it distinctly)

        Box dimensions are written as a comment line in extended XYZ format:
            Lattice="ax 0 0  0 ay 0  0 0 az"
        so VMD/OVITO can draw the periodic box correctly.

        Parameters
        ----------
        traj_file          : str — LAMMPS production dump (first frame used)
        mapping_file       : str — Takeuchi cavity mapping file
        target_cavity_type : str — 'small' or 'large'
        out_xyz            : str — output filename
        """
        print(f"\nWriting verification XYZ: {out_xyz}")
        print(f"  Target cavity : {target_cavity_type}")
        print(f"  Using first frame of production trajectory: {traj_file}")

        # Load cavity centres if not already done
        if self.small_centres is None or self.large_centres is None:
            self.load_cavity_centres(mapping_file)

        # Read first frame of production trajectory only
        frames = self.read_lammps_trajectory(traj_file, max_frames=1)
        if not frames:
            raise ValueError(f"No frames found in {traj_file}")
        frame = frames[0]

        # Augment with blocker CH4s at fixed mapping positions
        augmented_frame, _, other_type = \
            self.build_frame_with_blockers(frame, target_cavity_type)

        atoms   = augmented_frame['atoms']
        box     = augmented_frame['box']
        n_atoms = len(atoms)

        # CH4 → 'C' so VMD treats it as a distinct atom type (cyan/grey)
        element_map = {'O': 'O', 'H': 'H', 'CH4': 'C'}

        with open(out_xyz, 'w') as f:
            # Line 1: total atom count
            f.write(f"{n_atoms}\n")
            # Line 2: extended XYZ comment — box for VMD/OVITO
            f.write(
                f'Lattice="{box[0]:.6f} 0.000000 0.000000  '
                f'0.000000 {box[1]:.6f} 0.000000  '
                f'0.000000 0.000000 {box[2]:.6f}" '
                f'Properties=species:S:1:pos:R:3 '
                f'target_cavity={target_cavity_type} '
                f'blocker={other_type} '
                f'timestep={frame["timestep"]}\n'
            )
            # Atom lines: element  x  y  z
            for atom in atoms:
                elem    = element_map.get(atom['element'], atom['element'])
                x, y, z = atom['pos']
                f.write(f"{elem:2s}  {x:14.6f}  {y:14.6f}  {z:14.6f}\n")

        # Confirmation summary
        n_O   = sum(1 for a in atoms if a['element'] == 'O')
        n_H   = sum(1 for a in atoms if a['element'] == 'H')
        n_CH4 = sum(1 for a in atoms if a['element'] == 'CH4')
        print(f"\n  Written {n_atoms} atoms to {out_xyz}:")
        print(f"    O  (water oxygen)   : {n_O}")
        print(f"    H  (water hydrogen) : {n_H}")
        print(f"    C  (blocker CH4)    : {n_CH4}  [{other_type} cavities]")
        print(f"  Box : {box} A")
        print(f"\n  To load in VMD:")
        print(f"    vmd {out_xyz}")
        print(f"  Suggested representations:")
        print(f"    name O  → VDW, red    (water oxygen)")
        print(f"    name H  → VDW, white  (water hydrogen)")
        print(f"    name C  → VDW, cyan   (blocker CH4 at {other_type} cavity centres)")

    # ========================================================================
    # PLOTTING  (unchanged style from original)
    # ========================================================================

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
                         label=f'\u00b1SEM: {sem_BF:.3e}')
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
                    label=f'\u00b1SEM: {sem_BF:.3e}')
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
        out_plot = f'langmuir_analysis_{cavity_type}_MC_empty.png'
        plt.savefig(out_plot, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {out_plot}")
        plt.close()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING LANGMUIR CONSTANT CALCULATION — EMPTY HYDRATE")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # USER-CONFIGURABLE PARAMETERS
    # ------------------------------------------------------------------
    TEMPERATURE     = 273.0        # K
    BOX_SIZE_INIT   = 24.06        # Å  (initial guess; actual read from traj)
    SEED            = 42

    EQ_FILE         = "dump_eq_empty.lammpstrj"
    MAPPING_FILE    = "sI_2x2x2_cavity_mapping.txt"
    TRAJECTORY_FILE = "dump_prod_empty.lammpstrj"

    # 'small' → C for 512 cages     (blockers = 48 large-cavity centres)
    # 'large' → C for 51262 cages   (blockers = 16 small-cavity centres)
    TARGET_CAVITY   = 'small'

    N_INSERT        = 10000    # Widom insertions per frame
    MAX_FRAMES      = None         # None = all frames

    # Literature reference: Ravipati & Punnathanam (2015), 273 K, small cage
    C_LITERATURE_Pa = 8.5625e-7    # Pa^-1

    # ------------------------------------------------------------------
    # Instantiate calculator
    # ------------------------------------------------------------------
    calc = OpenMMLangmuirCalculatorEmpty(
        box_size    = BOX_SIZE_INIT,
        temperature = TEMPERATURE,
        seed        = SEED,
    )

    # ------------------------------------------------------------------
    # Write verification XYZ for VMD (first production frame + blockers)
    # Inspect in VMD before committing to the full trajectory analysis
    # ------------------------------------------------------------------
    calc.write_xyz_for_verification(
        traj_file          = TRAJECTORY_FILE,
        mapping_file       = MAPPING_FILE,
        target_cavity_type = TARGET_CAVITY,
        out_xyz            = f'verify_{TARGET_CAVITY}_cavity_blockers.xyz'
    )

    # ------------------------------------------------------------------
    # Run full trajectory analysis
    # ------------------------------------------------------------------

    results = calc.run_trajectory_analysis(
        eq_file            = EQ_FILE,
        mapping_file       = MAPPING_FILE,
        trajectory_file    = TRAJECTORY_FILE,
        target_cavity_type = TARGET_CAVITY,
        N_insert           = N_INSERT,
        max_frames         = MAX_FRAMES,
    )

    # ------------------------------------------------------------------
    # Literature comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LITERATURE COMPARISON")
    print("=" * 80)
    C_calc        = results['C_Pa']
    deviation_pct = (C_calc - C_LITERATURE_Pa) / C_LITERATURE_Pa * 100.0
    print(f"  Literature value : {C_LITERATURE_Pa:.4e} Pa^-1")
    print(f"  Calculated value : {C_calc:.4e} +/- {results['C_Pa_sem']:.4e} Pa^-1")
    print(f"  Deviation        : {deviation_pct:+.2f}%")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_file = f'langmuir_results_MC_empty_{TARGET_CAVITY}.txt'
    with open(output_file, 'w') as fout:
        fout.write("=" * 80 + "\n")
        fout.write("LANGMUIR CONSTANT — EMPTY HYDRATE\n")
        fout.write("=" * 80 + "\n\n")
        fout.write(f"Equilibration file : {EQ_FILE}\n")
        fout.write(f"Mapping file       : {MAPPING_FILE}\n")
        fout.write(f"Production traj    : {TRAJECTORY_FILE}\n")
        fout.write(f"Temperature        : {results['temperature']} K\n")
        fout.write(f"Cavity type        : {TARGET_CAVITY}\n")
        fout.write(f"N_cavities         : {results['N_target']}\n")
        fout.write(f"Frames processed   : {results['n_frames']}\n")
        fout.write(f"Insertions/frame   : {results['n_insertions']}\n\n")
        fout.write("Results:\n")
        fout.write(f"  <BF> = {results['mean_BF']:.6e} +/- {results['sem_BF']:.6e}\n")
        fout.write(f"  C    = {results['C_Pa']:.6e} +/- {results['C_Pa_sem']:.6e} Pa^-1\n")
        fout.write(f"  C    = {results['C_bar']:.6e} +/- {results['C_bar_sem']:.6e} bar^-1\n\n")
        fout.write(f"Literature : {C_LITERATURE_Pa:.4e} Pa^-1\n")
        fout.write(f"Deviation  : {deviation_pct:+.2f}%\n\n")
        fout.write("Frame  Boltzmann_Factor  Volume(m3)\n")
        for fi, (bf, vol) in enumerate(zip(results['frame_BF'],
                                           results['frame_volumes'])):
            fout.write(f"{fi:5d}  {bf:.10e}  {vol:.10e}\n")
        fout.write("=" * 80 + "\n")

    np.savetxt(
        output_file.replace('.txt', '_data.csv'),
        np.column_stack([
            np.arange(len(results['frame_BF'])),
            results['frame_BF'],
            results['frame_volumes']
        ]),
        header="frame  boltzmann_factor  volume_m3",
        fmt='%d %.10e %.10e'
    )

    print(f"Results saved to : {output_file}")
    print(f"Data CSV         : {output_file.replace('.txt', '_data.csv')}")
    print(f"\nCalculation complete!")
    print(f"Final: C = {results['C_Pa']:.6e} +/- {results['C_Pa_sem']:.6e} Pa^-1")
    print(f"       C = {results['C_bar']:.6e} +/- {results['C_bar_sem']:.6e} bar^-1")
