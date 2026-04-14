#!/usr/bin/env python3
"""
Convert ROOT xAOD (PHYSLITE) files to HDF5 format for event-level tokenization.

This script processes ATLAS DAOD_PHYSLITE files and extracts reconstructed
particles (electrons, muons, photons, jets), saving them in a structured
HDF5 format compatible with the heptokens training pipeline.

The output HDF5 mirrors the jet-level structure used by heptokens but at event
level: "events" (analogous to "jets") holds per-event summary features, and
"particles" (analogous to "tracks") holds per-particle features with a "valid"
field for masking.

Usage:
    python convert_xaod_to_h5.py input.root output.h5 [--max-events 1000]

Requirements:
    - ROOT with PyROOT
    - h5py
    - numpy
    - tqdm (for progress bars)
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Try to import ROOT, provide helpful error if not available
try:
    import ROOT

    ROOT.gROOT.SetBatch(True)  # Suppress ROOT graphics
    ROOT.xAOD.Init()
except ImportError:
    print("ERROR: ROOT with PyROOT is required but not found.")
    print("Please install ROOT with Python bindings:")
    print("  conda install -c conda-forge root")
    print("  or follow instructions at: https://root.cern/install/")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Particle type IDs assigned to each reconstructed object type
PARTICLE_TYPE_ELECTRON = 0
PARTICLE_TYPE_MUON = 1
PARTICLE_TYPE_PHOTON = 2
PARTICLE_TYPE_JET = 3

PARTICLE_TYPE_NAMES = {
    PARTICLE_TYPE_ELECTRON: "electron",
    PARTICLE_TYPE_MUON: "muon",
    PARTICLE_TYPE_PHOTON: "photon",
    PARTICLE_TYPE_JET: "jet",
}

# Maximum number of particles per event (for padding)
MAX_PARTICLES_PER_EVENT = 200

# Define the structured dtypes for HDF5 output
EVENT_DTYPE = np.dtype([
    ("n_particles", np.int32),
])

PARTICLE_DTYPE = np.dtype([
    ("valid", np.bool_),
    ("pt", np.float32),
    ("eta", np.float32),
    ("phi", np.float32),
    ("particle_type", np.int32),
])


def extract_reco_particles(
    tree,
    event_idx: int,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """
    Extract reconstructed particles from a PHYSLITE event.

    Accesses the Analysis* containers used in DAOD_PHYSLITE.

    Args:
        tree: ROOT transient tree containing the event
        event_idx: Event index to process

    Returns:
        Tuple of (pt_list, eta_list, phi_list, particle_type_list)
    """
    tree.GetEntry(event_idx)

    pt_list = []
    eta_list = []
    phi_list = []
    particle_type_list = []

    # 1. Electrons
    electrons = getattr(tree, "AnalysisElectrons", None)
    if electrons:
        n_electrons = electrons.size() if hasattr(electrons, "size") else len(electrons)
        for i in range(n_electrons):
            electron = electrons.at(i) if hasattr(electrons, "at") else electrons[i]
            pt = electron.pt() / 1000.0  # MeV -> GeV
            if pt < 7.0:
                continue
            if abs(electron.eta()) > 2.47:
                continue
            pt_list.append(pt)
            eta_list.append(electron.eta())
            phi_list.append(electron.phi())
            particle_type_list.append(PARTICLE_TYPE_ELECTRON)

    # 2. Muons
    muons = getattr(tree, "AnalysisMuons", None)
    if muons:
        n_muons = muons.size() if hasattr(muons, "size") else len(muons)
        for i in range(n_muons):
            muon = muons.at(i) if hasattr(muons, "at") else muons[i]
            pt = muon.pt() / 1000.0
            if pt < 6.0:
                continue
            if abs(muon.eta()) > 2.5:
                continue
            pt_list.append(pt)
            eta_list.append(muon.eta())
            phi_list.append(muon.phi())
            particle_type_list.append(PARTICLE_TYPE_MUON)

    # 3. Photons
    photons = getattr(tree, "AnalysisPhotons", None)
    if photons:
        n_photons = photons.size() if hasattr(photons, "size") else len(photons)
        for i in range(n_photons):
            photon = photons.at(i) if hasattr(photons, "at") else photons[i]
            pt = photon.pt() / 1000.0
            if pt < 10.0:
                continue
            if abs(photon.eta()) > 2.37:
                continue
            pt_list.append(pt)
            eta_list.append(photon.eta())
            phi_list.append(photon.phi())
            particle_type_list.append(PARTICLE_TYPE_PHOTON)

    # 4. Jets
    jets = getattr(tree, "AnalysisJets", None)
    if jets:
        n_jets = jets.size() if hasattr(jets, "size") else len(jets)
        for i in range(n_jets):
            jet = jets.at(i) if hasattr(jets, "at") else jets[i]
            pt = jet.pt() / 1000.0
            if pt < 20.0:
                continue
            if abs(jet.eta()) > 4.5:
                continue
            pt_list.append(pt)
            eta_list.append(jet.eta())
            phi_list.append(jet.phi())
            particle_type_list.append(PARTICLE_TYPE_JET)

    return pt_list, eta_list, phi_list, particle_type_list


def build_event_arrays(
    pt_list: list[float],
    eta_list: list[float],
    phi_list: list[float],
    particle_type_list: list[int],
    max_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build structured arrays for one event, padding or truncating to fixed length.

    Args:
        pt_list, eta_list, phi_list, particle_type_list: Particle data
        max_particles: Fixed particle dimension

    Returns:
        event_array: structured array of shape () with event-level fields
        particles_array: structured array of shape (max_particles,) with per-particle fields
    """
    n_real = min(len(pt_list), max_particles)

    # Event-level record
    event_array = np.zeros((), dtype=EVENT_DTYPE)
    event_array["n_particles"] = n_real

    # Particle-level records (zero-initialized = padded positions have valid=False)
    particles_array = np.zeros(max_particles, dtype=PARTICLE_DTYPE)
    particles_array["valid"][:n_real] = True
    particles_array["pt"][:n_real] = pt_list[:n_real]
    particles_array["eta"][:n_real] = eta_list[:n_real]
    particles_array["phi"][:n_real] = phi_list[:n_real]
    particles_array["particle_type"][:n_real] = particle_type_list[:n_real]

    return event_array, particles_array


def process_xaod_file(
    input_file: str,
    output_file: str,
    max_events: int | None = None,
    max_particles: int = MAX_PARTICLES_PER_EVENT,
) -> None:
    """
    Process a PHYSLITE ROOT file and convert to HDF5.

    Args:
        input_file: Path to input ROOT file
        output_file: Path to output HDF5 file
        max_events: Maximum number of events to process (None for all)
        max_particles: Maximum particles per event
    """
    logger.info(f"Processing {input_file} -> {output_file}")

    try:
        root_file = ROOT.TFile.Open(input_file, "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Cannot open ROOT file: {input_file}")

        tree = ROOT.xAOD.MakeTransientTree(root_file, "CollectionTree")
        if not tree:
            raise RuntimeError(f"Cannot find CollectionTree in {input_file}")

        n_events = tree.GetEntries()
        if max_events:
            n_events = min(n_events, max_events)

        logger.info(f"Processing {n_events} events")

        all_events = []
        all_particles = []

        for event_idx in tqdm(range(n_events), desc="Processing events"):
            try:
                pt, eta, phi, ptype = extract_reco_particles(tree, event_idx)

                # Skip events with no reconstructed particles
                if len(pt) == 0:
                    continue

                event_arr, particles_arr = build_event_arrays(
                    pt, eta, phi, ptype, max_particles
                )
                all_events.append(event_arr)
                all_particles.append(particles_arr)

            except Exception as e:
                logger.warning(f"Error processing event {event_idx}: {e}")
                continue

        all_events = np.array(all_events)
        all_particles = np.array(all_particles)

        logger.info(f"Processed {len(all_events)} events successfully")
        logger.info(f"Particles shape: {all_particles.shape}")

        # Save to HDF5 with the same structure heptokens expects:
        #   "events"    (analogous to "jets")    — structured, one row per event
        #   "particles" (analogous to "tracks")  — structured, [n_events, max_particles]
        with h5py.File(output_file, "w") as h5f:
            h5f.create_dataset("events", data=all_events, compression="gzip")
            h5f.create_dataset("particles", data=all_particles, compression="gzip")

            h5f.attrs["description"] = "Event-level particle data for tokenizer training"
            h5f.attrs["source_file"] = input_file
            h5f.attrs["n_events"] = len(all_events)
            h5f.attrs["max_particles_per_event"] = max_particles
            h5f.attrs["particle_type_names"] = str(PARTICLE_TYPE_NAMES)

        logger.info(f"Successfully saved {output_file}")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise
    finally:
        if "root_file" in locals():
            root_file.Close()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert PHYSLITE ROOT files to HDF5 for event-level tokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_xaod_to_h5.py data.root output.h5
    python convert_xaod_to_h5.py data.root output.h5 --max-events 1000 --max-particles 150
        """,
    )

    parser.add_argument("input_file", help="Input PHYSLITE ROOT file")
    parser.add_argument("output_file", help="Output HDF5 file")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process (default: all)",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=MAX_PARTICLES_PER_EVENT,
        help=f"Maximum particles per event (default: {MAX_PARTICLES_PER_EVENT})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not Path(args.input_file).exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        sys.exit(1)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        process_xaod_file(
            args.input_file,
            args.output_file,
            args.max_events,
            args.max_particles,
        )
        logger.info("Conversion completed successfully!")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
