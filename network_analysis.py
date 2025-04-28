"""
GTFS Transit Network Analysis Tool

This script loads GTFS (General Transit Feed Specification) data, builds a
network graph representation, performs various network analyses, and saves
the results to CSV files.

Analyses Included:
1.  Network Structure:
    - Connectivity (Beta, Gamma, Density)
    - Centrality (Degree, Closeness, Betweenness)
    - Accessibility (Cumulative opportunities to target stops)
2.  Route Characteristics:
    - Route Overlap (Jaccard Index / Overlap Coefficient)
    - Route Circuity (Actual vs. Shortest Path Time)
3.  Operational Aspects:
    - Transfer Analysis (Number of transfers on shortest paths)
4.  Network Robustness:
    - Resilience Analysis (Impact of removing critical nodes)
5.  External Comparison (Optional):
    - Comparison with Google Directions API travel times.

Setup:
1.  Place your GTFS files (stops.txt, routes.txt, trips.txt, stop_times.txt)
    in the directory specified by `GTFS_DIR`.
2.  (Optional) If you have a separate CSV for specific stops,
    place it at the path specified by `EDSA_STOPS_CSV` and ensure it has
    columns like 'Stop Name', 'Latitude', 'Longitude', and optionally 'stop_id'.
3.  Create the output directory specified by `OUTPUT_DIR` or the script
    will attempt to create it.
4.  (Optional) If using the Google API comparison, obtain an API key and
    set the `GOOGLE_API_KEY` variable (ensure this key is kept secure and
    not hardcoded directly if sharing the script).
"""

import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json # Added for saving route sequences

# Third-party libraries
try:
    import networkx as nx
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing essential library ({e}). Please install requirements.")
    print("pip install pandas numpy networkx")
    exit(1)

try:
    import googlemaps # Optional
except ImportError:
    googlemaps = None # Flag that the library is not available


# --- Configuration Constants ---

# Directories
GTFS_DIR: str = './gtfs/'  # Directory containing GTFS files
EDSA_STOPS_CSV: Optional[str] = './edsa_stops.csv' # Optional path to additional stops CSV
OUTPUT_DIR: str = './network_analysis_output/' # Directory for saving results

# Analysis Parameters
TARGET_STOP_PREFIXES: List[str] = ['LRT', 'MRT'] # Prefixes/names to identify target stops
TIME_FORMAT: str = '%H:%M:%S' # GTFS time format
CPU_INTENSIVE_THRESHOLD: int = 5000 # Warn if graph nodes exceed this
DEFAULT_TRANSFER_PENALTY_SECONDS: int = 180 # Conceptual penalty for transfers
RESILIENCE_TOP_N: int = 5 # Nodes to remove in resilience tests (per metric)
ACCESSIBILITY_THRESHOLDS_MIN: List[int] = [15, 30, 45, 60] # Time bands for accessibility
ROUTE_OVERLAP_THRESHOLD: float = 0.60 # Minimum overlap value to report
ROUTE_OVERLAP_METHOD: str = 'jaccard' #
TRANSFER_ANALYSIS_SAMPLE_SIZE: int = 1000 # Number of OD pairs for transfer analysis
RESILIENCE_ACCESSIBILITY_METRIC: str = f'reachable_targets_{max(ACCESSIBILITY_THRESHOLDS_MIN)}min' # Which accessibility column to track for resilience

# --- Google API Configuration (Optional) ---
GOOGLE_API_KEY: Optional[str] = "YOUR_API_KEY" # Set to None to disable
if GOOGLE_API_KEY == "YOUR_API_KEY":
    GOOGLE_API_KEY = None # Default to disabled if placeholder is unchanged
# Google API Comparison Parameters
NUM_API_COMPARISONS: int = 10        # Number of OD pairs to compare
NUM_TOP_HUBS_FOR_SAMPLING: int = 30 # Number of top hubs (by betweenness) to consider
API_CALL_DELAY_SECONDS: float = 0.5   # Delay between comparing OD pairs (to pace API calls)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas') # Suppress specific pandas warnings

# --- Helper Functions ---

def parse_time(time_str: str) -> Optional[float]:
    """
    Parses HH:MM:SS string into total seconds from midnight.

    Handles times potentially exceeding 23:59:59 

    Args:
        time_str: The time string in HH:MM:SS format.

    Returns:
        Total seconds from midnight as a float, or None if parsing fails.
    """
    if not isinstance(time_str, str):
        return None
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) != 3:
            return None
        delta = timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
        return delta.total_seconds()
    except (ValueError, TypeError, AttributeError):
        logging.debug(f"Failed to parse time string: {time_str}", exc_info=True)
        return None

def format_seconds(seconds: Optional[Union[int, float]]) -> str:
    """
    Formats a duration in seconds into a HH:MM:SS string.

    Handles None, NaN, or infinite inputs gracefully.

    Args:
        seconds: The duration in seconds.

    Returns:
        A formatted HH:MM:SS string, "N/A" for invalid inputs,
        or "Infinite" for non-finite numbers.
    """
    if seconds is None or pd.isna(seconds):
        return "N/A"
    if not np.isfinite(seconds):
        return "Infinite"
    try:
        seconds_int = int(round(seconds)) # Round to nearest second for display
        if seconds_int < 0:
            return "Invalid Time" # Handle negative time if relevant
        hours = seconds_int // 3600
        minutes = (seconds_int % 3600) // 60
        secs = seconds_int % 60
        return f"{hours:02}:{minutes:02}:{secs:02}"
    except (ValueError, TypeError):
        logging.debug(f"Failed to format seconds: {seconds}", exc_info=True)
        return "Invalid Time"

def create_output_dir(dir_path: str) -> bool:
    """
    Creates the output directory if it doesn't already exist.

    Args:
        dir_path: The path to the directory to create.

    Returns:
        True if the directory exists or was created successfully, False otherwise.
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logging.info(f"Created output directory: {dir_path}")
            return True
        except OSError as e:
            logging.error(f"Failed to create output directory {dir_path}: {e}")
            return False
    return True

def safe_save_csv(df: pd.DataFrame, path: str, description: str) -> None:
    """Safely saves a DataFrame to CSV with logging."""
    try:
        df.to_csv(path, index=False)
        logging.info(f"Successfully saved {description} to {path}")
    except Exception as e:
        logging.error(f"Failed to save {description} to {path}: {e}")


# --- Data Loading Functions ---

def load_gtfs_data(gtfs_dir: str, edsa_stops_csv: Optional[str] = None) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Loads essential GTFS files, optionally merges additional stops, validates,
    and filters data.

    Args:
        gtfs_dir: Path to the directory containing GTFS .txt files.
        edsa_stops_csv: Optional path to a CSV file with additional stops.
                        Expected columns: 'Stop Name', 'Latitude', 'Longitude'.
                        'stop_id' is optional; will be generated if missing.

    Returns:
        A dictionary containing DataFrames for 'stops', 'routes', 'trips',
        'stop_times', 'stops_valid' (filtered), and 'stop_times_filtered'.
        Returns None if essential files are missing or critical errors occur.
    """
    logging.info(f"Loading GTFS data from: {gtfs_dir}")
    data: Dict[str, pd.DataFrame] = {}
    required_files: List[str] = ['stops.txt', 'stop_times.txt', 'trips.txt', 'routes.txt']
    # Define dtypes for potential columns across files
    dtype_map: Dict[str, Any] = {
        'stop_id': str, 'trip_id': str, 'route_id': str, 'service_id': str,
        'parent_station': str, 'stop_sequence': 'Int64', # Use nullable integer
        'stop_lat': float, 'stop_lon': float,
        'arrival_time': str, 'departure_time': str,
        # For EDSA CSV mapping:
        'Latitude': float, 'Longitude': float, 'Stop Name': str,
    }

    try:
        # Load required GTFS files
        for filename in required_files:
            file_path = os.path.join(gtfs_dir, filename)
            if not os.path.exists(file_path):
                logging.error(f"Required GTFS file not found: {file_path}")
                return None
            try:
                # Determine applicable dtypes for this file
                file_cols = pd.read_csv(file_path, nrows=0).columns
                applicable_dtypes = {k: v for k, v in dtype_map.items() if k in file_cols}
                df = pd.read_csv(file_path, dtype=applicable_dtypes, low_memory=False)
                data[filename.split('.')[0]] = df
                logging.info(f"Loaded {len(df):,} records from {filename}")
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                return None

        stops_df = data['stops']

        # Handle Optional Additional Stops
        if edsa_stops_csv and os.path.exists(edsa_stops_csv):
            logging.info(f"Attempting to load and merge additional stops from {edsa_stops_csv}.")
            try:
                edsa_cols = pd.read_csv(edsa_stops_csv, nrows=0).columns
                edsa_dtypes = {k: dtype_map[k] for k in dtype_map if k in edsa_cols}
                edsa_stops = pd.read_csv(edsa_stops_csv, dtype=edsa_dtypes)
                logging.info(f"Read {len(edsa_stops):,} rows from {edsa_stops_csv}.")

                # Standardize columns to match GTFS stops.txt
                rename_map = {'Stop Name': 'stop_name', 'Latitude': 'stop_lat', 'Longitude': 'stop_lon'}
                edsa_stops.rename(columns=rename_map, inplace=True)

                # Generate stop_id if missing, ensuring uniqueness
                if 'stop_id' not in edsa_stops.columns or edsa_stops['stop_id'].isnull().any():
                    logging.warning("'stop_id' column missing or contains nulls in EDSA stops CSV. Generating unique IDs.")
                    # Simple sequential ID
                    edsa_stops['stop_id'] = [f"EDSA_{i+1}" for i in range(len(edsa_stops))]

                # Ensure required columns exist after potential generation/rename
                required_edsa_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
                if not all(col in edsa_stops.columns for col in required_edsa_cols):
                    logging.error(f"EDSA stops CSV missing required columns after processing: {required_edsa_cols}. Skipping merge.")
                elif edsa_stops[required_edsa_cols].isnull().any().any():
                     logging.error(f"EDSA stops CSV contains null values in required columns after processing. Skipping merge.")
                else:
                    edsa_stops['stop_id'] = edsa_stops['stop_id'].astype(str)
                    # Remove stops from original GTFS that have conflicting IDs with EDSA stops
                    original_count = len(stops_df)
                    stops_df = stops_df[~stops_df['stop_id'].isin(edsa_stops['stop_id'])]
                    if len(stops_df) < original_count:
                         logging.info(f"Removed {original_count - len(stops_df)} stops from original GTFS data due to ID conflict with EDSA stops.")

                    # Align columns and concatenate
                    gtfs_cols = data['stops'].columns
                    edsa_stops_aligned = edsa_stops.reindex(columns=gtfs_cols, fill_value=None) # Ensure same columns

                    stops_df = pd.concat([stops_df, edsa_stops_aligned[gtfs_cols]], ignore_index=True)
                    logging.info(f"Successfully merged/appended {len(edsa_stops)} stops from {edsa_stops_csv}. Total stops now: {len(stops_df):,}")
                    data['stops'] = stops_df # Update the main dictionary
            except Exception as e:
                logging.error(f"Error processing EDSA stops file {edsa_stops_csv}: {e}")

        # Data Validation and Filtering
        # Validate Stops: Ensure essential fields exist and drop duplicates/NaN coords
        if not all(col in stops_df.columns for col in ['stop_id', 'stop_lat', 'stop_lon']):
             logging.error("stops.txt missing one or more required columns: 'stop_id', 'stop_lat', 'stop_lon'.")
             return None

        stops_df.dropna(subset=['stop_lat', 'stop_lon'], inplace=True)
        stops_df['stop_id'] = stops_df['stop_id'].astype(str) # Ensure consistent type
        initial_stop_count = len(stops_df)
        stops_df.drop_duplicates(subset=['stop_id'], keep='first', inplace=True)
        if len(stops_df) < initial_stop_count:
             logging.warning(f"Removed {initial_stop_count - len(stops_df)} duplicate stop_ids. Kept first occurrence.")
        logging.info(f"Validated stops: {len(stops_df):,} unique stops with valid coordinates.")
        data['stops_valid'] = stops_df.copy()

        # Filter stop_times based on valid stops
        valid_stop_ids = set(data['stops_valid']['stop_id'].unique())
        stop_times_df = data['stop_times']
        if 'stop_id' not in stop_times_df.columns or 'trip_id' not in stop_times_df.columns:
            logging.error("stop_times.txt missing 'stop_id' or 'trip_id'.")
            return None
        stop_times_df['stop_id'] = stop_times_df['stop_id'].astype(str) # Ensure consistent type
        stop_times_filtered = stop_times_df[stop_times_df['stop_id'].isin(valid_stop_ids)].copy()
        logging.info(f"Filtered stop_times: {len(stop_times_filtered):,} records match valid stops (out of {len(stop_times_df):,}).")
        data['stop_times_filtered'] = stop_times_filtered

        # Check for missing trips referenced in filtered stop_times
        trips_in_stop_times = set(stop_times_filtered['trip_id'].unique())
        trips_df = data['trips']
        if 'trip_id' not in trips_df.columns:
            logging.error("trips.txt missing 'trip_id'.")
            return None
        valid_trip_ids = set(trips_df['trip_id'].unique())
        missing_trips = trips_in_stop_times - valid_trip_ids
        if missing_trips:
            logging.warning(f"{len(missing_trips):,} trip_ids found in stop_times but are missing from trips.txt. These sequences will be ignored.")
            # Optionally, filter stop_times further to remove these invalid trips
            stop_times_filtered = stop_times_filtered[stop_times_filtered['trip_id'].isin(valid_trip_ids)]
            data['stop_times_filtered'] = stop_times_filtered
            logging.info(f"Updated filtered stop_times after removing missing trips: {len(stop_times_filtered):,} records.")

        return data

    except FileNotFoundError as e:
        logging.error(f"Error loading GTFS data: {e}")
        return None
    except KeyError as e:
        logging.error(f"Missing expected column in GTFS data: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during GTFS loading: {e}", exc_info=True)
        return None


# --- Network Graph Construction ---

def build_transit_graph(
    gtfs_data: Dict[str, pd.DataFrame],
    use_travel_time: bool = True,
    ) -> Optional[nx.DiGraph]:
    """
    Builds a directed NetworkX graph from GTFS data.

    Nodes represent stops, edges represent travel segments between stops
    within a trip sequence or intra-station transfers. Edge weights can be
    based on scheduled travel time or be uniform (1.0).

    Args:
        gtfs_data: Dictionary containing GTFS DataFrames (output of load_gtfs_data).
        use_travel_time: If True, edge weights represent scheduled travel time
                         in seconds. If False, edge weights are 1.0.
        # transfer_penalty_seconds: (Not used for edge weights here) A conceptual
        #                           value representing transfer inconvenience.

    Returns:
        A NetworkX DiGraph representing the transit network, or None if
        essential data is missing or graph construction fails.
    """
    logging.info("Building transit network graph")
    start_time = time.time()
    G = nx.DiGraph()

    # Check for required data
    required_keys = ['stops_valid', 'stop_times_filtered', 'trips', 'routes']
    if not all(key in gtfs_data for key in required_keys):
        logging.error("Missing essential dataframes for graph building.")
        return None
    stops_df = gtfs_data['stops_valid']
    stop_times_df = gtfs_data['stop_times_filtered']
    trips_df = gtfs_data['trips']
    routes_df = gtfs_data['routes'] # Needed for route info on edges

    if stops_df.empty or stop_times_df.empty or trips_df.empty:
        logging.error("Essential dataframes (stops, stop_times, trips) are empty. Cannot build graph.")
        return None

    # 1. Add Stop Nodes
    node_count = 0
    for _, stop in stops_df.iterrows():
        G.add_node(
            stop['stop_id'],
            name=stop.get('stop_name', 'N/A'),
            lat=stop['stop_lat'],
            lon=stop['stop_lon'],
            parent_station=stop.get('parent_station', None) # Use None if missing
        )
        node_count += 1
    logging.info(f"Added {node_count:,} stop nodes.")
    if G.number_of_nodes() > CPU_INTENSIVE_THRESHOLD:
        logging.warning(f"Network graph has {G.number_of_nodes():,} nodes. Some analyses might be computationally intensive.")

    # 2. Prepare Stop Times Data for Edge Creation
    logging.info("Processing stop_times for edge creation")
    # Merge route_id from trips
    stop_times_processed = pd.merge(
        stop_times_df[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']],
        trips_df[['trip_id', 'route_id']],
        on='trip_id',
        how='left' # Keep all stop_times, even if trip info is missing (warned earlier)
    )
    stop_times_processed['route_id'].fillna('UNKNOWN_ROUTE', inplace=True)
    # Sort for sequential processing
    stop_times_processed.sort_values(['trip_id', 'stop_sequence'], inplace=True)

    # Calculate scheduled travel time if requested
    edge_weight_attr = 'travel_time'
    if use_travel_time:
        logging.info("Calculating scheduled travel times for edge weights")
        stop_times_processed['arrival_seconds'] = stop_times_processed['arrival_time'].apply(parse_time)
        stop_times_processed['departure_seconds'] = stop_times_processed['departure_time'].apply(parse_time)

        # Shift to get previous stop's details within each trip
        grouped = stop_times_processed.groupby('trip_id')
        stop_times_processed['prev_stop_id'] = grouped['stop_id'].shift(1)
        stop_times_processed['prev_departure_seconds'] = grouped['departure_seconds'].shift(1)

        # Calculate time difference only where both prev_departure and current_arrival are valid
        valid_time_calc = stop_times_processed['prev_departure_seconds'].notna() & \
                          stop_times_processed['arrival_seconds'].notna() & \
                          stop_times_processed['prev_stop_id'].notna()

        # Apply calculation row-wise for valid entries
        calculated_times = stop_times_processed.loc[valid_time_calc].apply(
            lambda row: row['arrival_seconds'] - row['prev_departure_seconds'], axis=1
        )

        # Ensure non-negative travel time, apply a minimum if needed 
        min_travel_time = 10.0
        stop_times_processed.loc[valid_time_calc, edge_weight_attr] = calculated_times.apply(
            lambda x: max(x, min_travel_time) if pd.notna(x) and x >= 0 else min_travel_time
        )
        # Fill NaNs (first stops in trips, or parse errors)
        logging.info("Finished calculating scheduled travel times.")
    else:
        # Use uniform weight if not using travel time
        edge_weight_attr = 'uniform_weight'
        stop_times_processed[edge_weight_attr] = 1.0
        stop_times_processed['prev_stop_id'] = stop_times_processed.groupby('trip_id')['stop_id'].shift(1)


    # 3. Add Trip Sequence Edges
    logging.info(f"Adding trip sequence edges to the graph (weight: {edge_weight_attr})")
    edge_count = 0
    skipped_edges = 0
    # Iterate through rows representing a segment *from* prev_stop_id *to* stop_id
    edge_creation_data = stop_times_processed.dropna(subset=['prev_stop_id', edge_weight_attr])

    for _, row in edge_creation_data.iterrows():
        u_node, v_node = row['prev_stop_id'], row['stop_id']
        route_id = row['route_id']
        edge_weight = row[edge_weight_attr]

        # Basic validation
        if not G.has_node(u_node) or not G.has_node(v_node):
             skipped_edges += 1
             continue # Should not happen if stop filtering was correct
        if pd.isna(edge_weight) or edge_weight <= 0:
            skipped_edges += 1
            continue # Skip edges with invalid weights

        # Add or update edge
        if G.has_edge(u_node, v_node):
            # Edge exists, update attributes
            G[u_node][v_node]['routes'].add(route_id)
            # If multiple trips use the same segment, take minimum time
            if use_travel_time:
                G[u_node][v_node][edge_weight_attr] = min(G[u_node][v_node].get(edge_weight_attr, float('inf')), edge_weight)
        else:
            # Add new edge
            G.add_edge(u_node, v_node, **{edge_weight_attr: edge_weight}, routes={route_id})
            edge_count += 1

    logging.info(f"Added {edge_count:,} unique directed edges from trip sequences.")
    if skipped_edges > 0:
         logging.warning(f"Skipped {skipped_edges:,} potential edges due to missing nodes or invalid weights.")


    # 4. Add Intra-station Transfer Edges (if parent_station is available)
    logging.info("Adding intra-station transfer edges (based on parent_station)")
    stops_with_parents = stops_df.dropna(subset=['parent_station'])
    stops_with_parents = stops_with_parents[stops_with_parents['parent_station'] != ''] # Ensure not empty string
    intra_station_edge_count = 0

    if not stops_with_parents.empty:
        # Group stops by their parent station
        grouped_by_parent = stops_with_parents.groupby('parent_station')
        for parent, group in grouped_by_parent:
            child_stops = group['stop_id'].tolist()
            # Create bidirectional edges between all child stops within the same parent station
            # Assign a weight of 0 or a small transfer time/penalty if needed
            transfer_weight = 0.0 # Assume zero time transfer within station for shortest path
            for i in range(len(child_stops)):
                for j in range(i + 1, len(child_stops)):
                    u, v = child_stops[i], child_stops[j]
                    # Ensure nodes exist in the graph (if not, check stops_valid for error)
                    if G.has_node(u) and G.has_node(v):
                        # Add edges in both directions if they don't exist
                        if not G.has_edge(u, v):
                            G.add_edge(u, v, **{edge_weight_attr: transfer_weight}, routes={'intra_station_transfer'})
                            intra_station_edge_count += 1
                        if not G.has_edge(v, u):
                            G.add_edge(v, u, **{edge_weight_attr: transfer_weight}, routes={'intra_station_transfer'})
                            intra_station_edge_count += 1
        logging.info(f"Added {intra_station_edge_count:,} bidirectional intra-station transfer edges.")
    else:
        logging.info("No parent_station information found or used for intra-station transfers.")

    total_time = time.time() - start_time
    logging.info(f"Graph build complete. Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,} (took {total_time:.2f}s)")
    return G


# --- Analysis Functions ---

# (1a) Connectivity Measures
def calculate_connectivity(G: nx.DiGraph) -> Dict[str, float]:
    """
    Calculates basic graph connectivity metrics: Beta, Gamma, and Density.

    - Beta Index: Edges per Node (E/N). Indicates average degree.
    - Gamma Index: Ratio of actual edges to max possible edges (E / N*(N-1)).
                   Measures network completeness.
    - Density: Same as Gamma for directed graphs without loops.

    Args:
        G: The NetworkX DiGraph representing the transit network.

    Returns:
        A dictionary containing the calculated 'beta', 'gamma', and 'density' indices.
        Returns zeros if the graph is empty.
    """
    logging.info("Calculating connectivity metrics (Beta, Gamma, Density)")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        logging.warning("Graph is empty. Connectivity metrics will be zero.")
        return {'beta': 0.0, 'gamma': 0.0, 'density': 0.0}

    # Beta Index (Edges / Nodes)
    beta_index = num_edges / num_nodes if num_nodes > 0 else 0.0

    # Gamma Index & Density (Edges / Max Possible Edges)
    # Max possible edges in a directed graph with N nodes is N * (N - 1)
    max_possible_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 0
    gamma_index = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    density = gamma_index 

    results = {'beta': beta_index, 'gamma': gamma_index, 'density': density}
    logging.info(f"Connectivity Results: Beta={beta_index:.4f}, Gamma={gamma_index:.4f}, Density={density:.4f}")
    return results

# (1b) Centrality Measures
def calculate_centrality(G: nx.DiGraph, weight: Optional[str] = 'travel_time') -> Dict[str, pd.DataFrame]:
    """
    Calculates Degree, Closeness, and Betweenness centrality for nodes.

    - Degree Centrality: Fraction of nodes connected to a given node.
    - Closeness Centrality: Inverse of the average shortest path distance to all
                           other reachable nodes. Calculated on the largest
                           Strongly Connected Component (SCC) if the graph
                           is not strongly connected.
    - Betweenness Centrality: Fraction of all-pairs shortest paths that pass
                             through a given node. Approximated using a sample
                             of nodes (k) if the graph is large.

    Args:
        G: The NetworkX DiGraph.
        weight: The edge attribute to use as distance for Closeness and
                Betweenness centrality (ex. 'travel_time'). If None,
                hop count (unweighted) is used.

    Returns:
        A dictionary where keys are centrality measure names (ex.
        'degree_centrality') and values are DataFrames containing 'stop_id'
        and the corresponding centrality score, sorted descending. Returns
        an empty dictionary if the graph is empty.
    """
    logging.info(f"Calculating centrality metrics (weight: {weight})")
    centrality_results: Dict[str, pd.DataFrame] = {}
    start_time = time.time()

    if G.number_of_nodes() == 0:
        logging.warning("Graph is empty. Skipping centrality calculations.")
        return {}

    # --- Degree Centrality (In-Degree, Out-Degree, Total Degree) ---
    logging.info("Calculating Degree Centrality")
    try:
        # Raw degrees:
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        total_degree = dict(G.degree())

        # Store raw degrees for potential more direct interpretation
        centrality_results['in_degree'] = pd.DataFrame(in_degree.items(), columns=['stop_id', 'in_degree']).sort_values('in_degree', ascending=False)
        centrality_results['out_degree'] = pd.DataFrame(out_degree.items(), columns=['stop_id', 'out_degree']).sort_values('out_degree', ascending=False)
        centrality_results['total_degree'] = pd.DataFrame(total_degree.items(), columns=['stop_id', 'total_degree']).sort_values('total_degree', ascending=False)

        logging.info(f"Degree Centrality calculated ({time.time() - start_time:.2f}s).")
    except Exception as e:
        logging.error(f"Degree Centrality calculation failed: {e}", exc_info=True)

    current_time = time.time()

    # --- Closeness Centrality ---
    logging.info("Calculating Closeness Centrality")
    closeness_values: Dict[str, float] = {}
    try:
        # Closeness centrality is typically defined for connected components.
        # Calculate on the largest SCC if the graph is not strongly connected.
        if not nx.is_strongly_connected(G):
            logging.warning("Graph is not strongly connected. Calculating Closeness Centrality on the largest SCC.")
            sccs = list(nx.strongly_connected_components(G))
            if sccs:
                largest_scc_nodes = max(sccs, key=len)
                num_scc_nodes = len(largest_scc_nodes)
                logging.info(f"Largest SCC contains {num_scc_nodes:,} nodes (out of {G.number_of_nodes():,}).")
                if num_scc_nodes > 1:
                    subgraph_scc = G.subgraph(largest_scc_nodes).copy() # Use copy for safety
                    closeness_values_scc = nx.closeness_centrality(subgraph_scc, distance=weight)
                    # Assign 0 to nodes not in the largest SCC
                    closeness_values = {node: closeness_values_scc.get(node, 0.0) for node in G.nodes()}
                else:
                    logging.warning("Largest SCC has only 1 node. Closeness centrality is undefined (setting to 0).")
                    closeness_values = {node: 0.0 for node in G.nodes()}
            else:
                logging.warning("Graph has no strongly connected components. Closeness centrality cannot be calculated (setting to 0).")
                closeness_values = {node: 0.0 for node in G.nodes()}
        else:
            # Graph is strongly connected, calculate for all nodes
            logging.info("Graph is strongly connected. Calculating Closeness Centrality for all nodes.")
            closeness_values = nx.closeness_centrality(G, distance=weight)

        if closeness_values:
            centrality_results['closeness_centrality'] = pd.DataFrame(closeness_values.items(), columns=['stop_id', 'closeness_centrality']).sort_values('closeness_centrality', ascending=False)
        logging.info(f"Closeness Centrality calculated ({time.time() - current_time:.2f}s).")

    except Exception as e:
        logging.error(f"Closeness Centrality calculation failed: {e}", exc_info=True)

    current_time = time.time()

    # --- Betweenness Centrality ---
    logging.info("Calculating Betweenness Centrality")
    betweenness_values: Dict[str, float] = {}
    try:
        k_sample = None
        n_nodes = G.number_of_nodes()
        # Use approximation for large graphs to manage computation time
        if n_nodes > CPU_INTENSIVE_THRESHOLD:
            # Sample size: e.g., 10% of nodes or a fixed number like 1000
            k_sample = min(max(100, n_nodes // 10), 1000) # Ensure k >=100 if possible
            logging.warning(f"Graph is large ({n_nodes:,} nodes). Approximating Betweenness Centrality using k={k_sample} sample nodes.")
        elif n_nodes > 1:
            logging.info("Calculating exact Betweenness Centrality.")
        else:
            logging.warning("Graph has <= 1 node. Skipping Betweenness Centrality.")

        if k_sample or (n_nodes > 1 and not k_sample):
            betweenness_values = nx.betweenness_centrality(
                G,
                k=k_sample,          # Number of nodes for approximation (None for exact)
                weight=weight,       # Use specified edge weight for shortest paths
                normalized=True      # Normalize by (N-1)(N-2) for directed graphs
            )
        else:
             betweenness_values = {node: 0.0 for node in G.nodes()} # Default for small/empty graph

        if betweenness_values:
             centrality_results['betweenness_centrality'] = pd.DataFrame(betweenness_values.items(), columns=['stop_id', 'betweenness_centrality']).sort_values('betweenness_centrality', ascending=False)
        logging.info(f"Betweenness Centrality calculated ({time.time() - current_time:.2f}s).")

    except Exception as e:
        logging.error(f"Betweenness Centrality calculation failed: {e}", exc_info=True)

    logging.info(f"Total centrality calculation time: {time.time() - start_time:.2f} seconds.")
    return centrality_results


# (1c) Accessibility Measures
def calculate_accessibility(
    G: nx.DiGraph,
    target_stop_ids: List[str],
    weight: str = 'travel_time',
    time_thresholds_minutes: List[int] = ACCESSIBILITY_THRESHOLDS_MIN
    ) -> pd.DataFrame:
    """
    Calculates cumulative accessibility from each stop to a set of target stops.

    Measures the number of unique target stops reachable from each origin stop
    within specified travel time thresholds using Dijkstra's algorithm.

    Args:
        G: The NetworkX DiGraph.
        target_stop_ids: A list of stop_ids considered as targets/destinations
                         ex. LRT/MRT stations, major hubs).
        weight: The edge attribute representing travel cost (ex. 'travel_time').
        time_thresholds_minutes: A list of time thresholds in minutes (ex. [15, 30, 60]).

    Returns:
        A DataFrame with columns 'stop_id' and 'reachable_targets_Xmin' for each
        threshold X, showing the count of reachable targets within that time.
        Returns an empty DataFrame if the graph is empty or no targets are specified.
    """
    if not target_stop_ids:
        logging.warning("No target stops provided for accessibility analysis. Returning empty DataFrame.")
        return pd.DataFrame(columns=['stop_id'] + [f'reachable_targets_{t}min' for t in time_thresholds_minutes])
    if G.number_of_nodes() == 0:
        logging.warning("Graph is empty. Skipping accessibility analysis.")
        return pd.DataFrame(columns=['stop_id'] + [f'reachable_targets_{t}min' for t in time_thresholds_minutes])

    time_thresholds_minutes = sorted(list(set(time_thresholds_minutes))) # Ensure unique and sorted
    time_thresholds_seconds = [t * 60 for t in time_thresholds_minutes]
    max_threshold_seconds = max(time_thresholds_seconds) if time_thresholds_seconds else 0
    target_set = set(target_stop_ids) # Faster lookups

    logging.info(f"Calculating accessibility to {len(target_set):,} target stops within thresholds: {time_thresholds_minutes} min.")
    start_time = time.time()

    accessibility_data = []
    nodes_processed = 0
    total_nodes = G.number_of_nodes()

    for origin_node in G.nodes():
        node_results = {'stop_id': origin_node}
        # Initialize counts for each threshold to 0
        for t_min in time_thresholds_minutes:
            node_results[f'reachable_targets_{t_min}min'] = 0

        reachable_targets_within_max_time: Dict[str, float] = {}
        try:
            # Calculate shortest path lengths from the origin node up to the max cutoff time
            lengths = nx.single_source_dijkstra_path_length(
                G,
                source=origin_node,
                cutoff=max_threshold_seconds,
                weight=weight
            )

            # Filter lengths to include only target stops and store their travel times
            for reachable_node, travel_time in lengths.items():
                if reachable_node in target_set:
                    reachable_targets_within_max_time[reachable_node] = travel_time

            # Count how many unique targets fall within each specific threshold
            # Sort thresholds to count cumulatively is not necessary here - count per band
            reachable_target_times = list(reachable_targets_within_max_time.values())
            for i, threshold_sec in enumerate(time_thresholds_seconds):
                 count = sum(1 for time_val in reachable_target_times if time_val <= threshold_sec)
                 node_results[f'reachable_targets_{time_thresholds_minutes[i]}min'] = count

        except nx.NetworkXNoPath:
            # If the origin itself is isolated or cannot reach any node within cutoff
             pass # Results remain 0, which is correct
        except Exception as e:
            logging.error(f"Error calculating accessibility from node {origin_node}: {e}", exc_info=False) # Avoid overly verbose logs

        accessibility_data.append(node_results)
        nodes_processed += 1
        if nodes_processed % 500 == 0 or nodes_processed == total_nodes:
            elapsed = time.time() - start_time
            logging.info(f"Processed accessibility for {nodes_processed:,}/{total_nodes:,} nodes ({elapsed:.2f}s)")

    logging.info(f"Accessibility calculation finished ({time.time() - start_time:.2f}s).")

    # Create DataFrame
    df = pd.DataFrame(accessibility_data)
    # Ensure all expected columns are present, even if calculation failed for some nodes
    col_order = ['stop_id'] + [f'reachable_targets_{t}min' for t in time_thresholds_minutes]
    df = df.reindex(columns=col_order, fill_value=0) # Fill missing columns/values with 0
    return df


# (2) Route Overlap Analysis
def get_route_stop_sequences(stop_times_df: pd.DataFrame, trips_df: pd.DataFrame) -> Dict[str, Tuple[str, ...]]:
    """
    Reconstructs a representative stop sequence (tuple of stop_ids) for each route_id.

    It typically uses the most frequent sequence of stops among all trips associated
    with a route. Filters out trips with fewer than 2 stops.

    Args:
        stop_times_df: DataFrame of stop_times (ideally filtered).
        trips_df: DataFrame of trips.

    Returns:
        A dictionary mapping route_id (str) to its representative stop sequence
        (tuple of stop_ids). Returns an empty dictionary if data is insufficient.
    """
    logging.info("Reconstructing representative route stop sequences")
    if 'trip_id' not in stop_times_df or 'stop_id' not in stop_times_df or 'stop_sequence' not in stop_times_df:
        logging.error("stop_times_df missing required columns for sequence reconstruction.")
        return {}
    if 'trip_id' not in trips_df or 'route_id' not in trips_df:
        logging.error("trips_df missing required columns for sequence reconstruction.")
        return {}

    # Merge route_id onto stop_times
    st_merged = pd.merge(
        stop_times_df[['trip_id', 'stop_id', 'stop_sequence']],
        trips_df[['trip_id', 'route_id']],
        on='trip_id',
        how='inner' # Only consider stop_times belonging to known trips
    )
    st_merged['route_id'].fillna('UNKNOWN_ROUTE', inplace=True) 

    # Sort by trip and sequence
    st_merged.sort_values(['trip_id', 'stop_sequence'], inplace=True)

    # Filter out trips with fewer than 2 stops as they don't form a sequence
    trip_lengths = st_merged.groupby('trip_id').size()
    valid_trips = trip_lengths[trip_lengths >= 2].index
    st_valid_trips = st_merged[st_merged['trip_id'].isin(valid_trips)]

    if st_valid_trips.empty:
        logging.warning("No valid trips (>= 2 stops) found after merging. Cannot reconstruct sequences.")
        return {}

    # Group by trip and aggregate stop_ids into tuples
    trip_stop_tuples = st_valid_trips.groupby('trip_id')['stop_id'].apply(tuple)
    trip_stop_tuples = trip_stop_tuples.reset_index()

    # Merge route_id back onto the trip sequences
    trip_sequences = pd.merge(trip_stop_tuples, trips_df[['trip_id', 'route_id']], on='trip_id')

    # Find the mode (most frequent) sequence for each route
    # If multiple sequences have the same highest frequency, mode() returns them all; we pick the first.
    try:
         route_modes = trip_sequences.groupby('route_id')['stop_id'].agg(lambda x: x.mode()[0] if not x.mode().empty else tuple())
    except Exception as e:
         logging.error(f"Error finding mode sequence per route: {e}. Trying value_counts approach.")
         # Fallback using value_counts (might differ slightly if multiple modes exist)
         try:
             route_modes = trip_sequences.groupby('route_id')['stop_id'].agg(lambda x: x.value_counts().idxmax() if not x.empty else tuple())
         except Exception as e2:
              logging.error(f"Fallback sequence reconstruction failed: {e2}")
              return {}


    # Filter out any empty sequences that might have resulted
    route_sequences_dict = {k: v for k, v in route_modes.to_dict().items() if v}

    logging.info(f"Reconstructed representative sequences for {len(route_sequences_dict):,} routes.")
    return route_sequences_dict


def calculate_route_overlap(
    route_sequences: Dict[str, Tuple[str, ...]],
    method: str = ROUTE_OVERLAP_METHOD,
    overlap_threshold: float = ROUTE_OVERLAP_THRESHOLD
    ) -> pd.DataFrame:
    """
    Calculates the overlap between pairs of route sequences based on shared stops.

    Supported methods:
    - 'jaccard': Intersection size / Union size
    - 'overlap_coefficient': Intersection size / Min(Size A, Size B)

    Args:
        route_sequences: Dictionary mapping route_id to its stop sequence tuple.
        method: The overlap metric to use ('jaccard' or 'overlap_coefficient').
        overlap_threshold: The minimum overlap value (0.0 to 1.0) required
                           to include the pair in the results.

    Returns:
        A DataFrame containing pairs of routes exceeding the overlap threshold,
        with details on the overlap metric, value, and stop counts. Returns an
        empty DataFrame if fewer than 2 routes exist or no pairs meet the threshold.
    """
    logging.info(f"Calculating route overlap (method: {method}, threshold: {overlap_threshold:.2f})")
    start_time = time.time()
    routes = list(route_sequences.keys())
    num_routes = len(routes)
    overlaps_data = []

    if num_routes < 2:
        logging.warning("Need at least 2 routes with sequences to calculate overlap.")
        return pd.DataFrame()

    # Use sets for efficient intersection/union calculation
    route_sets = {route_id: set(seq) for route_id, seq in route_sequences.items()}

    processed_pairs = 0
    total_pairs = num_routes * (num_routes - 1) // 2

    for i in range(num_routes):
        route_a_id = routes[i]
        set_a = route_sets[route_a_id]
        len_a = len(set_a)
        if len_a == 0: continue # Skip routes with empty sequences

        for j in range(i + 1, num_routes):
            route_b_id = routes[j]
            set_b = route_sets[route_b_id]
            len_b = len(set_b)
            if len_b == 0: continue

            # Calculate intersection
            intersection_set = set_a.intersection(set_b)
            intersection_size = len(intersection_set)

            if intersection_size == 0:
                continue # No overlap, skip calculation

            # Calculate overlap metric
            overlap_value = 0.0
            if method == 'jaccard':
                union_size = len(set_a.union(set_b))
                overlap_value = intersection_size / union_size if union_size > 0 else 0.0
            elif method == 'overlap_coefficient':
                min_len = min(len_a, len_b)
                overlap_value = intersection_size / min_len if min_len > 0 else 0.0
            else:
                logging.warning(f"Unknown overlap method '{method}'. Defaulting to Jaccard.")
                union_size = len(set_a.union(set_b))
                overlap_value = intersection_size / union_size if union_size > 0 else 0.0

            # Store if overlap meets threshold
            if overlap_value >= overlap_threshold:
                overlaps_data.append({
                    'route_a': route_a_id,
                    'route_b': route_b_id,
                    'overlap_metric': method,
                    'overlap_value': overlap_value,
                    'shared_stops': intersection_size,
                    'route_a_stops': len_a, # Number of unique stops in route A
                    'route_b_stops': len_b  # Number of unique stops in route B
                })
            processed_pairs += 1 # Count processed pairs for potential progress logging if needed

    logging.info(f"Overlap calculation done ({time.time() - start_time:.2f}s). Found {len(overlaps_data):,} overlapping pairs above threshold.")

    if not overlaps_data:
        return pd.DataFrame()
    else:
        return pd.DataFrame(overlaps_data).sort_values(by='overlap_value', ascending=False)


# (3a) Route Circuity
def calculate_route_circuity(
    G: nx.DiGraph,
    route_sequences: Dict[str, Tuple[str, ...]],
    weight: str = 'travel_time'
    ) -> pd.DataFrame:
    """
    Calculates the circuity index for each route based on its representative sequence.

    Circuity Index = (Actual Travel Time along sequence) / (Shortest Path Travel Time)
    A value close to 1 indicates a direct route, while higher values indicate detour.

    Args:
        G: The NetworkX DiGraph with edge weights.
        route_sequences: Dictionary mapping route_id to its stop sequence tuple.
        weight: The edge attribute representing travel cost (ex. 'travel_time').

    Returns:
        A DataFrame containing route_id, start/end stops, actual time, shortest time,
        and the calculated circuity index for routes where calculation is possible.
        Returns an empty DataFrame if no valid routes/paths are found.
    """
    logging.info(f"Calculating route circuity (using edge weight: {weight})")
    start_time = time.time()
    circuity_results = []

    if not route_sequences:
        logging.warning("No route sequences provided for circuity calculation.")
        return pd.DataFrame()
    if not G.edges(data=True):
         logging.warning("Graph has no edges. Cannot calculate circuity.")
         return pd.DataFrame()


    processed_routes = 0
    total_routes = len(route_sequences)

    for route_id, stop_sequence_tuple in route_sequences.items():
        stop_sequence = list(stop_sequence_tuple)

        # Need at least two stops for a path
        if len(stop_sequence) < 2:
            continue

        start_node, end_node = stop_sequence[0], stop_sequence[-1]

        # Check if start/end nodes exist in the graph
        if not G.has_node(start_node) or not G.has_node(end_node):
            logging.debug(f"Skipping route {route_id}: Start or end node not in graph.")
            continue

        # Calculate actual travel time along the sequence
        actual_time = 0.0
        valid_path_segments = True
        for i in range(len(stop_sequence) - 1):
            u, v = stop_sequence[i], stop_sequence[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data and weight in edge_data and pd.notna(edge_data[weight]):
                segment_time = edge_data[weight]
                if segment_time >= 0: # Ensure non-negative time
                     actual_time += segment_time
                else:
                     logging.debug(f"Skipping route {route_id}: Negative segment time ({u}->{v}).")
                     valid_path_segments = False; break
            else:
                # If a segment doesn't exist in the graph, the sequence is invalid w.r.t. the graph
                logging.debug(f"Skipping route {route_id}: Segment {u}->{v} not found or has no weight.")
                valid_path_segments = False
                break

        if not valid_path_segments or actual_time <= 0: # Also skip if total time is zero
            continue

        # Calculate shortest path time between start and end nodes
        shortest_time: Optional[float] = None
        try:
            shortest_time = nx.shortest_path_length(G, source=start_node, target=end_node, weight=weight)
        except nx.NetworkXNoPath:
            # If no path exists between start/end in the graph, circuity is undefined (or infinite)
            logging.debug(f"Skipping route {route_id}: No path found between {start_node} and {end_node} in graph.")
            continue
        except Exception as e:
            logging.error(f"Error finding shortest path for route {route_id} ({start_node} -> {end_node}): {e}", exc_info=False)
            continue

        # Calculate circuity index if shortest path exists and is positive
        if shortest_time is not None and shortest_time > 0:
            circuity_index = actual_time / shortest_time
            circuity_results.append({
                'route_id': route_id,
                'start_stop': start_node,
                'end_stop': end_node,
                'actual_time_seconds': actual_time,
                'shortest_path_time_seconds': shortest_time,
                'circuity_index': circuity_index
            })
        elif shortest_time == 0:
             logging.debug(f"Skipping route {route_id}: Shortest path time is zero.")
             pass


        processed_routes += 1
        if processed_routes % 500 == 0 or processed_routes == total_routes:
            logging.info(f"Processed circuity for {processed_routes:,}/{total_routes:,} routes")

    logging.info(f"Circuity calculation done ({time.time() - start_time:.2f}s). Found {len(circuity_results):,} routes with valid circuity.")

    if not circuity_results:
        return pd.DataFrame()
    else:
        return pd.DataFrame(circuity_results).sort_values(by='circuity_index', ascending=False)


# (3b) Transfer Analysis
def analyze_transfers(
    G: nx.DiGraph,
    od_pairs: List[Tuple[str, str]],
    weight: str = 'travel_time'
    ) -> pd.DataFrame:
    """
    Analyzes the number of transfers required for shortest paths between OD pairs.

    A transfer is counted when moving between consecutive edges in the shortest
    path if there is no common route_id serving both edges, excluding edges
    marked as 'intra_station_transfer'.

    Args:
        G: The NetworkX DiGraph with edge weights and 'routes' attribute (a set).
        od_pairs: A list of (origin_stop_id, destination_stop_id) tuples.
        weight: The edge attribute representing travel cost

    Returns:
        A DataFrame with origin, destination, travel time, number of transfers,
        and path length (number of stops) for each OD pair. 'num_transfers' is -1
        if no path exists, -99 on calculation error.
    """
    logging.info(f"Analyzing transfers for {len(od_pairs):,} OD pairs (weight: {weight})")
    start_time = time.time()
    transfer_results = []

    if not od_pairs:
        logging.warning("No OD pairs provided for transfer analysis.")
        return pd.DataFrame()
    if G.number_of_nodes() == 0:
         logging.warning("Graph is empty. Cannot analyze transfers.")
         return pd.DataFrame()

    processed_pairs = 0
    total_pairs = len(od_pairs)

    for origin_node, dest_node in od_pairs:
        # Validate nodes
        if not G.has_node(origin_node) or not G.has_node(dest_node):
            transfer_results.append({
                'origin': origin_node, 'destination': dest_node,
                'travel_time_seconds': float('inf'), 'num_transfers': -2, # Indicate invalid OD nodes
                'path_length_stops': 0
            })
            continue
        if origin_node == dest_node:
            transfer_results.append({
                'origin': origin_node, 'destination': dest_node,
                'travel_time_seconds': 0.0, 'num_transfers': 0,
                'path_length_stops': 1
            })
            continue


        num_transfers = 0
        total_time = float('inf')
        path_nodes = []

        try:
            # Find the shortest path (nodes and total time)
            total_time, path_nodes = nx.single_source_dijkstra(G, source=origin_node, target=dest_node, weight=weight)
            path_len = len(path_nodes)

            # Count transfers along the path
            if path_len > 1:
                # Get routes for the first segment
                first_edge_data = G.get_edge_data(path_nodes[0], path_nodes[1])
                # Use an empty set if 'routes' attribute is missing (shouldn't happen if built correctly)
                current_routes: Set[str] = first_edge_data.get('routes', set()).copy() if first_edge_data else set()

                # Iterate through the path segments (edges)
                for k in range(1, path_len - 1):
                    u, v = path_nodes[k], path_nodes[k+1]
                    edge_data = G.get_edge_data(u, v)
                    next_routes: Set[str] = edge_data.get('routes', set()) if edge_data else set()

                    # Skip transfer check if it's an intra-station link (assumes seamless transfer)
                    is_intra_station = 'intra_station_transfer' in next_routes
                    if is_intra_station:
                        # If the next segment is intra-station, we don't count a transfer yet.
                        continue # Don't count transfer at the intra-station link

                    # Find common routes between the current segment and the next
                    common_routes = current_routes.intersection(next_routes)

                    # If no common routes, a transfer is required
                    if not common_routes:
                        num_transfers += 1
                        # Update current_routes to the routes of the new segment
                        current_routes = next_routes.copy()
                    else:
                        # Stayed on a common route, update current_routes to the intersection
                        current_routes = common_routes.copy()

            transfer_results.append({
                'origin': origin_node, 'destination': dest_node,
                'travel_time_seconds': total_time,
                'num_transfers': num_transfers,
                'path_length_stops': path_len
            })

        except nx.NetworkXNoPath:
            transfer_results.append({
                'origin': origin_node, 'destination': dest_node,
                'travel_time_seconds': float('inf'), 'num_transfers': -1, # -1 indicates no path
                'path_length_stops': 0
            })
        except Exception as e:
            logging.error(f"Error during transfer analysis for OD ({origin_node} -> {dest_node}): {e}", exc_info=False)
            transfer_results.append({
                'origin': origin_node, 'destination': dest_node,
                'travel_time_seconds': float('inf'), 'num_transfers': -99, # -99 indicates calculation error
                'path_length_stops': 0
            })

        processed_pairs += 1
        if processed_pairs % 500 == 0 or processed_pairs == total_pairs:
            logging.info(f"Processed {processed_pairs:,}/{total_pairs:,} OD pairs for transfers ({time.time() - start_time:.2f}s)")

    logging.info(f"Transfer analysis done ({time.time() - start_time:.2f}s).")
    return pd.DataFrame(transfer_results)


# (4) Network Resilience Analysis

def _calculate_average_metric(
    G: nx.DiGraph,
    metric_func: callable, # e.g., nx.average_shortest_path_length
    weight: Optional[str] = None
    ) -> float:
    """
    Helper to calculate an average graph metric, handling disconnected graphs.

    Currently specialized for Average Shortest Path Length (ASPL), calculating
    it on the largest Strongly Connected Component (SCC) if the graph is not
    strongly connected. Returns float('inf') if ASPL cannot be calculated.

    Args:
        G: The NetworkX DiGraph.
        metric_func: The NetworkX function to calculate (expects G and weight).
        weight: Edge weight attribute for the metric calculation.

    Returns:
        The calculated average metric value, or float('inf') if undefined/error.
    """
    if G.number_of_nodes() == 0:
        return float('inf') # Metric undefined for empty graph

    # --- Special handling for Average Shortest Path Length (ASPL) ---
    if metric_func == nx.average_shortest_path_length:
        if not nx.is_strongly_connected(G):
            sccs = list(nx.strongly_connected_components(G))
            if not sccs:
                # No SCCs means graph is empty or only has isolated nodes/small components
                return float('inf') # ASPL undefined

            largest_scc_nodes = max(sccs, key=len)
            num_scc_nodes = len(largest_scc_nodes)

            if num_scc_nodes <= 1:
                # ASPL is undefined for graphs with 0 or 1 node
                return float('inf')

            subgraph_scc = G.subgraph(largest_scc_nodes)
            # Check if the subgraph itself is connected. 
            # Recalculating ASPL only on the largest SCC
            try:
                return nx.average_shortest_path_length(subgraph_scc, weight=weight)
            except nx.NetworkXError as e:
                 logging.warning(f"Could not calculate ASPL on largest SCC ({num_scc_nodes} nodes): {e}")
                 return float('inf')
            except Exception as e:
                 logging.error(f"Unexpected error calculating ASPL on SCC: {e}")
                 return float('inf')

        else:
            if G.number_of_nodes() <= 1:
                return float('inf') 
            try:
                return nx.average_shortest_path_length(G, weight=weight)
            except nx.NetworkXError as e: # e.g. graph not connected error, shouldn't happen here
                 logging.warning(f"Could not calculate ASPL on strongly connected graph: {e}")
                 return float('inf')
            except Exception as e:
                 logging.error(f"Unexpected error calculating ASPL: {e}")
                 return float('inf')

    # --- Placeholder for other average metrics ---
    else:
        logging.warning(f"Calculation logic for metric '{metric_func.__name__}' not fully implemented in _calculate_average_metric helper. Returning 0.0.")
        return 0.0


def _calculate_average_accessibility(
    accessibility_df: Optional[pd.DataFrame],
    column_name: str
    ) -> float:
    """
    Helper to calculate the average value of a specific accessibility column.

    Args:
        accessibility_df: DataFrame output from calculate_accessibility.
        column_name: The specific column to average (ex. 'reachable_targets_60min').

    Returns:
        The average accessibility value, or 0.0 if data is missing or empty.
    """
    if accessibility_df is None or column_name not in accessibility_df.columns or accessibility_df.empty:
        return 0.0
    try:
        # Calculate mean, ignoring potential NaNs if any occurred during calculation
        avg_value = accessibility_df[column_name].mean(skipna=True)
        return float(avg_value) if pd.notna(avg_value) else 0.0
    except Exception as e:
        logging.error(f"Error calculating average for accessibility column '{column_name}': {e}")
        return 0.0


def analyze_resilience_enhanced(
    G_original: nx.DiGraph,
    metric_to_remove_by: str,  # 'degree', 'betweenness', or potentially 'closeness'
    top_n_to_remove: int,
    centrality_dfs: Dict[str, pd.DataFrame], # Pre-calculated centrality results
    target_stops: List[str],
    weight: Optional[str] = 'travel_time',
    accessibility_metric_col: str = RESILIENCE_ACCESSIBILITY_METRIC
    ) -> Dict[str, Any]:
    """
    Performs resilience analysis by simulating node removal and measuring impact.

    Removes the top N nodes based on a specified centrality metric and measures
    the change in:
    1.  Size of the largest Strongly Connected Component (SCC).
    2.  Average Shortest Path Length (ASPL) within the largest SCC.
    3.  Average accessibility (mean of a specified `accessibility_metric_col`).

    Args:
        G_original: The original NetworkX DiGraph.
        metric_to_remove_by: Centrality metric for ranking nodes ('total_degree',
                             'betweenness_centrality', 'closeness_centrality').
        top_n_to_remove: Number of top nodes to remove.
        centrality_dfs: Dictionary containing pre-calculated centrality DataFrames.
        target_stops: List of target stop IDs for recalculating accessibility.
        weight: Edge weight attribute for ASPL and accessibility calculations.
        accessibility_metric_col: Column from accessibility results to average.

    Returns:
        A dictionary summarizing the resilience scenario and the calculated
        impact metrics (initial values, final values, percentage changes).
        Includes an 'error' key if the analysis could not be performed.
    """
    scenario_name = f"remove_top_{top_n_to_remove}_{metric_to_remove_by}"
    logging.info(f"\n--- Resilience Analysis Scenario: {scenario_name} ---")
    results: Dict[str, Any] = {"scenario": scenario_name}

    if G_original.number_of_nodes() == 0:
        logging.warning(f"[{scenario_name}] Original graph is empty. Cannot analyze resilience.")
        results["error"] = "Empty original graph"
        return results

    # Ensure the specified accessibility column exists based on thresholds
    if accessibility_metric_col not in [f'reachable_targets_{t}min' for t in ACCESSIBILITY_THRESHOLDS_MIN]:
         logging.error(f"[{scenario_name}] Invalid accessibility_metric_col '{accessibility_metric_col}'. Check ACCESSIBILITY_THRESHOLDS_MIN.")
         results["error"] = f"Invalid accessibility metric column '{accessibility_metric_col}'"
         return results
    acc_time_threshold_min = int(accessibility_metric_col.split('_')[-1].replace('min',''))


    # --- 1. Calculate Initial Network Metrics ---
    logging.info(f"[{scenario_name}] Calculating initial network metrics")
    # Initial SCC Size
    initial_components = list(nx.strongly_connected_components(G_original))
    initial_largest_scc_size = len(max(initial_components, key=len)) if initial_components else 0
    results["initial_largest_scc_size"] = initial_largest_scc_size
    logging.info(f"[{scenario_name}] Initial Largest SCC Size: {initial_largest_scc_size:,}")

    # Initial Average Shortest Path Length (within largest SCC)
    initial_aspl = _calculate_average_metric(G_original, nx.average_shortest_path_length, weight=weight)
    results["initial_aspl_scc"] = initial_aspl if np.isfinite(initial_aspl) else None # Use None for easier JSON/CSV later
    logging.info(f"[{scenario_name}] Initial Average Shortest Path Length (SCC): {format_seconds(initial_aspl)}")

    # Initial Average Accessibility (Recalculate for baseline consistency)
    logging.info(f"[{scenario_name}] Calculating initial average accessibility ({accessibility_metric_col})")
    # Only need the relevant time threshold for this specific calculation
    initial_accessibility_df = calculate_accessibility(G_original, target_stops, weight=weight, time_thresholds_minutes=[acc_time_threshold_min])
    initial_avg_accessibility = _calculate_average_accessibility(initial_accessibility_df, accessibility_metric_col)
    results["initial_avg_accessibility"] = initial_avg_accessibility
    logging.info(f"[{scenario_name}] Initial Average Accessibility ({accessibility_metric_col}): {initial_avg_accessibility:.3f}")


    # --- 2. Identify Nodes to Remove ---
    # Use the appropriate centrality DataFrame based on metric_to_remove_by
    centrality_key = metric_to_remove_by # Map keys if needed, e.g. 'degree' -> 'total_degree'
    if metric_to_remove_by == 'degree': centrality_key = 'total_degree' # Assuming total degree is preferred

    nodes_to_remove: List[str] = []
    if centrality_key in centrality_dfs and not centrality_dfs[centrality_key].empty:
        nodes_to_remove = centrality_dfs[centrality_key].head(top_n_to_remove)['stop_id'].tolist()
        # Ensure we don't try to remove more nodes than available
        nodes_to_remove = nodes_to_remove[:min(len(nodes_to_remove), G_original.number_of_nodes())]
        logging.info(f"[{scenario_name}] Identified Top {len(nodes_to_remove)} nodes by {centrality_key} to remove: {nodes_to_remove}")
        results["nodes_identified_for_removal"] = nodes_to_remove
    else:
        logging.error(f"[{scenario_name}] Required centrality data '{centrality_key}' not found or empty. Cannot proceed.")
        results["error"] = f"Missing centrality data for '{centrality_key}'"
        return results

    # --- 3. Simulate Disruption (Remove Nodes) ---
    G_resilience = G_original.copy()
    # Ensure nodes actually exist before trying to remove
    valid_nodes_to_remove = [n for n in nodes_to_remove if G_resilience.has_node(n)]
    if not valid_nodes_to_remove:
        logging.warning(f"[{scenario_name}] None of the identified nodes exist in the graph. No changes made.")
        results["removed_nodes"] = []
        # Assign final metrics same as initial
        results["final_largest_scc_size"] = initial_largest_scc_size
        results["final_aspl_scc"] = results["initial_aspl_scc"]
        results["final_avg_accessibility"] = initial_avg_accessibility
    else:
        G_resilience.remove_nodes_from(valid_nodes_to_remove)
        logging.info(f"[{scenario_name}] Removed {len(valid_nodes_to_remove)} nodes from the graph.")
        results["removed_nodes"] = valid_nodes_to_remove

        # --- 4. Calculate Final Metrics After Removal ---
        logging.info(f"[{scenario_name}] Calculating final network metrics after removal")
        # Final SCC Size
        final_components = list(nx.strongly_connected_components(G_resilience))
        final_largest_scc_size = len(max(final_components, key=len)) if final_components else 0
        results["final_largest_scc_size"] = final_largest_scc_size
        logging.info(f"[{scenario_name}] Final Largest SCC Size: {final_largest_scc_size:,}")

        # Final Average Shortest Path Length (within largest SCC)
        final_aspl = _calculate_average_metric(G_resilience, nx.average_shortest_path_length, weight=weight)
        results["final_aspl_scc"] = final_aspl if np.isfinite(final_aspl) else None
        logging.info(f"[{scenario_name}] Final Average Shortest Path Length (SCC): {format_seconds(final_aspl)}")

        # Final Average Accessibility
        logging.info(f"[{scenario_name}] Calculating final average accessibility ({accessibility_metric_col})")
        final_accessibility_df = calculate_accessibility(G_resilience, target_stops, weight=weight, time_thresholds_minutes=[acc_time_threshold_min])
        final_avg_accessibility = _calculate_average_accessibility(final_accessibility_df, accessibility_metric_col)
        results["final_avg_accessibility"] = final_avg_accessibility
        logging.info(f"[{scenario_name}] Final Average Accessibility ({accessibility_metric_col}): {final_avg_accessibility:.3f}")


    # --- 5. Calculate Percentage Changes ---
    init_scc = results["initial_largest_scc_size"]
    final_scc = results["final_largest_scc_size"]
    scc_change_pct = ((final_scc - init_scc) / init_scc * 100) if init_scc > 0 else 0.0
    results["scc_change_pct"] = scc_change_pct

    init_aspl = results["initial_aspl_scc"]
    final_aspl = results["final_aspl_scc"]
    aspl_change_pct = None
    if init_aspl is not None and final_aspl is not None:
         if init_aspl > 0:
             aspl_change_pct = ((final_aspl - init_aspl) / init_aspl) * 100
         elif final_aspl > 0: # Initial was 0 or invalid, final is positive -> Infinite increase
             aspl_change_pct = float('inf')
         else: # Both 0 or invalid
             aspl_change_pct = 0.0
    elif final_aspl is not None and init_aspl is None: # Went from undefined to defined
        aspl_change_pct = float('-inf') # Or some indicator of improvement from undefined
    elif init_aspl is not None and final_aspl is None: # Went from defined to undefined
        aspl_change_pct = float('inf') # Infinite increase / became undefined
    results["aspl_change_pct"] = aspl_change_pct

    init_acc = results["initial_avg_accessibility"]
    final_acc = results["final_avg_accessibility"]
    acc_change_pct = ((final_acc - init_acc) / init_acc * 100) if init_acc > 0 else 0.0
    results["accessibility_change_pct"] = acc_change_pct

    logging.info(f"[{scenario_name}] Resilience Impact Summary:")
    logging.info(f"  > SCC Size Change: {scc_change_pct:+.2f}% ({init_scc:,} -> {final_scc:,})")
    logging.info(f"  > ASPL (SCC) Change: {aspl_change_pct:+.2f}% (Initial: {format_seconds(init_aspl)}, Final: {format_seconds(final_aspl)})" if aspl_change_pct is not None and np.isfinite(aspl_change_pct) else f"  > ASPL (SCC) Change: N/A (Initial: {format_seconds(init_aspl)}, Final: {format_seconds(final_aspl)})")
    logging.info(f"  > Avg Accessibility Change: {acc_change_pct:+.2f}% ({init_acc:.3f} -> {final_acc:.3f})")

    return results


# (5) Google API Integration
def get_google_directions_route(
    origin_coords: Union[str, Tuple[float, float]],
    destination_coords: Union[str, Tuple[float, float]],
    mode: str = 'driving', # 'driving', 'transit', 'walking', 'bicycling'
    api_key: Optional[str] = None,
    traffic: bool = True,
    departure_time: Union[str, datetime] = 'now'
    ) -> Optional[Dict[str, Any]]:
    """
    Fetches route information (duration, distance) from Google Directions API.

    NOTE: Requires the 'googlemaps' library and a valid API key with the
          Directions API enabled. Ensure the API key is handled securely.

    Args:
        origin_coords: Origin coordinates as "lat,lng" string or (lat, lng) tuple.
        destination_coords: Destination coordinates as "lat,lng" string or (lat, lng) tuple.
        mode: Travel mode ('driving', 'transit', 'walking', 'bicycling').
        api_key: Your Google Cloud Platform API key.
        traffic: If True and mode is 'driving', request traffic-aware duration.
        departure_time: Departure time ('now' or a datetime object).

    Returns:
        A dictionary with 'duration_seconds', 'distance_meters', and 'raw_result',
        or None if the API call fails, no route is found, or the library/key
        is missing.
    """
    if not googlemaps:
        logging.warning("Google Maps library not installed ('pip install googlemaps'). Cannot query API.")
        return None
    if not api_key:
        logging.warning("Google API key not provided. Cannot query API.")
        return None

    gmaps = googlemaps.Client(key=api_key)
    dep_time = datetime.now() if departure_time == 'now' else departure_time
    # Traffic model only relevant for driving
    traffic_model_param = "best_guess" if mode == 'driving' and traffic else None

    logging.debug(f"Querying Google Directions API: {origin_coords} -> {destination_coords}, Mode: {mode}, Traffic: {traffic}")

    try:
        directions_result = gmaps.directions(
            origin=origin_coords,
            destination=destination_coords,
            mode=mode,
            departure_time=dep_time,
            traffic_model=traffic_model_param
        )

        # Process the result
        if directions_result and isinstance(directions_result, list) and directions_result[0].get('legs'):
            route_info = directions_result[0]['legs'][0] # Get the first leg of the first route

            # Get duration (potentially with traffic) and distance
            duration_data = route_info.get('duration_in_traffic') if mode == 'driving' and traffic else route_info.get('duration')
            duration_sec = duration_data.get('value') if duration_data else None
            distance_m = route_info.get('distance', {}).get('value')

            if duration_sec is not None and distance_m is not None:
                logging.info(f"Google Directions API ({mode}, traffic={traffic}) result: Duration {format_seconds(duration_sec)}, Distance {distance_m / 1000:.2f} km")
                return {
                    'duration_seconds': duration_sec,
                    'distance_meters': distance_m,
                    'raw_result': directions_result # Include the raw data if needed
                 }
            else:
                logging.warning(f"Could not extract duration/distance from Google API result for {mode}.")
                return None
        else:
            logging.info(f"No route found by Google Directions API for mode '{mode}'.")
            return None

    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Google API Error: {e}")
        return None
    except googlemaps.exceptions.HTTPError as e:
         logging.error(f"Google API HTTP Error: {e}")
         return None
    except googlemaps.exceptions.Timeout:
         logging.error("Google API request timed out.")
         return None
    except Exception as e:
        logging.error(f"An unexpected error occurred calling/processing Google API: {e}", exc_info=True)
        return None


# --- Main Execution Logic ---

def run_analysis():
    """Orchestrates the loading, graph building, analysis, and saving process."""
    main_start_time = time.time()
    logging.info("=" * 50)
    logging.info("Starting GTFS Transit Network Analysis Script")
    logging.info(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 50)

    # --- Setup ---
    if not create_output_dir(OUTPUT_DIR):
        logging.critical("Failed to create or access output directory. Exiting.")
        return # Exit if output dir fails

    # --- Phase 1: Load Data ---
    logging.info("\n--- Phase 1: Loading GTFS Data ---")
    gtfs_data = load_gtfs_data(GTFS_DIR, EDSA_STOPS_CSV)
    if gtfs_data is None:
        logging.critical("Failed to load GTFS data. Exiting.")
        return
    # Extract key dataframes for easier access
    stops: Optional[pd.DataFrame] = gtfs_data.get('stops_valid')
    routes: Optional[pd.DataFrame] = gtfs_data.get('routes')
    trips: Optional[pd.DataFrame] = gtfs_data.get('trips')
    stop_times: Optional[pd.DataFrame] = gtfs_data.get('stop_times_filtered')
    if stops is None or routes is None or trips is None or stop_times is None:
        logging.critical("Essential GTFS DataFrames are missing after loading. Exiting.")
        return
    logging.info("GTFS data loaded successfully.")

    # --- Phase 2: Build Network Graph ---
    logging.info("\n--- Phase 2: Building Network Graph ---")
    # Using travel time for weights is generally preferred for realism
    G = build_transit_graph(gtfs_data, use_travel_time=True)
    if G is None or G.number_of_nodes() == 0:
        logging.critical("Graph construction failed or resulted in an empty graph. Exiting.")
        return
    logging.info("Network graph built successfully.")
    # Define the weight attribute used in the graph for subsequent analyses
    graph_weight_attribute = 'travel_time' # Matches the setting in build_transit_graph

    # --- Phase 3: Perform Network Analyses ---
    logging.info("\n--- Phase 3: Performing Network Analyses ---")
    analysis_results = {} # Store results in memory if needed, primary output is CSV

    # (1a) Connectivity
    logging.info("\n--- Analysis 1a: Connectivity ---")
    connectivity_metrics = calculate_connectivity(G)
    analysis_results['connectivity'] = connectivity_metrics
    connectivity_df = pd.DataFrame([connectivity_metrics])
    safe_save_csv(connectivity_df, os.path.join(OUTPUT_DIR, 'connectivity_metrics.csv'), "Connectivity Metrics")

    # (1b) Centrality
    logging.info("\n--- Analysis 1b: Centrality ---")
    # Ensure the weight attribute exists if specified
    edge_weights = nx.get_edge_attributes(G, graph_weight_attribute)
    centrality_weight = graph_weight_attribute if edge_weights else None
    if not edge_weights:
         logging.warning(f"Edge attribute '{graph_weight_attribute}' not found. Centrality will be unweighted.")

    centrality_dfs = calculate_centrality(G, weight=centrality_weight)
    analysis_results['centrality'] = centrality_dfs # Store the dict of dfs
    stops_info = stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']] # For merging names/coords
    for name, df in centrality_dfs.items():
        if not df.empty:
            df_merged = pd.merge(df, stops_info, on='stop_id', how='left')
            safe_save_csv(df_merged, os.path.join(OUTPUT_DIR, f'{name}.csv'), f"{name.replace('_',' ').title()}")
        else:
            logging.warning(f"Centrality calculation for '{name}' yielded no results.")

    # (1c) Accessibility
    logging.info("\n--- Analysis 1c: Accessibility to Target Stops ---")
    target_stops_list: List[str] = []
    if TARGET_STOP_PREFIXES and not stops.empty:
        try:
            # Match by stop_id OR stop_name starting with prefixes (case-insensitive)
            prefixes_lower = [p.lower() for p in TARGET_STOP_PREFIXES]
            target_mask = stops['stop_id'].astype(str).str.lower().str.startswith(tuple(prefixes_lower), na=False) | \
                          stops['stop_name'].astype(str).str.lower().str.startswith(tuple(prefixes_lower), na=False)
            target_stops_list = stops.loc[target_mask, 'stop_id'].unique().tolist()
            logging.info(f"Identified {len(target_stops_list):,} target stops based on prefixes: {TARGET_STOP_PREFIXES}")
            # Log first few targets for verification
            if target_stops_list: logging.info(f"  (Examples: {target_stops_list[:5]}{'...' if len(target_stops_list)>5 else ''})")
        except Exception as e:
            logging.error(f"Error identifying target stops using prefixes {TARGET_STOP_PREFIXES}: {e}")
    else:
         logging.warning(f"No TARGET_STOP_PREFIXES defined or stops DataFrame is empty. Skipping target identification.")

    if not target_stops_list:
        logging.warning("Accessibility analysis skipped: No target stops were identified.")
        accessibility_df = pd.DataFrame() # Ensure it exists but is empty
    else:
        accessibility_df = calculate_accessibility(
            G,
            target_stop_ids=target_stops_list,
            weight=graph_weight_attribute,
            time_thresholds_minutes=ACCESSIBILITY_THRESHOLDS_MIN
        )
        analysis_results['accessibility'] = accessibility_df
        if not accessibility_df.empty:
            accessibility_df_merged = pd.merge(accessibility_df, stops_info, on='stop_id', how='left')
            safe_save_csv(accessibility_df_merged, os.path.join(OUTPUT_DIR, 'accessibility_to_targets.csv'), "Accessibility to Targets")
        else:
            logging.warning("Accessibility calculation yielded an empty DataFrame.")

    # (2) Route Overlap
    logging.info("\n--- Analysis 2: Route Overlap ---")
    route_sequences = get_route_stop_sequences(stop_times, trips)
    analysis_results['route_sequences'] = route_sequences # Save for potential later use
    if route_sequences:
        overlap_df = calculate_route_overlap(
            route_sequences,
            method=ROUTE_OVERLAP_METHOD,
            overlap_threshold=ROUTE_OVERLAP_THRESHOLD
        )
        analysis_results['overlap'] = overlap_df
        if not overlap_df.empty:
            # Merge route names for better readability
            routes_info = routes[['route_id', 'route_short_name', 'route_long_name']].copy()
            try:
                overlap_merged = pd.merge(overlap_df, routes_info.add_prefix('a_'), left_on='route_a', right_on='a_route_id', how='left')
                overlap_merged = pd.merge(overlap_merged, routes_info.add_prefix('b_'), left_on='route_b', right_on='b_route_id', how='left')
                # Select and order columns
                cols = ['route_a', 'a_route_short_name', 'a_route_long_name',
                        'route_b', 'b_route_short_name', 'b_route_long_name',
                        'overlap_metric', 'overlap_value', 'shared_stops',
                        'route_a_stops', 'route_b_stops']
                overlap_merged = overlap_merged[[c for c in cols if c in overlap_merged.columns]] # Handle missing route info gracefully
                safe_save_csv(overlap_merged, os.path.join(OUTPUT_DIR, 'route_overlap.csv'), "Route Overlap")
            except Exception as e:
                 logging.error(f"Failed to merge route names into overlap results: {e}. Saving basic overlap data.")
                 safe_save_csv(overlap_df, os.path.join(OUTPUT_DIR, 'route_overlap_basic.csv'), "Route Overlap (Basic)")
        else:
            logging.info(f"No route pairs found with overlap >= {ROUTE_OVERLAP_THRESHOLD:.2f} using method '{ROUTE_OVERLAP_METHOD}'.")
    else:
        logging.warning("Route sequence reconstruction failed. Skipping overlap analysis.")

    # (3a) Route Circuity
    logging.info("\n--- Analysis 3a: Route Circuity ---")
    if route_sequences: # Requires sequences from overlap step
        circuity_df = calculate_route_circuity(G, route_sequences, weight=graph_weight_attribute)
        analysis_results['circuity'] = circuity_df
        if not circuity_df.empty:
            # Merge stop and route names
            try:
                circuity_merged = pd.merge(circuity_df, stops[['stop_id', 'stop_name']].add_prefix('start_'), left_on='start_stop', right_on='start_stop_id', how='left')
                circuity_merged = pd.merge(circuity_merged, stops[['stop_id', 'stop_name']].add_prefix('end_'), left_on='end_stop', right_on='end_stop_id', how='left')
                circuity_merged = pd.merge(circuity_merged, routes[['route_id', 'route_short_name', 'route_long_name']], on='route_id', how='left')
                # Format times and select/order columns
                circuity_merged['actual_time_formatted'] = circuity_merged['actual_time_seconds'].apply(format_seconds)
                circuity_merged['shortest_path_time_formatted'] = circuity_merged['shortest_path_time_seconds'].apply(format_seconds)
                cols = ['route_id', 'route_short_name', 'route_long_name', 'circuity_index',
                        'actual_time_formatted', 'shortest_path_time_formatted',
                        'start_stop', 'start_stop_name', 'end_stop', 'end_stop_name',
                        'actual_time_seconds', 'shortest_path_time_seconds']
                circuity_merged = circuity_merged[[c for c in cols if c in circuity_merged.columns]]
                safe_save_csv(circuity_merged.sort_values(by='circuity_index', ascending=False), os.path.join(OUTPUT_DIR, 'route_circuity.csv'), "Route Circuity")
            except Exception as e:
                 logging.error(f"Failed to merge names/format times for circuity results: {e}. Saving basic circuity data.")
                 safe_save_csv(circuity_df.sort_values(by='circuity_index', ascending=False), os.path.join(OUTPUT_DIR, 'route_circuity_basic.csv'), "Route Circuity (Basic)")
        else:
            logging.info("Circuity calculation yielded no results (no valid routes/paths found).")
    else:
        logging.warning("Route sequences not available. Skipping circuity analysis.")

    # (3b) Transfer Analysis
    logging.info("\n--- Analysis 3b: Transfer Analysis (Sample OD Pairs) ---")
    all_nodes = list(G.nodes())
    sample_od_pairs: List[Tuple[str, str]] = []
    if len(all_nodes) > 1:
        num_samples = min(TRANSFER_ANALYSIS_SAMPLE_SIZE, len(all_nodes) * (len(all_nodes) - 1)) # Max possible pairs
        if num_samples > 0:
            try:
                # Generate unique pairs efficiently
                idx = np.random.choice(len(all_nodes), size=(num_samples * 2), replace=True) # Oversample initially
                potential_pairs = list(zip(np.array(all_nodes)[idx[::2]], np.array(all_nodes)[idx[1::2]]))
                # Filter out self-loops and take unique pairs up to the desired sample size
                unique_pairs = list({tuple(sorted(p)) for p in potential_pairs if p[0] != p[1]}) # Use sorted tuple for uniqueness regardless of order
                final_pairs_indices = np.random.choice(len(unique_pairs), size=min(num_samples, len(unique_pairs)), replace=False)
                sample_od_pairs = [unique_pairs[i] for i in final_pairs_indices]

                logging.info(f"Generated {len(sample_od_pairs):,} unique random OD pairs for transfer analysis.")
            except Exception as e:
                logging.error(f"Error generating random OD pairs: {e}")
                sample_od_pairs = []
        else:
             logging.warning("Not enough node combinations possible for sampling.")
    else:
        logging.warning("Not enough nodes (< 2) in the graph for transfer analysis.")

    if sample_od_pairs:
        transfers_df = analyze_transfers(G, sample_od_pairs, weight=graph_weight_attribute)
        analysis_results['transfers'] = transfers_df
        if not transfers_df.empty:
             # Merge stop names
            try:
                transfers_merged = pd.merge(transfers_df, stops[['stop_id', 'stop_name']].add_prefix('origin_'), left_on='origin', right_on='origin_stop_id', how='left')
                transfers_merged = pd.merge(transfers_merged, stops[['stop_id', 'stop_name']].add_prefix('dest_'), left_on='destination', right_on='dest_stop_id', how='left')
                # Format time
                transfers_merged['travel_time_formatted'] = transfers_merged['travel_time_seconds'].apply(format_seconds)
                cols = ['origin', 'origin_stop_name', 'destination', 'dest_stop_name',
                        'travel_time_formatted', 'num_transfers', 'path_length_stops',
                        'travel_time_seconds']
                transfers_merged = transfers_merged[[c for c in cols if c in transfers_merged.columns]]
                safe_save_csv(transfers_merged, os.path.join(OUTPUT_DIR, 'transfer_analysis_sample.csv'), "Transfer Analysis (Sample)")
            except Exception as e:
                 logging.error(f"Failed to merge names/format times for transfer results: {e}. Saving basic transfer data.")
                 safe_save_csv(transfers_df, os.path.join(OUTPUT_DIR, 'transfer_analysis_sample_basic.csv'), "Transfer Analysis (Sample, Basic)")
        else:
            logging.info("Transfer analysis did not yield results for the sampled pairs.")
    else:
        logging.info("No OD pairs generated or available for transfer analysis.")

    # (4) Enhanced Resilience Analysis
    logging.info("\n--- Analysis 4: Enhanced Resilience Analysis ---")
    resilience_results_list = []
    metrics_to_test = []
    if 'betweenness_centrality' in centrality_dfs and not centrality_dfs['betweenness_centrality'].empty:
        metrics_to_test.append('betweenness_centrality')
    if 'total_degree' in centrality_dfs and not centrality_dfs['total_degree'].empty:
         metrics_to_test.append('total_degree') # Use 'total_degree' as calculated

    if not metrics_to_test:
        logging.warning("Skipping resilience analysis: No suitable centrality results found (required: 'betweenness_centrality' or 'total_degree').")
    elif not target_stops_list:
         logging.warning("Skipping resilience analysis: No target stops identified for accessibility impact measurement.")
    else:
        for metric in metrics_to_test:
            resil_result = analyze_resilience_enhanced(
                G_original=G,
                metric_to_remove_by=metric, # Pass the exact key
                top_n_to_remove=RESILIENCE_TOP_N,
                centrality_dfs=centrality_dfs,
                target_stops=target_stops_list,
                weight=graph_weight_attribute,
                accessibility_metric_col=RESILIENCE_ACCESSIBILITY_METRIC
            )
            # Add G characteristics to the result for context
            resil_result['graph_nodes'] = G.number_of_nodes()
            resil_result['graph_edges'] = G.number_of_edges()
            resilience_results_list.append(resil_result)

        # Save combined resilience results
        if resilience_results_list:
            resilience_summary_df = pd.DataFrame(resilience_results_list)
            # Convert list of removed nodes to string for better CSV compatibility
            if 'removed_nodes' in resilience_summary_df.columns:
                 resilience_summary_df['removed_nodes_str'] = resilience_summary_df['removed_nodes'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))
            # Format floating point numbers for readability
            float_cols = resilience_summary_df.select_dtypes(include=['float']).columns
            resilience_summary_df[float_cols] = resilience_summary_df[float_cols].round(4)
            # Handle potential infinities before saving
            resilience_summary_df.replace([float('inf'), float('-inf')], [999999.0, -999999.0], inplace=True) # Or use strings 'Infinity'

            safe_save_csv(resilience_summary_df, os.path.join(OUTPUT_DIR, 'resilience_analysis_summary.csv'), "Resilience Analysis Summary")
        else:
            logging.info("No resilience analysis scenarios were successfully run.")


    # --- Phase 4: Google API Comparison (Optional) ---
    logging.info("\n--- Phase 4: Google API Comparison ---")

    # Prerequisites check (remains the same)
    if not GOOGLE_API_KEY:
        logging.info("Google API Key not provided. Skipping Google API comparison.")
    elif not googlemaps:
        logging.info("Google Maps library not installed. Skipping Google API comparison.")
    elif G.number_of_nodes() < 2:
        logging.warning("Graph has fewer than 2 nodes. Skipping Google API comparison.")
    elif 'betweenness_centrality' not in centrality_dfs or centrality_dfs['betweenness_centrality'].empty:
        logging.warning("Betweenness centrality data not available. Cannot select hubs for comparison. Skipping.")
    elif stops is None or stops.empty:
         logging.warning("Stops data not available. Cannot get coordinates for comparison. Skipping.")
    else:
        logging.info(f"Attempting Google API comparison for {NUM_API_COMPARISONS} OD pairs, prioritizing hubs...")
        comparison_results = [] # To store results for saving

        try:
            # --- Select OD Pairs ---
            # 1. Identify Top Hubs
            betweenness_df = centrality_dfs['betweenness_centrality']
            top_hubs = betweenness_df.nlargest(NUM_TOP_HUBS_FOR_SAMPLING, 'betweenness_centrality')['stop_id'].tolist()
            logging.info(f"Identified top {len(top_hubs)} hubs based on betweenness for sampling.")

            # 2. Get all valid stop IDs
            all_stop_ids = stops['stop_id'].tolist()
            if len(all_stop_ids) < 2: raise ValueError("Not enough stops for OD pair generation.")

            # 3. Generate Candidate OD Pairs
            candidate_pairs: Set[Tuple[str, str]] = set()
            hub_sample_size = min(len(top_hubs), 50)
            if hub_sample_size >= 2:
                hub_indices = np.random.choice(hub_sample_size, size=(NUM_API_COMPARISONS * 2), replace=True)
                for i in range(0, len(hub_indices), 2):
                    o_idx, d_idx = hub_indices[i], hub_indices[i+1]
                    if o_idx != d_idx: candidate_pairs.add(tuple(sorted((top_hubs[o_idx], top_hubs[d_idx]))))
            num_hub_random_pairs = NUM_API_COMPARISONS * 2
            if top_hubs and all_stop_ids:
                hub_origins = np.random.choice(top_hubs, size=num_hub_random_pairs, replace=True)
                random_destinations = np.random.choice(all_stop_ids, size=num_hub_random_pairs, replace=True)
                for o, d in zip(hub_origins, random_destinations):
                     if o != d: candidate_pairs.add(tuple(sorted((o, d))))

            # 4. Final Selection
            unique_pairs_list = list(candidate_pairs)
            if len(unique_pairs_list) == 0:
                 logging.warning("No unique hub-related pairs generated, falling back to random sampling.")
                 indices = np.random.choice(len(all_stop_ids), size=(NUM_API_COMPARISONS * 3), replace=True)
                 potential_pairs = list(zip(np.array(all_stop_ids)[indices[::2]], np.array(all_stop_ids)[indices[1::2]]))
                 unique_pairs_list = list({tuple(sorted(p)) for p in potential_pairs if p[0] != p[1]})

            num_pairs_to_select = min(NUM_API_COMPARISONS, len(unique_pairs_list))
            if num_pairs_to_select < NUM_API_COMPARISONS:
                 logging.warning(f"Generated only {num_pairs_to_select} unique OD pairs, requested {NUM_API_COMPARISONS}.")
            if num_pairs_to_select > 0:
                selected_indices = np.random.choice(len(unique_pairs_list), size=num_pairs_to_select, replace=False)
                selected_od_pairs = [unique_pairs_list[i] for i in selected_indices]
                logging.info(f"Selected {len(selected_od_pairs)} OD pairs for comparison.")
            else:
                 selected_od_pairs = []
                 logging.error("Could not generate any valid OD pairs for comparison.")

            # --- Loop Through Selected OD Pairs ---
            # get stop details, get PT time, call API, store results
            for i, (node1_id, node2_id) in enumerate(selected_od_pairs):
                logging.info("-" * 35)
                logging.info(f"Comparing OD Pair {i+1}/{len(selected_od_pairs)}: {node1_id} <-> {node2_id}")
                try:
                    # Get stop details
                    origin_stop = stops.loc[stops['stop_id'] == node1_id].iloc[0]
                    dest_stop = stops.loc[stops['stop_id'] == node2_id].iloc[0]
                    origin_name = origin_stop.get('stop_name', node1_id)
                    dest_name = dest_stop.get('stop_name', node2_id)
                    origin_coords = (origin_stop['stop_lat'], origin_stop['stop_lon'])
                    dest_coords = (dest_stop['stop_lat'], dest_stop['stop_lon'])
                    logging.info(f"  Origin:      {origin_name} ({node1_id}) @ {origin_coords}")
                    logging.info(f"  Destination: {dest_name} ({node2_id}) @ {dest_coords}")

                    # 1. PT time from Graph
                    pt_time_graph_sec: Optional[float] = None
                    pt_path_len: Optional[int] = None
                    try:
                        pt_time_graph_sec, pt_path = nx.single_source_dijkstra(G, source=node1_id, target=node2_id, weight=graph_weight_attribute)
                        pt_path_len = len(pt_path)
                        logging.info(f"  > PT (Graph):    {format_seconds(pt_time_graph_sec)} ({pt_path_len} stops)")
                    except nx.NetworkXNoPath: logging.info(f"  > PT (Graph):    No path found.")
                    except Exception as e_path: logging.error(f"  > PT (Graph):    Error finding path: {e_path}")

                    # 2. Driving time from Google API
                    driving_time_google_sec: Optional[float] = None
                    driving_dist_google_m: Optional[float] = None
                    car_route_info = get_google_directions_route(origin_coords, dest_coords, mode='driving', api_key=GOOGLE_API_KEY, traffic=True)
                    if car_route_info and 'duration_seconds' in car_route_info:
                        driving_time_google_sec = car_route_info['duration_seconds']
                        driving_dist_google_m = car_route_info.get('distance_meters')
                        logging.info(f"  > Driving (API):   {format_seconds(driving_time_google_sec)}")
                    else: logging.info(f"  > Driving (API):   Failed/No Route")

                    # 3. Transit time from Google API
                    transit_time_google_sec: Optional[float] = None
                    transit_route_info = get_google_directions_route(origin_coords, dest_coords, mode='transit', api_key=GOOGLE_API_KEY)
                    if transit_route_info and 'duration_seconds' in transit_route_info:
                        transit_time_google_sec = transit_route_info['duration_seconds']
                        logging.info(f"  > PT (API):      {format_seconds(transit_time_google_sec)}")
                    else: logging.info(f"  > PT (API):      Failed/No Route")

                    # Store results
                    comparison_results.append({
                        'origin_id': node1_id, 'origin_name': origin_name,
                        'destination_id': node2_id, 'destination_name': dest_name,
                        'pt_time_graph_seconds': pt_time_graph_sec, 'pt_path_stops': pt_path_len,
                        'driving_time_google_seconds': driving_time_google_sec, 'driving_distance_google_meters': driving_dist_google_m,
                        'transit_time_google_seconds': transit_time_google_sec,
                    })
                except IndexError: logging.warning(f"Could not find stop details for {node1_id} or {node2_id}. Skipping pair.")
                except Exception as e_pair: logging.error(f"Error processing OD pair ({node1_id}, {node2_id}): {e_pair}", exc_info=False) # Less verbose exc_info

                logging.debug(f"Waiting {API_CALL_DELAY_SECONDS}s before next API comparison")
                time.sleep(API_CALL_DELAY_SECONDS)
            # --- End Loop ---

            # --- Create DataFrame from results ---
            comparison_df = pd.DataFrame() # Initialize empty df
            if comparison_results:
                 comparison_df = pd.DataFrame(comparison_results)

            # --- *** NEW: Calculate and Log Summary Statistics *** ---
            if not comparison_df.empty:
                logging.info("-" * 35)
                logging.info(f"Google API Comparison Summary ({len(comparison_df)} pairs attempted):")

                # Ensure time columns are numeric for calculations, coercing errors
                num_cols = ['pt_time_graph_seconds', 'driving_time_google_seconds', 'transit_time_google_seconds']
                for col in num_cols:
                     if col in comparison_df.columns:
                          comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
                     else: # Add column as NaN if it wasn't created
                          comparison_df[col] = np.nan

                # Calculate averages (skipna=True is default for mean)
                avg_pt_graph = comparison_df['pt_time_graph_seconds'].mean()
                avg_driving_google = comparison_df['driving_time_google_seconds'].mean()
                avg_transit_google = comparison_df['transit_time_google_seconds'].mean()

                # Count successful lookups
                valid_pt_graph_count = comparison_df['pt_time_graph_seconds'].notna().sum()
                successful_driving_lookups = comparison_df['driving_time_google_seconds'].notna().sum()
                successful_transit_lookups = comparison_df['transit_time_google_seconds'].notna().sum()

                logging.info(f"  Avg PT Time (Graph):    {format_seconds(avg_pt_graph)} (from {valid_pt_graph_count} valid paths)")
                logging.info(f"  Avg Driving Time (API): {format_seconds(avg_driving_google)} (from {successful_driving_lookups} successful lookups)")
                logging.info(f"  Avg Transit Time (API): {format_seconds(avg_transit_google)} (from {successful_transit_lookups} successful lookups)")

                # PT (Graph) vs Driving (API) Comparison
                valid_driving_comp = comparison_df.dropna(subset=['pt_time_graph_seconds', 'driving_time_google_seconds'])
                if not valid_driving_comp.empty:
                    num_valid_driving_comp = len(valid_driving_comp)
                    avg_diff_pt_driving = (valid_driving_comp['pt_time_graph_seconds'] - valid_driving_comp['driving_time_google_seconds']).mean()
                    pt_faster_driving_count = (valid_driving_comp['pt_time_graph_seconds'] < valid_driving_comp['driving_time_google_seconds']).sum()
                    logging.info(f"  Avg Diff (PT Graph - Driving API): {format_seconds(avg_diff_pt_driving)} "
                                 f"(Positive means PT longer; based on {num_valid_driving_comp} pairs)")
                    logging.info(f"  Count where PT (Graph) faster than Driving (API): {pt_faster_driving_count} / {num_valid_driving_comp}")
                else:
                     logging.info("  Not enough data for PT Graph vs Driving API comparison.")

                # PT (Graph) vs Transit (API) Comparison
                valid_transit_comp = comparison_df.dropna(subset=['pt_time_graph_seconds', 'transit_time_google_seconds'])
                if not valid_transit_comp.empty:
                    num_valid_transit_comp = len(valid_transit_comp)
                    avg_diff_pt_transit = (valid_transit_comp['pt_time_graph_seconds'] - valid_transit_comp['transit_time_google_seconds']).mean()
                    pt_faster_transit_count = (valid_transit_comp['pt_time_graph_seconds'] < valid_transit_comp['transit_time_google_seconds']).sum()
                    logging.info(f"  Avg Diff (PT Graph - Transit API): {format_seconds(avg_diff_pt_transit)} "
                                 f"(Positive means Graph PT longer; based on {num_valid_transit_comp} pairs)")
                    logging.info(f"  Count where PT (Graph) faster than Transit (API): {pt_faster_transit_count} / {num_valid_transit_comp}")
                else:
                     logging.info("  Not enough data for PT Graph vs Transit API comparison.")
                logging.info("-" * 35)
            # --- End of Summary Block ---

            # --- Save Comparison Results ---
            # Now save the comparison_df (which might be empty or have NaNs)
            if not comparison_df.empty:
                # Add formatted columns for readability (handling potential NaNs)
                comparison_df['pt_time_graph_formatted'] = comparison_df['pt_time_graph_seconds'].apply(format_seconds)
                comparison_df['driving_time_google_formatted'] = comparison_df['driving_time_google_seconds'].apply(format_seconds)
                comparison_df['transit_time_google_formatted'] = comparison_df['transit_time_google_seconds'].apply(format_seconds)
                # Reorder columns
                cols_order = [
                     'origin_id', 'origin_name', 'destination_id', 'destination_name',
                     'pt_time_graph_formatted', 'driving_time_google_formatted', 'transit_time_google_formatted',
                     'pt_path_stops', 'driving_distance_google_meters',
                     'pt_time_graph_seconds', 'driving_time_google_seconds', 'transit_time_google_seconds'
                ]
                comparison_df = comparison_df[[col for col in cols_order if col in comparison_df.columns]]

                output_path = os.path.join(OUTPUT_DIR, 'google_api_comparison_sample.csv')
                safe_save_csv(comparison_df, output_path, "Google API Comparison Results")
            else:
                 # This case happens if comparison_results list remained empty
                 logging.info("No comparison results were generated to save.")

        except Exception as e_main_comp:
            logging.error(f"An error occurred during the Google API comparison setup or loop: {e_main_comp}", exc_info=True)

    # Save Route Sequences (existing logic remains here)
    if route_sequences:
        try:
            output_path_seq = os.path.join(OUTPUT_DIR, 'route_sequences.json')
            sequences_for_json = {k: list(v) for k, v in route_sequences.items()}
            with open(output_path_seq, 'w', encoding='utf-8') as f:
                json.dump(sequences_for_json, f, indent=4)
            logging.info(f"Saved reconstructed route sequences to {output_path_seq}")
        except Exception as e:
            logging.error(f"Failed to save route sequences JSON: {e}")

    # --- Script Finish ---
    logging.info("="*50)
    logging.info(f"Network Analysis Script Finished")
    logging.info(f"Total execution time: {time.time() - main_start_time:.2f} seconds ({format_seconds(time.time() - main_start_time)})")
    logging.info(f"Results saved in directory: {os.path.abspath(OUTPUT_DIR)}")
    logging.info("="*50)


if __name__ == "__main__":
    run_analysis()