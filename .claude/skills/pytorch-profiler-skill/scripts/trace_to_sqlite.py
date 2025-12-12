#!/usr/bin/env python3
"""
Convert PyTorch Profiler trace to SQLite for efficient querying.

Usage:
    python trace_to_sqlite.py trace.json.gz -o trace.db
    
Then query:
    sqlite3 trace.db "SELECT name, SUM(dur)/1e3 as ms FROM events WHERE cat='kernel' GROUP BY name ORDER BY ms DESC LIMIT 20"
"""

import gzip
import json
import sqlite3
import argparse
from pathlib import Path


def create_schema(conn: sqlite3.Connection):
    """Create database schema."""
    conn.executescript("""
        -- Main events table
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            ph TEXT,
            cat TEXT,
            name TEXT,
            pid INTEGER,
            tid INTEGER,
            ts REAL,
            dur REAL,
            -- Flattened args
            external_id INTEGER,
            correlation INTEGER,
            python_id INTEGER,
            python_parent_id INTEGER,
            device INTEGER,
            stream INTEGER,
            grid_x INTEGER,
            grid_y INTEGER,
            grid_z INTEGER,
            block_x INTEGER,
            block_y INTEGER,
            block_z INTEGER,
            registers_per_thread INTEGER,
            shared_memory INTEGER,
            occupancy REAL
        );
        
        -- Device properties
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY,
            name TEXT,
            total_global_mem INTEGER,
            compute_major INTEGER,
            compute_minor INTEGER,
            num_sms INTEGER,
            max_threads_per_block INTEGER,
            regs_per_block INTEGER,
            shared_mem_per_block INTEGER,
            warp_size INTEGER
        );
        
        -- Distributed info
        CREATE TABLE IF NOT EXISTS distributed (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        
        -- Indices for common queries
        CREATE INDEX IF NOT EXISTS idx_events_cat ON events(cat);
        CREATE INDEX IF NOT EXISTS idx_events_name ON events(name);
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation);
        CREATE INDEX IF NOT EXISTS idx_events_external_id ON events(external_id);
        CREATE INDEX IF NOT EXISTS idx_events_python_id ON events(python_id);
    """)


def load_trace(path: Path) -> dict:
    """Load trace file (supports .json and .json.gz)."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def insert_events(conn: sqlite3.Connection, events: list):
    """Insert events into database."""
    cursor = conn.cursor()
    
    for ev in events:
        args = ev.get('args', {})
        grid = args.get('grid', [None, None, None])
        block = args.get('block', [None, None, None])
        
        # Ensure grid/block are lists
        if not isinstance(grid, list):
            grid = [None, None, None]
        if not isinstance(block, list):
            block = [None, None, None]
        
        # Pad to length 3
        grid = (grid + [None, None, None])[:3]
        block = (block + [None, None, None])[:3]
        
        cursor.execute("""
            INSERT INTO events (
                ph, cat, name, pid, tid, ts, dur,
                external_id, correlation, python_id, python_parent_id,
                device, stream,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                registers_per_thread, shared_memory, occupancy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ev.get('ph'),
            ev.get('cat'),
            ev.get('name'),
            ev.get('pid'),
            ev.get('tid'),
            ev.get('ts'),
            ev.get('dur'),
            args.get('External id'),
            args.get('correlation'),
            args.get('Python id'),
            args.get('Python parent id'),
            args.get('device'),
            args.get('stream'),
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            args.get('registers per thread'),
            args.get('shared memory'),
            args.get('est. achieved occupancy %'),
        ))


def insert_devices(conn: sqlite3.Connection, devices: list):
    """Insert device properties."""
    cursor = conn.cursor()
    for d in devices:
        cursor.execute("""
            INSERT OR REPLACE INTO devices (
                id, name, total_global_mem, compute_major, compute_minor,
                num_sms, max_threads_per_block, regs_per_block,
                shared_mem_per_block, warp_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d.get('id'),
            d.get('name'),
            d.get('totalGlobalMem'),
            d.get('computeMajor'),
            d.get('computeMinor'),
            d.get('numSms'),
            d.get('maxThreadsPerBlock'),
            d.get('regsPerBlock'),
            d.get('sharedMemPerBlock'),
            d.get('warpSize'),
        ))


def insert_distributed(conn: sqlite3.Connection, dist: dict):
    """Insert distributed info."""
    cursor = conn.cursor()
    for key, value in dist.items():
        cursor.execute(
            "INSERT OR REPLACE INTO distributed (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if isinstance(value, (dict, list)) else str(value))
        )


def convert(trace_path: Path, db_path: Path):
    """Convert trace to SQLite database."""
    print(f"Loading {trace_path}...")
    data = load_trace(trace_path)
    
    print(f"Creating {db_path}...")
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    
    events = data.get('traceEvents', [])
    print(f"Inserting {len(events)} events...")
    insert_events(conn, events)
    
    devices = data.get('deviceProperties', [])
    if devices:
        print(f"Inserting {len(devices)} device records...")
        insert_devices(conn, devices)
    
    dist = data.get('distributedInfo', {})
    if dist:
        print("Inserting distributed info...")
        insert_distributed(conn, dist)
    
    conn.commit()
    conn.close()
    print("Done!")
    
    # Print some stats
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n--- Database Stats ---")
    cursor.execute("SELECT COUNT(*) FROM events")
    print(f"Total events: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT cat, COUNT(*) FROM events GROUP BY cat ORDER BY COUNT(*) DESC")
    print("\nEvents by category:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch trace to SQLite')
    parser.add_argument('trace', help='Path to trace.json or trace.json.gz')
    parser.add_argument('-o', '--output', help='Output database path (default: trace.db)')
    
    args = parser.parse_args()
    trace_path = Path(args.trace)
    db_path = Path(args.output) if args.output else trace_path.with_suffix('.db')
    
    convert(trace_path, db_path)


if __name__ == '__main__':
    main()
