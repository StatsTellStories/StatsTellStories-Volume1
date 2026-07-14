'''
This script streams compressed ADS-B aircraft trace data from a tar archive, 
cleans and normalizes the records, and writes the result as a chunked Parquet dataset.
Each aircraft trace is processed row-wise to ensure consistent typing, 
remove invalid values, and enforce a fixed schema.
The output is stored as multiple Parquet files to enable efficient downstream matching, 
#filtering, and time-series analysis without loading the full dataset into memory.
'''


from pathlib import Path
# import pandas as pd
import tarfile, gzip, json
# import pyarrow as pa
# import pyarrow.parquet as pq

from   datetime import datetime, timezone
from   glob import glob

from   flydenity import Parser
import h3
from   rich.progress import track

from paths import *
# from src.common.data_io import iter_traces

# TARGET = "/traces/"
# SUFFIX = ".json"

aircraft_keys = set([
    'alert',
    'alt_geom',
    'baro_rate',
    'category',
    'emergency',
    'flight',
    'geom_rate',
    'gva',
    'ias',
    'mach',
    'mag_heading',
    'nac_p',
    'nac_v',
    'nav_altitude_fms',
    'nav_altitude_mcp',
    'nav_heading',
    'nav_modes',
    'nav_qnh',
    'nic',
    'nic_baro',
    'oat',
    'rc',
    'roll',
    'sda',
    'sil',
    'sil_type',
    'spi',
    'squawk',
    'tas',
    'tat',
    'track',
    'track_rate',
    'true_heading',
    'type',
    'version',
    'wd',
    'ws'])

def convertToJsonL (json_files, OUT_DIR, gzipCompressed):
    parser = Parser() # Aircraft Registration Parser
    out_filename, out_file, num_recs = None, None, 0

    for filename in track(json_files, 'JSON Enrichment..'):
        if gzipCompressed:
            rec = json.loads(gzip.open(filename).read())
        else:
            with open(filename, 'r') as f:
                rec = json.loads(f.read())

        # Keep integer form to calculate offset in trace dictionary
        timestamp        = rec['timestamp']
        rec['timestamp'] =str(datetime.fromtimestamp(rec['timestamp'], tz=timezone.utc))

        if 'year' in rec.keys() and str(rec['year']) == '0000':
            continue

        if 'noRegData' in rec.keys():
            continue

        # Force casing on fields
        rec['icao']      = rec['icao'].lower()

        for key in ('desc', 'r', 't', 'ownOp'):
            if key in rec.keys():
                rec[key]  = rec[key].upper()

        if 'r' in rec.keys():
            reg_details = parser.parse(rec['r'])

            if reg_details:
                rec['reg_details'] = reg_details

        for trace in rec['trace']:
            num_recs = num_recs + 1
            _out_filename = OUT_DIR / ('traces_%02d.jsonl' % int(num_recs / 1_000_000))

            if _out_filename != out_filename:
                if out_file:
                    out_file.close()

                out_file = open(_out_filename, 'w')
                out_filename = _out_filename

            rec['trace'] = {
                'timestamp': str(datetime.fromtimestamp(timestamp + trace[0], tz=timezone.utc)),
                'lat':                     trace[1],
                'lon':                     trace[2],
                'h3_5': h3.latlng_to_cell(trace[1], trace[2], 5),
                'altitude':
                    trace[3]
                    if str(trace[3]).strip().lower() != 'ground'
                    else None,
                'ground_speed':            trace[4],
                'track_degrees':           trace[5],
                'flags':                   trace[6],
                'vertical_rate':           trace[7],
                'aircraft':
                    {k: trace[8].get(k, None)
                        if trace[8] else None
                    for k in aircraft_keys},
                'source':                  trace[9],
                'geometric_altitude':      trace[10],
                'geometric_vertical_rate': trace[11],
                'indicated_airspeed':      trace[12],
                'roll_angle':              trace[13]}

            out_file.write(json.dumps(rec, sort_keys=True) + '\n')

    if out_file:
        out_file.close()
