_RAW_DIR = "Raw Data\\"
_RAW_FN = "recording_731M_Spontaneous_50Hz_Naive_no_injections.csv"
RAW_PATH = f"{_RAW_DIR}{_RAW_FN}"

CHUNKSIZE = 1000

# From 120 to 140 seconds (seizure)
FROM_INDEX = 6000
TO_INDEX = 7000

RESOLUTION = 256

SCALE = 8

TOL = 0.01

PROCESSED_DIR = "Time Frames\\"
