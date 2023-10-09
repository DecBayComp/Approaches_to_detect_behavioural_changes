from enum import Enum, unique
import os.path

@unique
class Tracker(Enum):
    T5    = 0
    T15   = 1
    T2    = 3

    def get_stimulus_time(tracker):
        if tracker == Tracker.T5:
            return 45.
        elif tracker == Tracker.T15:
            return 30.
        elif tracker == Tracker.T2:
            return 60.
        else:
            raise ValueError('Stimulus time is not defined for tracker {}'.format(tracker.name))

    def get_start_time(tracker):
        return Tracker.get_stimulus_time(tracker)-10.

    def from_path(path):
        if 't5' in os.path.basename(path):
            return Tracker.T5 
        elif 't15' in os.path.basename(path):
            return Tracker.T15

class Feature(Enum):
    TIME          =  0
    IS_RUN        =  1
    FIRST_LABEL   =  1
    IS_BEND       =  2
    IS_STOP       =  3
    IS_HUNCH      =  4
    IS_BACK       =  5
    IS_ROLL       =  6
    LAST_LABEL    =  6 
    LENGTH        =  7
    X1            =  8
    FIRST_COORD   =  8       
    Y1            =  9
    X2            = 10
    Y2            = 11
    X3            = 12
    X_MID_SEGMENT = 12
    Y3            = 13
    Y_MID_SEGMENT = 13
    X4            = 14
    Y4            = 15
    X5            = 16
    Y5            = 17
    LAST_COORD    = 17

@unique
class Label(Enum):
    RUN   = 0
    BEND  = 1
    STOP  = 2
    HUNCH = 3
    BACK  = 4
    ROLL  = 5

@unique
class Timeslot(Enum):
    SETUP    = 0
    BEFORE   = 1
    DURING   = 2
    AFTER    = 3

    def from_timestamp(timestamp, tracker):
        stimulus_time = Tracker.get_stimulus_time(tracker)
        start_time = Tracker.get_start_time(tracker)
        if timestamp < start_time:
            return Timeslot.SETUP
        elif timestamp < stimulus_time:
            return Timeslot.BEFORE
        elif timestamp < stimulus_time + 2.:
            return Timeslot.DURING
        else:
            return Timeslot.AFTER