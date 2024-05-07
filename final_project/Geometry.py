from dataclasses import dataclass


@dataclass
class Airplane:
    wingspan: float
    surface_area: float
    taper: float
    root_twist: float
    tip_twist: float
    root_chord: float
    tip_chord: float
    taper_start: float
