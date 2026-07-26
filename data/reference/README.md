# data/reference/

`distances.csv` - origin_id, destination_id, km. `src/routing.py` looks up
km for both loaded legs (a trip's own origin -> destination) and empty
legs (one trip's destination -> the next trip's origin, same vehicle)
from this matrix. A pair not present is flagged (`km_missing=True`),
never silently dropped or treated as 0 km.

**The committed `distances.csv` is placeholder data for the synthetic
location IDs (`LOC001`-`LOC010`) only** - illustrative km values with
no relationship to any real road network, generated purely so the
routing pipeline has something to join against and can be exercised
end to end.

**Before running this on real data**, `distances.csv` must be rebuilt
with actual road distances between the real locations - via a routing
API (e.g. OSRM, Google Distance Matrix) or manual lookup - keyed by
whatever origin/destination IDs the real trip export uses. Running the
real analysis against the placeholder file will silently produce
meaningless km/empty-cost figures for real location IDs that happen not
to collide with the placeholder ones, or - if they do collide by
coincidence - meaningless km attached to the wrong locations entirely.
Replace the file before pointing the pipeline at real trip data.
