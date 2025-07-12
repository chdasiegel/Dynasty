import pandas as pd
from nfl_data_py.seasonal import import_seasonal_data
from nfl_data_py.rosters import import_rosters
from nfl_data_py.snap_counts import import_snap_counts
from nfl_data_py.ff_participation import import_ff_participation

# 1. Define the range of years to gather
years = list(range(2015, 2025))

# 2. Load seasonal stats and roster data
print("Importing seasonal and roster data...")
seasonal = import_seasonal_data(years, s_type="REG")
roster = import_rosters(years)

# 3. Merge to get position info
data = seasonal.merge(roster[["player_id", "season", "position"]], on=["player_id", "season"], how="left")

# 4. Filter for WRs only
wr = data[data["position"] == "WR"].copy()

# 5. Add snap share
print("Importing snap counts...")
snaps = import_snap_counts(years)
wr = wr.merge(snaps[["player_id", "season", "snap_pct"]], on=["player_id", "season"], how="left")

# 6. Tag rookie/second year
print("Tagging rookie seasons...")
wr["rookie_season"] = wr.groupby("player_id")["season"].transform("min")
wr["year_number"] = wr["season"] - wr["rookie_season"] + 1

# 7. Add route participation for TPRR/YPRR
print("Importing route data...")
routes = import_ff_participation(years)
wr = wr.merge(routes[["player_id", "season", "routes_run"]], on=["player_id", "season"], how="left")

# 8. Calculate TPRR / YPRR
print("Calculating efficiency metrics...")
wr["TPRR"] = wr["targets"] / wr["routes_run"]
wr["YPRR"] = wr["rec_yds"] / wr["routes_run"]

# 9. Label breakouts — top 25% in next year receiving yards
print("Labeling breakouts...")
wr = wr.sort_values(["player_id", "season"])
wr["next_rec_yds"] = wr.groupby("player_id")["rec_yds"].shift(-1)
threshold = wr["rec_yds"].quantile(0.75)
wr["breakout"] = (wr["next_rec_yds"] >= threshold).astype(int)

# 10. Save to CSV
print("Saving to wr_breakout_dataset.csv")
wr.to_csv("wr_breakout_dataset.csv", index=False)
print("✅ Done! File saved.")
