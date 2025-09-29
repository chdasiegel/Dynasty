# src.features_config.py
# =====================================================
# Central config for feature names, interactions, aliases
# =====================================================

BASE_FEATURES = [
    "DOM++","40 Time","BMI","YPC","ELU","YCO/A","Break%","Draft Capital","Bama","Draft Age"
]

INTERACTIONS = {
    "DOMxDraft": ("DOM++", "Draft Capital"),
    "YPCxELU":   ("YPC",   "ELU"),
    "ELUxYCOA":  ("ELU",   "YCO/A"),
}

ALIASES = {
    "DOM++":         ["DOM++","DOMpp","DOM_plus_plus","DOMpp_Weighted","DOM"],
    "40 Time":       ["40 Time","Forty","40"],
    "BMI":           ["BMI"],
    "YPC":           ["YPC","Yards per Carry","Yards/Carry","Rushing YPC"],
    "ELU":           ["ELU","Elusiveness","Elusiveness Rating"],
    "YCO/A":         ["YCO/A","YAC/A","Yards After Contact / Att","Yards After Contact per Attempt"],
    "Break%":        ["Break%","Break %","Breakaway %","Breakaway Percentage","Breakaway%"],
    "Draft Capital": ["Draft Capital","Draft Cap","Draft Round","Round","Rnd"],
    "Bama":          ["Bama","Bama Rating","BamaAdj","BAMA"],
    "Shuttle":       ["Shuttle","Short Shuttle","20 Shuttle","20 Yard Shuttle"],
    "Three Cone":    ["3 Cone","Three Cone","3-Cone"],
    "Rec Yards":     ["Receiving Yards","Rec Yds","RecYds"],
    "Draft Age":     ["Draft Age","Age at Draft","DraftAge","Age (Draft)","AgeDraft","Age_at_Draft"],
}
TARGET_CANDS = ["RB Grade","RBGrade","RB_Grade"]
NAME_CANDS   = ["Player","Player Name","Name"]