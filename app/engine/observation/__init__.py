# app/engine/observation/__init__.py
#
# The 9-state AerobicObserverParams that lived here has been deprecated.
# Archived at: app/engine/observation/_deprecated/aerobic_observer_9state.py
#
# Active observation models live in each slice:
#   app/slices/cardiorespiratory/observation.py  (6-state, h_cardio)
#   app/slices/neuromuscular_tissue/observation.py
#   app/slices/neuroendocrine/observation.py
#   ... (one observation.py per slice)
