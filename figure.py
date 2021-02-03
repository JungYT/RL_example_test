import numpy as np
import fym.logging as logging
import fym.plotting as plotting
import matplotlib.pyplot as plt
import os
import fym.utils.rot as rot
#import keyboard

# Data load
past = -1
loglist = sorted(os.listdir('./log'))
load_path = os.path.join('log', loglist[past], 'data.h5')
data = logging.load(load_path)
save_path = os.path.join('log', loglist[past])

# Data arrangement
for key in data['state']['pendulum'].keys():
    data[key] = np.squeeze(data['state']['pendulum'][key])

#data["euler"] = []
#for dcm in data["dcm"]:
#    data["euler"].append(rot.quat2angle(rot.dcm2quat(dcm)))
#data["euler"] = np.array(data["euler"])


# Setting
draw_dict = {
    "theta": {       # figure name
        "projection": "2d", # 2d or 3d
        "plot": [["time", "th"]], # if 2d, [[x, y]], if 3d, [pos]
        "type": None, # none or scatter
        "label": None, # legend
        "c": "b", # color
        "alpha": None, # transparent
        "xlabel": "time [s]",
        "ylabel": ["theta [deg]"],
        "xlim": None,
        "ylim": None,
        "axis": None, # none or equal
        "grid": True,
    },
    "omega": {
        "projection": "2d", # 2d or 3d
        "plot": [["time", "thdot"]], # if 2d, [x, y], if 3d, [pos]
        "type": None, # none or scatter
        "label": None, # legend
        "c": "b", # color
        "alpha": None, # transparent
        "xlabel": "time [s]",
        "ylabel": ["omega [deg/s]"],
        "xlim": None,
        "ylim": None,
        "axis": None, # none or equal
        "grid": True,
    },
    "input": {       # figure name
        "projection": "2d", # 2d or 3d
        "plot": [["time", "input"]], # if 2d, [[x, y]], if 3d, [pos]
        "type": None, # none or scatter
        "label": None, # legend
        "c": "b", # color
        "alpha": None, # transparent
        "xlabel": "time [s]",
        "ylabel": ["u [Nm]"],
        "xlim": None,
        "ylim": None,
        "axis": None, # none or equal
        "grid": True,
    },
}

weight_dict = {
    "th": 180 / np.pi,
    "omega": 180 / np.pi
}

option = {
    "savefig": {
        "onoff": True,
        "dpi": 150,  # resolution
        "transparent": False,
        "bbox_inches": 'tight',     # None or tight, 
        "format": None,   # file format. png(default), pdf, svg, ...
    },
    "showfig": {
        "onoff": True,
        "showkey": [],
    }
}
# Plot
plotting.plot(data, draw_dict, weight_dict, save_dir=save_path, option=option)

## Close window
#if "showfig" in option and "onoff" in option["showfig"] and option["showfig"]\
#    ["onoff"] is True:
#    while True:
#        if keyboard.is_pressed('q'):
#            plt.close('all')
#            break
