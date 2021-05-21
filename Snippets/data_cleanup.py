import numpy as np
import pandas as pd
import obspy

def correct_data(data, stations, correction_list, station_list):
    data_reject = obspy.core.Stream()
    corlist = pd.read_csv(correction_list)
    stalist = pd.read_csv(station_list)
    joinlist = corlist.merge(stalist, on="CODE2", how="left")
    
    for station in stations:
        subdata = data.select(station=station)
        actionlist = joinlist[joinlist["CODE1"]==station]["ACTION"].values
        if len(actionlist) < 1:
            for tr in subdata:
                data_reject += tr
                data.remove(tr)
        else:
            action = actionlist[0]

            """
            Type A: Sensors found faulty and replaced, we discard the records. 
            """
            if action == "remove":
                for tr in subdata:
                    data_reject += tr
                    data.remove(tr)

            """
            Type B: Sensors found upside-down, we invert the EW and Z components (rotation about NS axis). 
            """
            if action == "rot180NS":
                subdata.select(channel="HNZ")[0].data *= -1
                subdata.select(channel="HNE")[0].data *= -1

            """    
            Type C: Sensors found in wrong orientation, rotation about z, no component correlation 
            """
            if action == "rot90Z":
                """
                C1: 90degree rotation about Z axis (New_EW = Old_NS & New_NS = -Old_EW). 
                """
                tmp1 = subdata.select(channel="HNE")[0].data
                tmp2 = subdata.select(channel="HNN")[0].data
                subdata.select(channel="HNE")[0].data = tmp2
                subdata.select(channel="HNN")[0].data = -tmp1


            if action == "rot270Z":
                """
                C2: -90degree (or 270degree) rotation about Z axis (New_EW = -Old_NS & New_NS = Old_EW). 
                """
                tmp1 = subdata.select(channel="HNE")[0].data
                tmp2 = subdata.select(channel="HNN")[0].data
                subdata.select(channel="HNE")[0].data = -tmp2
                subdata.select(channel="HNN")[0].data = tmp1     

            if action == "rot180Z":
                """
                C3: 180degree rotation about Z axis (New_EW = -Old_EW & New_NS = -Old_NS). 
                """
                subdata.select(channel="HNE")[0].data  *= -1
                subdata.select(channel="HNN")[0].data  *= -1   

            """
            Type D: Sensors found in wrong orientation, rotation about z, correlated components. 
            """
            if action == "rot45Z":
                """
                D1: 45degree rotation about Z axis
                """
                c = np.cos(np.deg2rad(45))
                s = np.sin(np.deg2rad(45))
                tmp1 = subdata.select(channel="HNE")[0].data
                tmp2 = subdata.select(channel="HNN")[0].data
                subdata.select(channel="HNE")[0].data = c*tmp1-s*tmp2
                subdata.select(channel="HNN")[0].data = s*tmp1+c*tmp2

            if action == "rot225Z":
                """
                D1: -135degree rotation about Z axis
                """
                c = np.cos(np.deg2rad(225))
                s = np.sin(np.deg2rad(225))
                tmp1 = subdata.select(channel="HNE")[0].data
                tmp2 = subdata.select(channel="HNN")[0].data
                subdata.select(channel="HNE")[0].data = c*tmp1-s*tmp2
                subdata.select(channel="HNN")[0].data = s*tmp1+c*tmp2

            if action == "rot315Z":
                """
                D1: -45degree rotation about Z axis
                """
                c = np.cos(np.deg2rad(315))
                s = np.sin(np.deg2rad(315))
                tmp1 = subdata.select(channel="HNE")[0].data
                tmp2 = subdata.select(channel="HNN")[0].data
                subdata.select(channel="HNE")[0].data = c*tmp1-s*tmp2
                subdata.select(channel="HNN")[0].data = s*tmp1+c*tmp2
            
    return data_reject, joinlist



def correct_data_graves(data, stations, correction_list, station_list):
    data_reject = obspy.core.Stream()
    corlist = pd.read_csv(correction_list)
    stalist = pd.read_csv(station_list)
    joinlist = corlist.merge(stalist, on="CODE2", how="left")
    
    for station in stations:
        subdata = data.select(station=station)
        action = joinlist[joinlist["CODE2"]==station]["ACTION"]
        if len(action) == 0:
            for tr in subdata:
                data_reject += tr
                data.remove(tr)
            continue
        else:
            for tr in subdata:
                tr.stats.station = joinlist[joinlist["CODE2"]==station]["CODE1"].values[0]
            action = joinlist[joinlist["CODE2"]==station]["ACTION"].values[0]

        
        
        """
        Type A: Sensors found faulty and replaced, we discard the records. 
        """
        if action == "remove":
            for tr in subdata:
                data_reject += tr
                data.remove(tr)
        
        """
        The others are corrected in the real data and shouldn't be a problem for synthetics...
        """
            
    return data_reject, joinlist