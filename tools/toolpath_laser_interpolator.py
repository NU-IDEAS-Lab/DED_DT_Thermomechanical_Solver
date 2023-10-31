#This script interpolates a laser power profile along a predefined toolpath and outputs ARPI commands
import numpy as np
import os

def print_rel_movement(textfile, move_vec, scan_speed):
    with open(textfile, 'a') as output_text_file:
        x_move = move_vec[0]
        y_move = move_vec[1]
        z_move = move_vec[2]
        string = ("LINEAR X" + str(x_move)
                  + " Y" + str(y_move)
                  + " Z" + str(z_move)
                  + " F" + str(scan_speed))
        output_text_file.write(string + os.linesep)
        
# Mode
laser_power_mode = "const"        

geom_name = "circle_control"
laser_profile = 600.;
project = "PSED"

# "GUIDE" or "MAIN"
laser = "GUIDE"

# Parameters
interp_freq = 1.0 # in hz - approximate number of times to update per second.

if laser_power_mode == "profile":
    laserprof_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "laser_inputs", geom_name, laser_profile)
    laser_hist = np.loadtxt(laserprof_loc + ".csv", skiprows=1, delimiter=',')
    output_toolpath_path = project + "_" + laser_profile + "_" + laser + ".txt"
elif laser_power_mode == "const":
    output_toolpath_path = project + "_" + str(laser_profile) + "W_const_" + laser + ".txt"
else:
    raise Exception("Invalid laser profile type")

f = open(output_toolpath_path, "w")
f.close()

# Load toolpath
toolpath_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", project, "toolpath.crs")
toolpath = np.loadtxt(toolpath_loc)

old_laser_state = False     # False for turned off

current_time = toolpath[0, 0];
current_location = toolpath[0, 1:4]
if laser_power_mode == "const":
    with open(output_toolpath_path, 'a') as output_obj:
        output_obj.write("$AO[0].X = " + str(laser_profile / 100.) + os.linesep)
        
for itr in range(1, toolpath.shape[0]):
    # Old destination is previous point
    old_time = current_time
    old_location = current_location
    
    # End of current scan
    current_time = toolpath[itr, 0]
    current_location = toolpath[itr, 1:4]
    
    # Delta T
    delT_move = current_time - old_time
    
    # Relative move vector
    move_vector = current_location - old_location

    # Scan speed
    move_dist = np.linalg.norm(move_vector)
    scan_speed = move_dist / delT_move

    new_laser_state = int(toolpath[itr, 4])
    if old_laser_state and new_laser_state:
        # Laser is on and stays on
        # Number of time steps
        num_steps = min(np.int64(delT_move*interp_freq), 1)
        delT_laserchange = delT_move / num_steps
        move_vector_laserchange = move_vector / num_steps
        for jtr in range(0, num_steps):
            time_jtr = old_time + (jtr * delT_laserchange)
            
            # Print laser power
            if laser_power_mode == "profile":
                laser_power = np.interp(time_jtr, laser_hist[:, 1], laser_hist[:, 0])
                with open(output_toolpath_path, 'a') as output_obj:
                    output_obj.write("$AO[0].X = " + str(laser_power / 100.) + os.linesep)
            print_rel_movement(output_toolpath_path, move_vector_laserchange, scan_speed)
        old_laser_state = 1

    elif not old_laser_state and not new_laser_state:
        # laser is off and stays off
        print_rel_movement(output_toolpath_path, move_vector, scan_speed)
        old_laser_state = 0

    elif old_laser_state and not new_laser_state:
        # laser is on and turns off
        with open(output_toolpath_path, 'a') as output_obj:
            if laser == "GUIDE":
                output_obj.write("$DO[1].X = 0" + os.linesep)
            elif laser == "MAIN":
                output_obj.write("$DO[0].X = 0" + os.linesep)
            else:
                raise Exception("illegal laser mode!")
        print_rel_movement(output_toolpath_path, move_vector, scan_speed)
        old_laser_state = 0

    elif not old_laser_state and new_laser_state:
        # laser is off and turns on
        with open(output_toolpath_path, 'a') as output_obj:
            if laser == "GUIDE":
                output_obj.write("$DO[1].X = 1" + os.linesep)
            elif laser == "MAIN":
                output_obj.write("$DO[0].X = 1" + os.linesep)
                
            else:
                raise Exception("illegal laser mode!")
        num_steps = min(np.int64(delT_move*interp_freq), 1)
        delT_laserchange = delT_move / num_steps
        move_vector_laserchange = move_vector / num_steps
        for jtr in range(0, num_steps):
            time_jtr = old_time + (jtr * delT_laserchange)
            # Print laser power
            if laser_power_mode == "profile":
                laser_power = np.interp(time_jtr, laser_hist[:, 1], laser_hist[:, 0])
                with open(output_toolpath_path, 'a') as output_obj:
                    output_obj.write("$AO[0].X = " + str(laser_power / 100.) + os.linesep)
            print_rel_movement(output_toolpath_path, move_vector_laserchange, scan_speed)
        old_laser_state = 1