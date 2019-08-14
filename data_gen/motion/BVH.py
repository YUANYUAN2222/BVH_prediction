from data_gen.motion.Quaternions import Quaternions
from data_gen.motion.Animation import Animation
import numpy as np
import re

channel_map = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channel_map_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

order_map = {
    'x': 0,
    'y': 1,
    'z': 2,
}


def load(file_name, start=None, end=None, order=None, world=False):
    """
    Reads a BVH file and constructs an animation
    
    Parameters
    ----------
    file_name: str
        File to be opened
        
    start : int
        Optional Starting Frame
        
    end : int
        Optional Ending Frame
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
        
    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------
    
    (animation, joint_names, frame_time)
        Tuple of loaded animation and joint names
    """
    
    f = open(file_name, "r")

    i = 0
    active = -1
    end_site = False
    
    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    frame_time = 0

    for line in f:
        
        if "HIERARCHY" in line:
            continue
        if "MOTION" in line:
            continue

        r_match = re.match(r"ROOT (\w+)", line)
        if r_match:
            names.append(r_match.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "{" in line:
            continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue
        
        off_match = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if off_match:
            if not end_site:
                offsets[active] = np.array([list(map(float, off_match.groups()))])
            continue
           
        chan_match = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chan_match:
            channels = int(chan_match.group(1))
            if order is None:
                channel_is = 0 if channels == 3 else 3
                channel_ie = 3 if channels == 3 else 6
                parts = line.split()[2 + channel_is: 2 + channel_ie]
                if any([p not in channel_map for p in parts]):
                    continue
                order = "".join([channel_map[p] for p in parts])
            continue

        j_match = re.match("\s*JOINT\s+(\w+)", line)
        if j_match:
            names.append(j_match.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue
        
        if "End Site" in line:
            end_site = True
            continue
              
        f_match = re.match("\s*Frames:\s+(\d+)", line)
        if f_match:
            if start and end:
                f_num = (end - start) - 1
            else:
                f_num = int(f_match.group(1))
            positions = offsets[np.newaxis].repeat(f_num, axis=0)
            rotations = np.zeros((f_num, len(orients), 3))
            continue

        f_match = re.match("\s*Frame Time:\s+([\.\d]+)", line)
        if f_match:
            frame_time = float(f_match.group(1))
            continue
        
        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue
        
        d_match = line.strip().split()
        if d_match:
            data_block = np.array(list(map(float, d_match)))
            n = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0: 1] = data_block[0: 3]
                rotations[fi, :] = data_block[3:].reshape(n, 3)
            elif channels == 6:
                data_block = data_block.reshape(n, 6)
                positions[fi, :] = data_block[:, 0: 3]
                rotations[fi, :] = data_block[:, 3: 6]
            elif channels == 9:
                positions[fi, 0] = data_block[0: 3]
                data_block = data_block[3:].reshape(n-1, 9)
                rotations[fi, 1:] = data_block[:, 3: 6]
                positions[fi, 1:] += data_block[:, 0: 3] * data_block[:, 6: 9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
    
    return Animation(rotations, positions, orients, offsets, parents), names, frame_time

    
def save(filename, anim, names=None, frame_time=1.0 / 24.0, order='zyx', positions=False):
    """
    Saves an Animation to file as BVH
    
    Parameters
    ----------
    filename: str
        File to be saved to
        
    anim : Animation
        Animation to save
        
    names : [str]
        List of joint names
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
    
    frame_time : float
        Optional Animation Frame time
        
    positions : bool
        Optional specifier to save bone
        positions for each frame
        
    """
    
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channel_map_inv[order[0]], channel_map_inv[order[1]], channel_map_inv[order[2]]))

        save_order = [0]
            
        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(f, anim, names, t, i, save_order, order=order, positions=positions)
      
        t = t[: -1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0])
        f.write("Frame Time: %f\n" % frame_time)

        rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        poss = anim.positions
        
        for i in range(anim.shape[0]):
            for j in save_order:
                if positions or j == 0:
                    f.write("%f %f %f %f %f %f " % (poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                                                    rots[i, j, order_map[order[0]]], rots[i, j, order_map[order[1]]],
                                                    rots[i, j, order_map[order[2]]]))
                else:
                    f.write("%f %f %f " % (rots[i, j, order_map[order[0]]], rots[i, j, order_map[order[1]]],
                                           rots[i, j, order_map[order[2]]]))
            f.write("\n")
    
    
def save_joint(f, anim, names, t, i, save_order, order='zyx', positions=False):
    
    save_order.append(i)
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))
    
    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channel_map_inv[order[0]], channel_map_inv[order[1]], channel_map_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" %
                (t, channel_map_inv[order[0]], channel_map_inv[order[1]], channel_map_inv[order[2]]))
    
    end_site = True
    
    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t = save_joint(f, anim, names, t, j, save_order, order=order, positions=positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[: -1]
    f.write("%s}\n" % t)
    
    return t
