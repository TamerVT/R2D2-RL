# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:50:17 2026

@author: felix
"""

class robot:
    def __init__(self):
        pass
    
    
    
    
    
def main():
    Robot = robot()
    Robot.initialize()
    colors = [input("specify color")]
    coords_bowl = input("specify coordinates of bowl")
    # main loop
    for c in colors:
        cubes = Robot.find(c) # evaluate model, and if not found get to default position and find cube, if not found return None
        if len(cubes)>0:
            for cub in cubes:
                
    
    
    
if __name__ == "__main__":
    main()