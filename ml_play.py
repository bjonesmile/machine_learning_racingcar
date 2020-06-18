import pickle
import os.path as path
import numpy as np
from scipy.spatial.distance import pdist

class MLPlay:
    def __init__(self, player):
        self.player = player
        if self.player == "player1":
            self.player_no = 0
        elif self.player == "player2":
            self.player_no = 1
        elif self.player == "player3":
            self.player_no = 2
        elif self.player == "player4":
            self.player_no = 3
        self.car_vel = 0
        self.car_pos = (0,0)
        self.last_cmd = ""                    # pos initial
        self.car_lane = self.car_pos[0] // 70       # lanes 0 ~ 8
        self.lanes = [35, 105, 175, 245, 315, 385, 455, 525, 595]  # lanes center
        filename = path.join(path.dirname(__file__),"save/model_RandomForest.pickle") #model_RandomForest
        with open(filename, 'rb') as file:
            self.clf = pickle.load(file)
        pass

    def update(self, scene_info):
        def check_grid():
            grid = set()
            speed_ahead = 100
            if self.car_pos[0] <= 65: # left bound
                grid.add(1)
                grid.add(4)
                grid.add(7)
            elif self.car_pos[0] >= 565: # right bound
                grid.add(3)
                grid.add(6)
                grid.add(9)

            for car in scene_info["cars_info"]:
                if car["id"] != self.player_no:
                    x = self.car_pos[0] - car["pos"][0] # x relative position
                    y = self.car_pos[1] - car["pos"][1] # y relative position
                    if x <= 40 and x >= -40 :      
                        if y > 0 and y < 300:
                            grid.add(2)
                            if y < 200:
                                speed_ahead = car["velocity"]
                                grid.add(5) 
                        elif y < 0 and y > -200:
                            grid.add(8)
                    if x > -100 and x < -40 :
                        if y > 80 and y < 250:
                            grid.add(3)
                        elif y < -80 and y > -200:
                            grid.add(9)
                        elif y < 80 and y > -80:
                            grid.add(6)
                    if x < 100 and x > 40:
                        if y > 80 and y < 250:
                            grid.add(1)
                        elif y < -80 and y > -200:
                            grid.add(7)
                        elif y < 80 and y > -80:
                            grid.add(4)
            return move(grid= grid, speed_ahead = speed_ahead)
            
        def move(grid, speed_ahead): 
            # if self.player_no == 0:
            #     print(grid)
            if len(grid) == 0:
                return ["SPEED"]
            else:
                if (2 not in grid): # Check forward 
                    # Back to lane center
                    if self.car_pos[0] > self.lanes[self.car_lane]:
                        return ["SPEED", "MOVE_LEFT"]
                    elif self.car_pos[0 ] < self.lanes[self.car_lane]:
                        return ["SPEED", "MOVE_RIGHT"]
                    else :return ["SPEED"]
                else:
                    if (5 in grid): # NEED to BRAKE
                        if (4 not in grid) and (7 not in grid): # turn left 
                            if self.car_vel < speed_ahead:
                                if self.last_cmd == "MOVE_RIGHT":
                                    return ["SPEED", "MOVE_RIGHT"]
                                self.last_cmd = "MOVE_LEFT"
                                return ["SPEED", "MOVE_LEFT"]
                            else:
                                return ["BRAKE", "MOVE_LEFT"]
                        elif (6 not in grid) and (9 not in grid): # turn right
                            if self.car_vel < speed_ahead:
                                if self.last_cmd == "MOVE_LEFT":
                                    return ["SPEED", "MOVE_LEFT"]
                                self.last_cmd = "MOVE_RIGHT"
                                return ["SPEED", "MOVE_RIGHT"]
                            else:
                                return ["BRAKE", "MOVE_RIGHT"]
                        else : 
                            if self.car_vel < speed_ahead:  # BRAKE
                                return ["SPEED"]
                            else:
                                return ["BRAKE"]
                    if (self.car_pos[0] < 60 ):
                        self.last_cmd = "MOVE_RIGHT"
                        return ["SPEED", "MOVE_RIGHT"]
                    if (1 not in grid) and (4 not in grid) and (7 not in grid): # turn left 
                        self.last_cmd = "MOVE_LEFT"
                        return ["SPEED", "MOVE_LEFT"]
                    if (3 not in grid) and (6 not in grid) and (9 not in grid): # turn right
                        self.last_cmd = "MOVE_RIGHT"
                        return ["SPEED", "MOVE_RIGHT"]
                    if (1 not in grid) and (4 not in grid): # turn left
                        self.last_cmd = "MOVE_LEFT"
                        return ["SPEED", "MOVE_LEFT"]
                    if (3 not in grid) and (6 not in grid): # turn right
                        self.last_cmd = "MOVE_RIGHT"
                        return ["SPEED", "MOVE_RIGHT"]
                    if (4 not in grid) and (7 not in grid): # turn left 
                        self.last_cmd = "MOVE_LEFT"
                        return ["MOVE_LEFT"]    
                    if (6 not in grid) and (9 not in grid): # turn right
                        self.last_cmd = "MOVE_RIGHT"
                        return ["MOVE_RIGHT"]
        isBrake = False
        self.car_pos = scene_info[self.player]
        for car in scene_info["cars_info"]:
            if car["id"]==self.player_no:
                self.car_vel = car["velocity"]
                #print(self.car_vel)
                break
        #self.car_lane = self.car_pos[0] // 70
        if scene_info["status"] != "ALIVE":
            return ["RESET"]
        if self.car_vel ==0 and scene_info["frame"]>10: #check for divid zero error when car is out
            return ["RESET"]
        if not all(self.car_pos): #check for out of range tuple error
            return ["RESET"]
        ser_x = -1
        ser_y = -1
        min_distance = 10000
        for car in scene_info["cars_info"]:
            if car["id"]!=self.player_no:
                pos = car["pos"]
                #if pos[1] < 0:
                    #continue  
                if(pos[1]<self.car_pos[1]+40 and abs(pos[0]-self.car_pos[0])<40):
                    X = np.array([[pos[0],pos[1]],[self.car_pos[0],self.car_pos[1]]],dtype='float64')
                    distance = pdist(X, 'euclidean')
                    """
                    print("detect front car")
                    print("id:"+str(car["id"])+" distance: "+str(distance))
                    print(pos)
                    """
                    if((distance-80)/self.car_vel<20 and min_distance > distance):
                        min_distance = distance
                        ser_x = pos[0]
                        ser_y = pos[1]
                        isBrake = True

        if min_distance <= 120: #dircetly brake when too close
            return check_grid()
        
        x = -1
        isMoveRight = True
        isMoveLeft = True
        if(isBrake):
            #consider move right
            ser_range = (ser_x + 80,ser_y+40)
            if(ser_range[0]>800):
                ser_range = (800,ser_y+40)
            for car in scene_info["cars_info"]:
                pos = car["pos"]
                if(pos[0]>self.car_pos[0] and pos[0]<ser_range[0]) and (pos[1]<self.car_pos[1]+60 and pos[1]>ser_range[1]):
                    isMoveRight = False
                    #print("id:"+str(car["id"])+" exist couldnt move right")
                    break
            #avoid hit to right wall
            if(isMoveRight and self.car_pos[0]>590):
                isMoveRight = False
            
            #consider move left
            ser_range = (ser_x - 80,ser_y+40)
            if(ser_range[0]<0):
                ser_range = (0,ser_y+40)
            for car in scene_info["cars_info"]:
                pos = car["pos"]
                if(pos[0]<self.car_pos[0] and pos[0]>ser_range[0]) and (pos[1]<self.car_pos[1]+60 and pos[1]>ser_range[1]):
                    isMoveLeft = False
                    #print("id:"+str(car["id"])+" exist couldnt move left")
                    break
            #avoid hit to left wall
            if(isMoveLeft and self.car_pos[0] <= 40):
                isMoveLeft = False

            if(isMoveRight or isMoveLeft):
                x = self.car_pos[0]
                isBrake = False
        else:
            isMoveRight = False
            isMoveLeft = False

        if scene_info["status"] != "ALIVE":
            return "RESET"
        #return ["MOVE_LEFT", "MOVE_RIGHT", "SPEED", "BRAKE"]
        else:
            if(isBrake):
                return check_grid()
            else:
                if(isMoveRight and isMoveLeft):
                    PlayerCar_x = self.car_pos[0]
                    PlayerCar_y = self.car_pos[1]
                    Velocity = self.car_vel
                    ComputerCar_lane = 0
                    Difference_y = 0
                    ComputerCarDif_x = []
                    ComputerCarDif_y = []
                    ComputerCar_x = []
                    ComputerCar_y = []

                    p_x = self.car_pos[0]
                    p_y = self.car_pos[1]
                    lane = self.car_lane

                    for car in scene_info["cars_info"]:
                        if car["id"] != self.player_no:
                            ComputerCar_x.append(car["pos"][0])
                            ComputerCar_y.append(car["pos"][1])
                            ComputerCarDif_x.append(car["pos"][0]-p_x)
                            ComputerCarDif_y.append(car["pos"][1]-p_y)

                    for i in range(len(ComputerCarDif_x)):
                        if ComputerCar_x[i] //70 == lane and ComputerCarDif_y[i] <= 250 and ComputerCarDif_y[i] >0:
                        # if computer car's lane is equal to player's lane and the y_difference is bewteen 0 and 250 means the forward site exist cars
                            Difference_y = ComputerCarDif_y[i] 
                            ComputerCar_lane = ComputerCar_x[i] # record the information of the moment which we add the label
                            break
    
                        elif ComputerCar_x[i]//70 == lane+1 and ComputerCarDif_y[i]<=80 and ComputerCarDif_y[i]>=-80:
                        # if computer car's lane is in the right site of player's lane and the y_difference is bewteen -80 and 80 means the right site exust cars
                            Difference_y = ComputerCarDif_y[i]
                            ComputerCar_lane = ComputerCar_x[i] # record the information of the moment which we add the label
                            break
            
                        elif ComputerCar_x[i]//70 == lane-1 and ComputerCarDif_y[i]<=80 and ComputerCarDif_y[i]>=-80:
                        # if computer car's lane is in the left site of player's lane and the y_difference is bewteen -80 and 80 means the left site exust cars
                            Difference_y = ComputerCarDif_y[i]
                            ComputerCar_lane = ComputerCar_x[i] # record the information of the moment which we add the label
                            break
                                          
                        elif ComputerCar_x[i] //70 == lane and ComputerCarDif_y[i] >=250:
                        # if computer car's lane is equal to player's lane and the y_difference is more than 250 means the player's car can speed up
                            Difference_y = ComputerCarDif_y[i]
                            ComputerCar_lane = ComputerCar_x[i] # record the information of the moment which we add the label
                            break
            
                        if (i == len(ComputerCarDif_x)-1):
                        # if all the computer's cars is not match the prevoius condition then means the player's car can speed up 
                            Difference_y = ComputerCarDif_y[i]
                            ComputerCar_lane = ComputerCar_x[i] # record the information of the moment which we add the label
                            break
        
                    feature = np.array([ComputerCar_lane,Difference_y,Velocity])
                    feature = feature.reshape((-1,len(feature)))
                    action = self.clf.predict(feature)
                    pred = []
                    print("frame: ",scene_info["frame"],"action :",action)
                    if(action == 11):
                        pred = ["SPEED", "MOVE_RIGHT"]
                    elif(action == 12):
                        pred = ["SPEED", "MOVE_LEFT"]
                    elif(action == 10):
                        pred = ["SPEED"]
                    elif(action == 21):
                        pred = ["BRAKE", "MOVE_RIGHT"]
                    elif(action == 22):
                        pred = ["BRAKE", "MOVE_LEFT"]
                    elif(action == 20):
                        pred = ["BRAKE"]
                    elif(action == 1):
                        pred = ["MOVE_RIGHT"]
                    elif(action == 2):
                        pred = ["MOVE_LEFT"]
                    else:
                        pred = ["SPEED"]
                    if(abs(x-0)>abs(x-800)):
                        if self.last_cmd == "BRAKE" and "BRAKE" in pred:
                            print("take action")
                            return pred
                        # consider last cmd avoid left->right->left->right...
                        if self.last_cmd == "MOVE_RIGHT":
                            #print("sink in loop!")
                            if self.car_pos[0] >= 595:
                                self.last_cmd = "BRAKE"
                                return check_grid()
                            self.last_cmd = "MOVE_RIGHT"
                            return ["MOVE_RIGHT", "SPEED"]
                        self.last_cmd = "MOVE_LEFT"
                        return ["MOVE_LEFT", "SPEED"]
                    else:
                        if self.last_cmd == "BRAKE" and "BRAKE" in pred:
                            print("take action")
                            return pred
                        # consider last cmd avoid left->right->left->right...
                        if self.last_cmd == "MOVE_LEFT":
                            #print("sink in loop!")
                            if self.car_pos[0] <= 35:
                                self.last_cmd = "BRAKE"
                                return check_grid()
                            self.last_cmd = "MOVE_LEFT"
                            return ["MOVE_LEFT", "SPEED"]
                        self.last_cmd = "MOVE_RIGHT"
                        return ["MOVE_RIGHT", "SPEED"]
                elif(isMoveRight):
                    self.last_cmd = "MOVE_RIGHT"
                    return ["MOVE_RIGHT", "SPEED"]
                elif(isMoveLeft):
                    self.last_cmd = "MOVE_LEFT"
                    return ["MOVE_LEFT", "SPEED"]
            self.last_cmd = "SPEED"
            return ["SPEED"]        


    def reset(self):
        """
        Reset the status
        """
        pass
