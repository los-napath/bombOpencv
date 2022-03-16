import cv2 as cv
import random as rd
from math import fabs

class Bomb():
    def __init__(self, frame_width = 1280, frame_height = 720):
        # frame_width & height -> size of the image that will be background
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.bomb = cv.imread('../imgs/bomb_256.png')
        self.bombgray = cv.cvtColor(self.bomb,cv.COLOR_BGR2GRAY)
        self.bomb_width = self.bombgray.shape[1]
        self.bomb_height = self.bombgray.shape[0]
        ret, self.mask = cv.threshold(self.bombgray, 10, 255, cv.THRESH_BINARY)
        self.mask_inv = cv.bitwise_not(self.mask)
        self.bomb_fg = cv.bitwise_and(self.bomb,self.bomb, mask = self.mask)
        self.position = []
        self.countdown = []
        self.x_range = self.frame_width-self.bomb_width
        self.y_range = self.frame_height-self.bomb_height

    def getPosition(self):
        return self.position

    def addPosition(self, pos):
        self.position.append(pos)

    def setPositon(self, position):
        self.position = position

    def getCountdown(self):
        return self.countdown

    def setCountdown(self, countdown):
        self.countdown = countdown 

    def randomCountdown(self):
        time = rd.randint(100, 1000)
        self.countdown.append(time)

    def bombsCollapse(self, gen_x, gen_y):
        if len(self.getPosition()) == 0:
            return False
        else:
            for pos in self.getPosition():
                check_x = True
                check_y = True
                if gen_x > pos[0]-self.bomb_width+1 and gen_x < pos[0]+self.bomb_width-1:
                    check_x = False
                if gen_y > pos[1]-self.bomb_height+1 and gen_y < pos[1]+self.bomb_height-1:
                    check_y = False
                if not(check_x or check_y):
                    return True
            return False

    def randomPosition(self):
        x = rd.randint(0,self.frame_width-self.bomb_width)
        y = rd.randint(0,self.frame_height-self.bomb_height)
        while self.bombsCollapse(x, y):
            x = rd.randint(0,self.frame_width-self.bomb_width)
            y = rd.randint(0,self.frame_height-self.bomb_height)
        pos = (x, y)
        self.addPosition(pos)
        self.randomCountdown()

    def centerBomb(self, bomb_pos):
        return (int(bomb_pos[0]+self.bomb_width/2), int(bomb_pos[1]+self.bomb_height/2))

    def find_m(self, point1, point2):
        # point2 -> end_beam; otherwise problem with negative val
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        if x2-x1 == 0:
            print('Divided by zero')
            return 1
        return (y2-y1)/(x2-x1)

    def find_intersect(self, ml, mr, bomb_pos, end_beam, Q):
        x0, y0 = bomb_pos[0], bomb_pos[1]
        xb, yb = end_beam[0], end_beam[1]
        # if Q == 4 or Q == 2:
        #     check_xl = xb - (yb-y0)/ml
        #     check_yl = yb - ml*(xb-x0)
        #     check_xr = xb - (yb-y0)/mr
        #     check_yr = yb - mr*(xb-x0)
        # elif Q == 1 or Q == 3:
        #     check_xl = xb - (yb-(y0+self.bomb_width))/ml
        #     check_yl = yb - ml*(xb-x0)
        #     check_xr = xb - (yb-(y0+self.bomb_width))/mr
        #     check_yr = yb - mr*(xb-x0)
        check_xl = xb - (yb-y0)/ml
        check_yl = yb - ml*(xb-x0)
        check_xr = xb - (yb-y0)/mr
        check_yr = yb - mr*(xb-x0)
        return (check_xl, check_yl), (check_xr, check_yr)
        

    def isHit(self, eye_left, eye_right, end_beam, Q):
        for index, pos in enumerate(self.getPosition()):
            k = 20
            ml = self.find_m(eye_left, end_beam)
            mr = self.find_m(eye_right, end_beam)
        
            (xl, yl), (xr, yr) = self.find_intersect(ml, mr, (pos[0]+k, pos[1]+k), end_beam, Q)

            if Q == 4 or Q == 1 :
                if ((xr > pos[0] and xr < pos[0]+self.bomb_width) and (yl > pos[1] and yl < pos[1]+self.bomb_height)) or                                                             ((yr > pos[1] and yr < pos[1]+self.bomb_height) and (yl > pos[1] and yl < pos[1]+self.bomb_height)) or                                                             ((xr > pos[0] and xr < pos[0]+self.bomb_width) and (xl > pos[0] and xl < pos[0]+self.bomb_width)):
                    # print(end_beam, pos)
                    if fabs(pos[0]-end_beam[0]) > 400:
                        return False, None, None
                    return True, self.centerBomb(pos), index

            elif Q == 2 or Q == 3:
                if ((xl > pos[0] and xl < pos[0]+self.bomb_width) and (yr > pos[1] and yr < pos[1]+self.bomb_height)) or                                                             ((yr > pos[1] and yr < pos[1]+self.bomb_height) and (yl > pos[1] and yl < pos[1]+self.bomb_height)) or                                                             ((xr > pos[0] and xr < pos[0]+self.bomb_width) and (xl > pos[0] and xl < pos[0]+self.bomb_width)):
                    # print(end_beam, pos)
                    if fabs(pos[0]-end_beam[0]) > 400:
                        return False, None, None
                    return True, self.centerBomb(pos), index
        return False, None, None

    def displayBomb(self, image, bomb_pos):
        # bomb_pos -> a list of bombs position (x, y) in pixel
        mask = self.mask_inv
        for index, bpos in enumerate(bomb_pos):
            center = self.centerBomb(bpos)
            time = self.countdown[index]
            x = bpos[0]
            y = bpos[1]
            roi = image[y: y+self.bomb_height, x: x+self.bomb_width]
            img_bg = cv.bitwise_and(roi,roi,mask = mask)
            result = cv.add(img_bg, self.bomb_fg)
            image[y: y+self.bomb_height, x: x+self.bomb_width] = result
            cv.putText(image, str(time/10), (center[0]-50, center[1]+20), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        return image

    def clearState(self):
        self.position = []
        self.x_range = list(range(self.frame_width-self.bomb_width))
        self.y_range = list(range(self.frame_height-self.bomb_height))
