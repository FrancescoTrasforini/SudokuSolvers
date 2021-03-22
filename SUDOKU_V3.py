#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def box_placement(row,col):
    box = 0
    if(row in [0,1,2]):
        if(col in [0,1,2]):
            box = 1
            return box
        if(col in [3,4,5]):
            box = 2
            return box
        if(col in [6,7,8]):
            box = 3
            return box
    if(row in [3,4,5]):
        if(col in [0,1,2]):
            box = 4
            return box
        if(col in [3,4,5]):
            box = 5
            return box
        if(col in [6,7,8]):
            box = 6
            return box
    if(row in [6,7,8]):
        if(col in [0,1,2]):
            box = 7
            return box
        if(col in [3,4,5]):
            box = 8
            return box
        if(col in [6,7,8]):
            box = 9
            return box


# In[3]:


def box_content(box_number,board):
    row = 0
    column = 0
    
    content = np.array([],int)
    if(box_number == 1):
        for row in range(3):
            for column in range(3):
                content = np.append(content,board[row][column])    
        return content
    if(box_number == 2):
        for row in range(3):
            for column in range(3,6):
                content = np.append(content,board[row][column])        
        return content
    if(box_number == 3):
        for row in range(3):
            for column in range(6,9):
                content = np.append(content,board[row][column])  
        return content
    if(box_number == 4):
        for row in range(3,6):
            for column in range(3):
                content = np.append(content,board[row][column])
        return content
    if(box_number == 5):
        for row in range(3,6):
            for column in range(3,6):
                content = np.append(content,board[row][column])
        return content
    if(box_number == 6):
        for row in range(3,6):
            for column in range(6,9):
                content = np.append(content,board[row][column])       
        return content
    if(box_number == 7):
        for row in range(6,9):
            for column in range(3):
                content = np.append(content,board[row][column])     
        return content
    if(box_number == 8):
        for row in range(6,9):
            for column in range(3,6):
                content = np.append(content,board[row][column])
                
        return content
    if(box_number == 9):
        for row in range(6,9):
            for column in range(6,9):
                content = np.append(content,board[row][column])
                
        return content
    if(box_number < 1 or box_number > 9):
        return 0


# In[4]:


def check_empty(board,row,col):
    if(board[row][col] == 0):
        return True
    else:
        return False


# In[5]:


def find_empty(board):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if(check_empty(board,row,col)):
                return(row,col)
    return None


# In[6]:


def check_constraints(board,number,row,col):
    
    domain = np.array([1,2,3,4,5,6,7,8,9])
    
    row_container = np.array([],int)
    col_container = np.array([],int)
    
    for k in range(9):
        row_container = np.append(row_container,board[row][k])
        col_container = np.append(col_container,board[k][col])
    
    #eliminate 0s from row_container and col_container
    row_container = row_container[row_container != 0]
    col_container = col_container[col_container != 0]
    
    #I find the box of the current position
    box = box_placement(row,col)
    #box_container is a vector where I store the values currently inserted in the given box
    box_container = box_content(box,board)
    #eliminate 0s from box_container
    box_container = box_container[box_container != 0]

    constraints = np.concatenate((row_container, col_container, box_container))
    constraints = np.unique(constraints)
    valid = np.setdiff1d(domain,constraints)
    
    if(number in valid):
        return True
    else:
        return False


# In[7]:


def solve_cpb(board):
    
    empty_position = find_empty(board)
    if not empty_position:
        return True
    else:
        row, col = empty_position

    for k in range(1,10):
        if (check_constraints(board,k,row,col)):
            board[row][col] = k
            if(solve_cpb(board)):
                return True
            board[row][col] = 0
        
    return False


# In[8]:


from random import randint

ncells = 9
totcells = 81
p = np.ones((totcells*ncells, 1))/ncells
rij = np.zeros((totcells*ncells,totcells*ncells))

def initializeRij():
    global ncells, totcells, rij

    for i in range(totcells):
        for lb in range(ncells):
            for j in range(totcells):
                for mu in range(ncells):
                    rij[i*ncells + lb][j*ncells + mu] = compatibility(i,j,lb,mu)
    np.savetxt('rij.csv', rij, delimiter=',')


def compatibility(i, j, lb, mu):
    if i == j:
        return 0
    if lb != mu:
        return 1
    if checkRow(i,j) or checkColumn(i,j) or checkBox(i,j):
        return 0
    return 1


def checkColumn(i, j):
    global ncells
    return i%ncells == j%ncells

def checkRow(i,j):
    global ncells
    return i//ncells == j//ncells

def checkBox(i, j):
    global ncells

    i_x = i // ncells
    i_y = i % ncells
    j_x = j // ncells
    j_y = j % ncells
    start_i_x = i_x - i_x%3
    start_i_y = i_y - i_y%3
    start_j_x = j_x - j_x%3
    start_j_y = j_y - j_y%3
    return start_i_x == start_j_x and start_i_y == start_j_y


def averageConsistency(q):
    global ncells, p
    return np.sum(p*q)

def relaxationLabeling():
    global rij, p
    diff = 1
    avg_b = 0
    t = 0
    while diff > 0.001:
        q = np.dot(rij, p)
        num = p * q
        row_sums = num.reshape(ncells*ncells,ncells).sum(axis=1)
        p = (num.reshape(ncells*ncells,ncells)/row_sums[:, np.newaxis]).reshape(729,1)
        avg = averageConsistency(q)
        diff = avg - avg_b
        avg_b = avg
        t += 1
    p = p.reshape(totcells, ncells)

def solve_relaxationLabeling(board, create = True):
    global ncells,totcells, p, rij
    if (create):
        initializeRij() # initialize matrix Rij at startup
        create = False
        p = np.ones((totcells*ncells, 1))/ncells

    ncells = len(board_c)

    for row in range(ncells):
        for col in range(ncells):
            domain = np.array([1,2,3,4,5,6,7,8,9])
            row_container = np.array([],int)
            col_container = np.array([],int)
            for k in range(9):
                row_container = np.append(row_container,board_c[row][k])
                col_container = np.append(col_container,board_c[k][col])
            #eliminate 0s from row_container and col_container
            row_container = row_container[row_container != 0]
            col_container = col_container[col_container != 0]
            #I find the box of the current position
            box = box_placement(row,col)
            #box_container is a vector where I store the values currently inserted in the given box
            box_container = box_content(box,board_c)
            #eliminate 0s from box_container
            box_container = box_container[box_container != 0]
            constraints = np.concatenate((row_container, col_container, box_container))
            constraints = np.unique(constraints)
            valid = np.setdiff1d(domain,constraints)
            n = len(valid)
            prob = np.zeros((1, ncells))[0]
            if (not board_c[row][col] == 0):
                # value just assigned
                val = int(board_c[row][col])
                prob[val-1] = 1
            else:
                for k in valid:
                    prob[int(k) - 1] = 1/n + randint(0,20)/100.0
            prob = prob/np.sum(prob)
            p.reshape(ncells,ncells,ncells)[row][col] = prob


    rij = np.loadtxt("rij.csv", delimiter=",") 
    relaxationLabeling()
    for i in range(totcells):
        pos = np.argmax(p[i])
        if i % ncells == 0:
            print("")
        print(pos+1, end=" ")


# In[9]:


import pandas as pd
from pandas import read_csv
r = 100
dataset = read_csv("C:/Users/Francesco/Desktop/sudoku.csv",usecols=["quizzes"],nrows=r,header=0)


# In[10]:


#data preparation
#elements in the board are not separed, so I need to do some work to split them
def prep_data(dataloc):
    s = dataset.iloc[dataloc]
    s = [*s]

    st = s[0]
    arr = np.zeros(81,dtype=np.int32)
    for i in range (0,81):
        slice_object = slice(i,i+1)
        arr[i] = st[slice_object] 
    return arr


# In[11]:


import time as t
exec_time = np.zeros((r,2))
for i in range (0,r):
    b = np.zeros(81,dtype=np.int32)
    b = prep_data(i)
    new_arr = b.reshape(9,9)
    board = new_arr
    board_c = np.copy(board)
    start = t.time()
    solve_cpb(board)
    end = t.time()
    exec_time[i][0] = end - start
    start = t.time()
    solve_relaxationLabeling(board_c)
    end = t.time()
    exec_time[i][1] = end - start


# In[12]:


print("Average execution time\nConstraint Propagation and Backtracking - Relaxation Labeling: ",np.mean(exec_time, axis=0))


# In[13]:


print("Execution time standard deviation\nConstraint Propagation and Backtracking - Relaxation Labeling: ",np.std(exec_time, axis=0))


# In[15]:


np.mean(exec_time[:,1],axis=0)/np.mean(exec_time[:,0],axis=0)

