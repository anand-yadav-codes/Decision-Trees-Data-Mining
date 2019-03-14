# -*- coding: utf-8 -*-
"""


@author: Anand
"""
import numpy as np
import cv2
import math

def load(filename):    
    """Loads filename as a dataset. Assumes the last column is classes, and 
	observations are organized as rows.

	Args:
		filename: file to read

	Returns:
		A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
		where X[i] comes from the i-th row in filename; y is a list or ndarray of 
		the classes of the observations, in the same order
	"""
    file1=open(filename,'r')

    array1 = []
    array2 = []
    for a in file1 :
        rc= a.split(',')
        array1.append(rc[:-1])
        array2.append(rc[10].strip('\n'))
    Data=(array1,array2)   
    return Data
          
def entropy(D):
    classes = D[1]   
    class0_inD=0
    class1_inD=0
    i=0
    for records in classes:
        
        if (int(classes[i])==0):
            class0_inD= class0_inD + 1
        else:
            class1_inD= class1_inD + 1
        i+=1

    C0=float(class0_inD)/float(len(classes))
    C1=float(class1_inD)/float(len(classes))
    if(C0==0.0 or C0==1.0):
        H_D=0
    else :       
        H_D=-(C0)*math.log(C0,2)-(C1)*math.log(C1,2)

    return H_D        
    
def infoIG(D, index, value):        

    data = D[0]
    classes = D[1]  
    D_tr=[]
    D_tr2=[]
    D_tr3=[]
    D_tr4=[]  
    DY=[] 
    DN=[]
    i=0
    for records in data:
        
        if (float(data[i][index])<=value):
            D_tr.append(data[i])
            D_tr2.append(classes[i])            
        else:            
            D_tr3.append(data[i]) 
            D_tr4.append(classes[i])
            
        i=i+1
    DY=(D_tr,D_tr2)
    DN=(D_tr3,D_tr4)    
    H_DyDn= (float(len(D_tr))/float(len(data))*entropy(DY)) + (float(len(D_tr3))/float(len(data))*entropy(DN))
            
    return H_DyDn
    
def IG(D, index, value):
    """Compute the Information Gain of a split on attribute index at value
	for dataset D.
	
	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Information Gain for the given split
	"""

    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """    
    hdydn=infoIG(D,index,value)
    gain = (entropy(D))-hdydn
    return gain
    
def infogini (D):
    
    classes = D[1]
    class0_inD=0
    class1_inD=0
    i=0
    for records in classes:
        
        if (int(classes[i])==0):
            class0_inD= class0_inD + 1
        else:
            class1_inD= class1_inD + 1
        i+=1
    C0=float(class0_inD)/float(len(classes))
    C1=float(class1_inD)/float(len(classes))
    if(C0==0.0 or C0==1.0):
        G_D=0.0
    else:    
        G_D= 1-(C0*C0+C1*C1)    
    
    return G_D    
    
def G(D,index,value):  
    """Compute the Gini index of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the Gini index for the given split
	"""        
    data = D[0]
    classes = D[1]  
    D_tr=[]
    D_tr2=[]
    D_tr3=[]
    D_tr4=[] 
    DY= [] 
    DN = []
    i=0
    for records in data:
            if (float(data[i][index])<=value):
                D_tr.append(data[i])
                D_tr2.append(classes[i])
            else:
                D_tr3.append(data[i]) 
                D_tr4.append(classes[i])
            i=i+1
    DY=(D_tr,D_tr2)
    DN=(D_tr3,D_tr4)
    gini=(float(len(D_tr))/float(len(data))*infogini(DY))+(float(len(D_tr3))/float(len(data))*infogini(DN))
    
    return gini


def infoCART(D):
    classes = D[1]
    class0_inD=0
    class1_inD=0
    i=0
    for records in classes:
        
        if (int(classes[i])==0):
            class0_inD= class0_inD + 1
        else:
            class1_inD= class1_inD + 1
        i+=1
  
    C0=float(class0_inD)/float(len(classes))
    C1=float(class1_inD)/float(len(classes))
    rval=(C0,C1)    
    return rval
    
def CART(D,index,value):
    """Compute the CART measure of a split on attribute index at value
	for dataset D.

	Args:
		D: a dataset, tuple (X, y) where X is the data, y the classes
		index: the index of the attribute (column of X) to split on
		value: value of the attribute at index to split at

	Returns:
		The value of the CART measure for the given split
	"""    
    data = D[0]
    classes = D[1]  
    
    D_tr=[]
    D_tr2=[]
    D_tr3=[]
    D_tr4=[]
    DY= [] 
    DN = []
    i=0
    for records in data:
            if (float(data[i][index])<=value):
                D_tr.append(data[i])
                D_tr2.append(classes[i])
            else:
                D_tr3.append(data[i]) 
                D_tr4.append(classes[i])
            i=i+1
    DY=(D_tr,D_tr2)
    DN=(D_tr3,D_tr4)
    val1 =2*(float(len(D_tr))/float(len(data)))*(float(len(D_tr3))/float(len(data)))

    C0_by_DY=infoCART(DY)[0]
    C1_by_DY=infoCART(DY)[1]
    C0_by_DN=infoCART(DN)[0]    
    C1_by_DN=infoCART(DN)[1]
    val2=abs(C0_by_DY-C0_by_DN)
    val3=abs(C1_by_DY-C1_by_DN)
    
    Cart= val1*(val2+val3)
    return Cart
    
def bestSplit(D,criterion):
    """Computes the best split for dataset D using the specified criterion

	Args:
		D: A dataset, tuple (X, y) where X is the data, y the classes
		criterion: one of "IG", "GINI", "CART"

	Returns:
		A tuple (i, value) where i is the index of the attribute to split at value
	"""    
    i=0
    return_array=[]
    data=D[0]
    Bestsplit=0.0
    max_value_in_column=[]    
    q=0
       
    while q<10 :
        p=0
        max_variable=0.0 
        for records in data:
            if(max_variable<data[p][q]):
                max_variable=data[p][q]
                        
            p=p+1
        max_value_in_column.append(max_variable) 
        q=q+1
                            
    if(criterion=='IG'):
        while i<10 :
            j=0       
            for records in data:
                if(data[j][i]<max_value_in_column[i]):
                    d=float(data[j][i])               
                    if(Bestsplit<IG(D,i,d)):
                        Bestsplit = IG(D,i,d)          
                        return_array=(i,d)            
                j=j+1
            i=i+1
            
    if(criterion=='GINI'):
        Bestsplit=1.0        
        while i<10 :
            j=0
            
            for records in data:
                if(data[j][i]<max_value_in_column[i]):
                    d=float(data[j][i])
               
                    if(Bestsplit>G(D,i,d)):
                        Bestsplit = G(D,i,d)          
                        return_array=(i,d)            
                j=j+1
       
            i=i+1
       
    
    if(criterion=='CART'):
        while i<10 :
            j=0
       
            for records in data:
                if(data[j][i]<max_value_in_column[i]):
                    d=float(data[j][i])
                    if(Bestsplit<CART(D,i,d)):
                        Bestsplit = CART(D,i,d)          
                        return_array=(i,d)            
                j=j+1
       
            i=i+1    
    return return_array       



def classifyIG(train,test):
    """Builds a single-split decision tree using the Information Gain criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""
    class1_zero_val_count=0
    count=0
    classification_error_count=0    
    index,value=bestSplit(train,'IG')
    i=0
    new_class=[]
    for records in train[0]:
        if(train[0][i][index]<=value):
            count+=1
            if(train[1][i]==0):
                class1_zero_val_count+=1
                
    if(float(class1_zero_val_count)>count/2):
        DY_classifier=0
        DN_classifier=1
    else: 
        DY_classifier=1                
        DN_classifier=0
                
    for records in test[0]:     
               
        if(float(test[0][i][index])<=float(value)):                       
            new_class.append(0)             
            if int(test[1][i])!=DY_classifier:
                classification_error_count+=1
        
        if(float(test[0][i][index])>float(value)):
            new_class.append(1)             
            if int(test[1][i])!=DN_classifier:
                classification_error_count+=1                            
        i+=1
    
    return new_class
    
def classifyG(train,test):
    """Builds a single-split decision tree using the GINI criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""    
    class1_zero_val_count=0    
    count=0
    classification_error_count=0    
    new_class=[]
    index,value=bestSplit(train,'GINI')
    i=0
    for records in train[0]:
        if(train[0][i][index]<=value):
            count+=1
            if(train[1][i]==0):
                class1_zero_val_count+=1
                
    if(float(class1_zero_val_count)>count/2):
        DY_classifier=0
        DN_classifier=1
    else: 
        DY_classifier=1                
        DN_classifier=0
        
        
    for records in test[0]:     
        
        if(float(test[0][i][index])<=float(value)):
            new_class.append(0)                       
            if int(test[1][i])!=DY_classifier:
                classification_error_count+=1                
        
        if(float(test[0][i][index])>float(value)):
            new_class.append(1)            
            if float(test[1][i])!=DN_classifier:
                classification_error_count+=1                            
        i+=1
    
    return new_class
    
    
def classifyCART(train,test):    
    """Builds a single-split decision tree using the CART criterion
	and dataset train, and returns a list of predicted classes for dataset test

	Args:
		train: a tuple (X, y), where X is the data, y the classes
		test: the test set, same format as train

	Returns:
		A list of predicted classes for observations in test (in order)
	"""
    class1_zero_val_count=0
    count=0
    classification_error_count=0    
    index,value=bestSplit(train,'CART')
    i=0
    new_class=[]    
    for records in train[0]:
        if(train[0][i][index]<=value):
            count+=1
            if(train[1][i]==0):
                class1_zero_val_count+=1
    
    if(float(class1_zero_val_count)>=count/2):
        DY_classifier=0
        DN_classifier=1
    else: 
        DY_classifier=1                
        DN_classifier=0

    for records in test[0]:     
        if(float(test[0][i][index])<=float(value)):                       
            new_class.append(0)             
            if int(test[1][i])!=DY_classifier:
                classification_error_count+=1
        if(float(test[0][i][index])>float(value)):
            new_class.append(1)             
            if int(test[1][i])!=DN_classifier:
                classification_error_count+=1                            
        i+=1
   
    return new_class
    
    
def main():
    a=bestSplit(load('train.txt'),'IG')	
    b=bestSplit(load('train.txt'),'GINI')
    c=bestSplit(load('train.txt'),'CART')
    d=classifyIG(load('train.txt'),load('test.txt'))
    e=classifyG(load('train.txt'),load('test.txt'))
    f=classifyCART(load('train.txt'),load('test.txt'))
    
    print(a,b,c)
    print(d,e,f)
    



if __name__=="__main__": 
	  """__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	  """
	  main()  

    
    
    
    
    
    
    
    
    
    
    
    