from numpy import matrix
from numpy import linalg
import numpy

def createVector(*nums):
    aTup = ()
    for i in nums:
        aTup = aTup + (i,)

def gn_qua():
    return guassNewton(quadratic)

def guassNewton(f):
    filePath = input("Please tpye the path of your file (ex C:\\Users\\Ikenna\\Desktop\\file.txt): ")
    file = open("test_case_qua.txt")

    floatingPair = []
    for item in file.readlines():
        item = item.strip()
        x,y = item.split(",")
        floatingPair.append((float(x), float(y)))

    a = float(input("Enter a guess for a: "))
    b = float(input("Enter a guess for b: "))
    c = float(input("Enter a guess for c: "))

    iterations = int(input("How many iterations? "))

    B = matrix([[a,b,c]]).T #Laxy raw to make 3x1
    r = []

    for pair in floatingPair:
        num = pair[1] - f(B, pair[0], -1)
        r.append([num])

    r = matrix(r)

    J = []

    for i in range(len(floatingPair)):
        aList = []
        for j in range(3):
            aList.append(f(B, floatingPair[i][0], j))
        J.append(aList)

    J = matrix(J)

    for i in range(iterations):
        B = B - (inverse((J.T*J)) * J.T) * matrix(r)
        r = []

        for pair in floatingPair:
            num = pair[1] - f(B, pair[0], -1)
            r.append([num])

        r = matrix(r)

        J = []

        for i in range(len(floatingPair)):
            aList = []
            for j in range(3):
                aList.append(f(B, floatingPair[i][0], j))
            J.append(aList)
        J = matrix(J)
    print(B)
        

    
def quadratic(aMatrix,x,var): #matrix should be 3x1
    #The quadratic formula for this problem is ax^2 + bx + c
    #The partial derivatives are as follows:
    #B1 -> -x^2 + B2*x + B3
    #B2 -> -B1*x^2 + x + B3
    #B3 -> -B1*x^2 + B*x + 1

    a = numpy.array(aMatrix)[0][0]
    b = numpy.array(aMatrix)[1][0]
    c = numpy.array(aMatrix)[2][0]

    if(var == 0):
        return -1 * x**2
    if(var == 1):
        return -1 * x 
    if(var == 2):
        return -1 * 1
    return a*x**2 + b*x + c

def determinant(matrix):
    if(matrix.shape == (2,2)):
       a = numpy.array(matrix)[0][0]
       c = numpy.array(matrix)[0][1]
       b = numpy.array(matrix)[1][0]
       d = numpy.array(matrix)[1][1]

       return a*d - b*c
    else:
        sign = 1
        totalSum = 0
        for i in range(matrix.shape[0]):
            det = numpy.array(matrix)[0][i] * determinant(getSubMatrix(matrix, i))
            totalSum = totalSum + sign * det
            sign = sign * -1
        return totalSum

def getSubMatrix(aMatrix, column):
    aList = []
    for i in range(1, aMatrix.shape[0]):
        for j in range(aMatrix.shape[0]):
            if(column != j):
                aList.append(numpy.array(aMatrix)[i][j])
    return matrix(aList).reshape((aMatrix.shape[0] - 1, aMatrix.shape[0] - 1))

def inverse(matrix):
    return reducedRowEchelon(matrix)[:,3:]

def reducedRowEchelon(aMatrix):
    array = numpy.array(aMatrix)
    identity = numpy.identity(aMatrix.shape[0])
    
    newArray = []
    
    #initializing list
    index = 0
    while index < aMatrix.shape[0]:
        newArray.append(array[index].tolist() + identity[index].tolist())
        index = index + 1

        
    #begin reducing matrix
    for i in range(aMatrix.shape[0]):
        #Divide and make a 1 for current item
        newArray[i] = (matrix(newArray[i]) / newArray[i][i]).tolist()[0]

        for j in range(aMatrix.shape[0]):
            if(j!=i):
                #perform row operations
                newArray[j] = (matrix(newArray[j]) - matrix(newArray[i]) * newArray[j][i]).tolist()[0]

    return(matrix(newArray))

        
    
