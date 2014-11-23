from numpy import matrix
from decimal import Decimal
import math
import random
import numpy
import os

### Tutorials I used to solidfy my knowledge on row-e: http://math.bu.edu/people/szczesny/Teaching/242S13/sec1_2ov.pdf
## Givens rotation help:
## http://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
## http://math.stackexchange.com/questions/563285/qr-factorisation-using-givens-rotation-find-upper-triangular-matrix-using-given
## http://www.cs.rpi.edu/~flaherje/pdf/lin13.pdf
## Householder help:
## http://onlinelibrary.wiley.com/doi/10.1002/9780470316757.app2/pdf
## http://planetmath.org/householdertransformation

def gn_qua():
    return guassNewton(quadratic)
def gn_exp():
    return guassNewton(exponential)
def gn_log():
    return guassNewton(logarithmic)
def gn_rat():
    return guassNewton(rational)

def guassNewton(f):
    filePath = input("Please type the path of your file (ex C:\\Users\\Ikenna\\Desktop\\file.txt): ")
    file = open(filePath)

    floatingPair = []
    for item in file.readlines():
        item = item.strip()
        x,y = item.split(",")
        floatingPair.append((float(x), float(y)))

    a = float(input("Enter a guess for a: "))
    b = float(input("Enter a guess for b: "))
    c = float(input("Enter a guess for c: "))

    iterations = int(input("How many iterations? "))

    B = matrix([[a,b,c]]).T #Lazy raw to make 3x1
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
        #B = B - (inverse((J.T*J)) * J.T) * matrix(r)
        Q,R = qr_fact_givens(J)
        B = B - (inverse(R[:R.shape[1],:])*Q[:,:R.shape[1]].T)*matrix(r)
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

def exponential(aMatrix, x, var):
    a = numpy.array(aMatrix)[0][0]
    b = numpy.array(aMatrix)[1][0]
    c = numpy.array(aMatrix)[2][0]

    if(var == 0):
        return -1 * math.exp(b*x)
    if(var == 1):
        return -1 * a*x * math.exp(b*x)
    if(var == 2):
        return -1 * 1
    return a*math.exp(b*x) + c

def logarithmic(aMatrix, x, var):
    a = numpy.array(aMatrix)[0][0]
    b = numpy.array(aMatrix)[1][0]
    c = numpy.array(aMatrix)[2][0]

    if(var == 0):
        return -1 * math.log(x + b)
    if(var == 1):
        return -1 * (a/(x+b))
    if(var == 2):
        return -1 * 1
    return a*math.log(x+b) + c

def rational(aMatrix, x, var):
    a = numpy.array(aMatrix)[0][0]
    b = numpy.array(aMatrix)[1][0]
    c = numpy.array(aMatrix)[2][0]

    if(var == 0):
        return -1 * x / (x+b)
    if(var == 1):
        return -1 * (-a*x)/((x+b)**2)
    if(var == 2):
        return -1 * 1
    return (a*x/(x+b)) + c

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
    
def trace(aMatrix):
    column = 0;
    trace = 0
    for i in numpy.array(aMatrix).tolist():
        trace = trace + i[column]
        column = column + 1
    return trace

def getSubMatrix(aMatrix, column):
    aList = []
    for i in range(1, aMatrix.shape[0]):
        for j in range(aMatrix.shape[0]):
            if(column != j):
                aList.append(numpy.array(aMatrix)[i][j])
    return matrix(aList).reshape((aMatrix.shape[0] - 1, aMatrix.shape[0] - 1))

def inverse(matrix):
    return gaussJordan(matrix)[:,matrix.shape[0]:]

def gaussJordan(aMatrix):
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

def qr_fact_givens(aMatrix):
    array = numpy.array(aMatrix)
    currentRow = aMatrix.shape[0] - 2
    currentColumn = 0
    rotations = []

    for i in range(aMatrix.shape[1]):
        rotation = []
        
        for j in range(i, aMatrix.shape[0] - 1):
            alpha = array[currentRow][currentColumn].item()
            beta = array[currentRow + 1][currentColumn].item()

            #c = alpha/r
            #r = square root (alpha^2 + beta^2)
            #s = -beta/r

            r = math.hypot(alpha,beta) #to prevent over/under flow
            #r = round(r, 4)
            
            c = alpha/r
            s = -beta/r

            identity = numpy.identity(aMatrix.shape[0]).tolist()
            identity[currentRow][currentRow] = c
            identity[currentRow+1][currentRow+1] = c
            identity[currentRow+1][currentRow] = -s
            identity[currentRow][currentRow+1] = s

            rotation.append(matrix(identity))

            currentRow = currentRow - 1

            aMatrix = matrix(identity).T * aMatrix
            array = numpy.array(aMatrix)

                
        currentRow = aMatrix.shape[0] - 2
        currentColumn = currentColumn + 1
        rotations.append(rotation)
        rotation = []
    

    #fix issue where numpy treats o as 2e-17 ( a really low number close to zero)
    R = []
    for i in aMatrix.tolist():
        for j in i:
            R.append(round(j, 6))

    Q = numpy.identity(aMatrix.shape[0])
    for i in rotations:
        for j in i:
            Q = Q * j

    return Q, matrix(R).reshape(aMatrix.shape)

def qr_fact_househ(aMatrix):
    array = numpy.array(aMatrix).tolist()
    rotations = []
    X = aMatrix

    for i in range(aMatrix.shape[1]):
        y = numpy.array(X.T).tolist()[i]
        for j in range(len(y)):
            if(j < i):
                y[j] = 0
        y = matrix(y)
        
        e = matrix(numpy.identity(aMatrix.shape[0])[i].tolist())
        u = (y - magnitude(y) * e).T / magnitude((y - magnitude(y) * e))
        H = numpy.identity(aMatrix.shape[0]) - 2 * u * u.T

        rotations.append(H)
        H = numpy.identity(H.shape[1])

        for i in rotations[::-1]:
            H = H * i
        X = H * aMatrix

    #fix issue where numpy treats o as 2e-17 ( a really low number close to zero)
    R = []
    for i in X.tolist():
        for j in i:
            R.append(round(j, 6))

    Q = numpy.identity(X.shape[0])

    for i in rotations:
        Q = Q * i.T
        
    return Q, matrix(R).reshape(X.shape)
        
def magnitude(aMatrix): # 1x3 matrix
    magnitude = 0
    for i in numpy.array(aMatrix)[0].tolist():
        magnitude = magnitude + float(i) ** 2
    return math.sqrt(magnitude)
        
def power_method(A, v, tolerance, N):
    y = matrix(v)
    x = [v]
    eigenvalue = 0
    
    for i in range(1,N+1):
        x.append((A**(i)) * v)

        a = numpy.array(v.T).tolist()[0]
        b = numpy.array(x[i].T).tolist()[0]
        c = numpy.array(x[i-1].T).tolist()[0]
        eigenvalue = numpy.dot(a,b) / numpy.dot(a,c)

        if(i > 0):
            a = numpy.array(v.T).tolist()[0]
            b = numpy.array(x[i].T).tolist()[0]
            c = numpy.array(x[i-1].T).tolist()[0]
            d = numpy.array(x[i-2].T).tolist()[0]

            if abs((numpy.dot(a,b) / numpy.dot(a,c)) - (numpy.dot(a,c) / numpy.dot(a,d))) <= tolerance:
                print("Eigenvector:\t\n", x[i]/magnitude(x[i].T))
                print("\n\n")
                print("Eigenvalue:\t", eigenvalue)
                print("Iterations\t", i)
                return i
                
    print(None)

def matrixGen(N):
    tupList = []
    for i in range(N):
        aMatrix = matrix([[random.uniform(-2,2), random.uniform(-2,2)], [random.uniform(-2,2),random.uniform(-2,2)]])
        
        while True:
            try:
                inverse(aMatrix)
                a = power_method(aMatrix, aMatrix[:,0], 5 * 10**-5, 100)
                b = power_method(inverse(aMatrix), (aMatrix)[:,0], 5 * 10**-5, 100)
                stats = determinant(aMatrix), trace(aMatrix), a
                statsB = determinant(inverse(aMatrix)), trace(inverse(aMatrix)), b
                tupList.append((stats, statsB))
                break
            except:
                aMatrix = matrix([[random.uniform(-2,2), random.uniform(-2,2)], [random.uniform(-2,2),random.uniform(-2,2)]])
    return tupList

def rotateLetters():
    K = matrix([[0,0,0],[4,0,0],[4,2,0],[2,2,0],[2,6,0],[0,6,0]])
    E = matrix([[0,0,0],[4,0,0],[4,2,0],[2,2,0],[2,6,0],[0,6,0]])
    L = matrix([[0,0,0],[4,0,0],[4,2,0],[2,2,0],[2,6,0],[0,6,0]])
    totalFrames = 121
    iterations = totalFrames//3

    directory = os.getcwd() + os.sep + "animations"

    count = 0

    K = translate(K, matrix(findCenter(L) + [1]).T)
    E = translate(E, matrix(findCenter(L) + [1]).T)
    L = translate(L, matrix(findCenter(L) + [1]).T)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(0, iterations):
        if(i > 0):
            K = rotate(K.T, (6 * math.pi) / iterations, 0)
            E = rotate(E.T, (2 * math.pi) / iterations, 0)
            L = rotate(L.T, (10 * math.pi) / iterations, 0)

        formattedListK = []
        formattedListE = []
        formattedListL = []

        for i in K.tolist():
            subList = []
            for j in i:
                subList.append(float(Decimal(j).quantize(Decimal('.00001'))))
            formattedListK.append(subList)

        for i in E.tolist():
            subList = []
            for j in i:
                subList.append(float(Decimal(j).quantize(Decimal('.00001'))))
            formattedListE.append(subList)

        for i in L.tolist():
            subList = []
            for j in i:
                subList.append(float(Decimal(j).quantize(Decimal('.00001'))))
            formattedListL.append(subList)

        kFile = open(directory + os.sep + str(count) + ".txt", "w")
        eFile = open(directory + os.sep + str(40 + count) + ".txt", "w")
        lFile = open(directory + os.sep + str(80 + count) + ".txt", "w")

        kFile.write(str(matrix(formattedListK)))
        eFile.write(str(matrix(formattedListE)))
        lFile.write(str(matrix(formattedListL)))

        count = count + 1
        kFile.close()
        eFile.close()
        lFile.close()
        

def rotate(aMatrix, rotationFactor, axis):
    if(axis == 0):    
        rotation = matrix([[1,0,0],[0,math.cos(rotationFactor), -math.sin(rotationFactor)], [0, math.sin(rotationFactor), math.cos(rotationFactor)]])
    if(axis == 1):
        rotation = matrix([[math.cos(rotationFactor), 0, math.sin(rotationFactor)], [0, 1, 0], [-math.sin(rotationFactor), 0, math.cos(rotationFactor)]])
    if(axis == 2):
        rotation = matrix([[math.cos(rotationFactor), -math.sin(rotationFactor), 0], [math.sin(rotationFactor), math.cos(rotationFactor), 0], [0,0,1]])
        
    aMatrix = (rotation * aMatrix).T
    return aMatrix

def translate(aMatrix, point):
    aList = []
    for i in aMatrix.tolist():
        trans = matrix([
                [1,0,0,i[0]],
                [0,1,0,i[1]],
                [0,0,1,i[2]],
                [0,0,0,1]
            ])
        aList.append((trans * point).T.tolist()[0])

    aMatrix = matrix(aList)[:aMatrix.shape[0],:3]
    R = []
    for i in aMatrix.tolist():
        for j in i:
            R.append(float(Decimal(j).quantize(Decimal('.00001'))))

    return matrix(R).reshape(aMatrix.shape)
def findCenter(aMatrix):
    maxX = max(aMatrix[:,0].T.tolist()[0])
    maxY = max(aMatrix[:,1].T.tolist()[0])
    maxZ = max(aMatrix[:,2].T.tolist()[0])

    return [-maxX/2,-maxY/2,-maxZ/2]

def a():
    a = "0.8147 0.0975 0.1576 0.9058 0.2785 0.9706 0.1270 0.5469 0.9572 0.9134 0.9575 0.4854 0.6324 0.9649 0.8003"
    aList = []
    for i in a.split():
        aList.append(round(float(i), 4))
    return matrix(aList).reshape(5,3)

        
    
