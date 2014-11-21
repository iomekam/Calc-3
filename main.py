from numpy import matrix
import math
import numpy

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

def getSubMatrix(aMatrix, column):
    aList = []
    for i in range(1, aMatrix.shape[0]):
        for j in range(aMatrix.shape[0]):
            if(column != j):
                aList.append(numpy.array(aMatrix)[i][j])
    return matrix(aList).reshape((aMatrix.shape[0] - 1, aMatrix.shape[0] - 1))

def inverse(matrix):
    if(matrix.shape[0] == matrix.shape[1]):
        return gaussJordan(matrix)[:,3:]
    else:
        return matrix.T * gaussJordan(matrix*matrix.T)[:,3:]

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
        

def a():
    a = "0.8147 0.0975 0.1576 0.9058 0.2785 0.9706 0.1270 0.5469 0.9572 0.9134 0.9575 0.4854 0.6324 0.9649 0.8003"
    aList = []
    for i in a.split():
        aList.append(round(float(i), 4))
    return matrix(aList).reshape(5,3)

        
    
