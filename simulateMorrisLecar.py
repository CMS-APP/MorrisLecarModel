import matplotlib.pyplot as plt
import numpy as np
import pylab as py
import scipy.integrate as integrate

class ParametersData:
    def __init__(self, paraList, args, u0):
        self.paraList = paraList
        self.args = args
        self.u0 = u0
    

def lambdaMV(lambdaM, V, V1, V2):
    return lambdaM * np.cosh((V - V1)/ 2*V2)


def lambdaNV(lambdaN, V, V3, V4):
    return lambdaN * np.cosh((V - V3)/ 2*V4)


def M_inf(V, V1, V2):
    return (1/2) * (1 + np.tanh((V - V1)/V2))


def N_inf(V, V3, V4):
    return (1/2) * (1 + np.tanh((V - V3)/V4))


def reducedSystem(t, x, I, gl, Vl, gca, Vca, gk, Vk, lambdaM, lambdaN, V1, V2, V3, V4, C):
    V, N = x[0], x[1]
    dV = (I - (gl * (V - Vl) + gca * M_inf(V, V1, V2) * (V - Vca) + gk * N * (V - Vk)))/C
    dN = lambdaN * (N_inf(V, V3, V4) - N)
    return np.array([dV, dN])


def rungeKutta(t, u, func, args, stepSize):
    k1 = func(t, u, *args)    
    k2 = func(t + stepSize/2, u + (stepSize * k1)/2, *args)
    k3 = func(t + stepSize/2, u + (stepSize * k2)/2, *args)
    k4 = func(t + stepSize, u + stepSize * k3, *args)    
    return np.array((u + (stepSize * (k1 + 2*k2 + 2*k3 + k4))/6, t + stepSize), dtype=object)


def solveTo(func, u0, args, deltaTMax, tInit, tEnd):
    t = tInit
    u = u0
    while t < tEnd:
        if t + deltaTMax > tEnd:
            deltaTMax = tEnd - t
        u, t = rungeKutta(t, u, func, args, deltaTMax)
    return u, t

def solveOde(func, u0, args, deltaTMax, tList):
    uList = [[] for val in u0]

    currentTime = tList[0]
    # For each time in the list
    for index, tEnd in enumerate(tList):
        results = solveTo(func, u0, args, deltaTMax, currentTime, tEnd)
	# Save the results from the 'solveTo' function
        for dimIndex, result in enumerate(results[0]):
            uList[dimIndex].append(result)
        u0 = results[0]
        currentTime = results[1]

    uList = np.column_stack([sols for sols in uList])
    return uList, tList

def plotSolutions(tList, sol, plotIndex):
    # For each dimension in the results
    for (index ,dimSol) in enumerate(sol.T):
        if index == 0:
            plt.plot(tList, dimSol)

    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (mV)')
    

def plotSolvedODEExample():
    paraList = ['I', 'gl', 'Vl', 'gca', 'Vca', 'gk', 'Vk', 'lambdaM', 'lambdaN', 'V1', 'V2', 'V3', 'V4', 'C']

    fig = plt.figure(figsize=(10, 10), dpi=80)

    tInit = 0; tEnd = 250			                # The time start and end.
    numberOfPoints = 200		                        # The number of time values values between this range.
    tList = np.linspace(tInit, tEnd, numberOfPoints) 
    
    for (index, I) in enumerate(range(1, 5, 1)):
        I *= 25
        args = np.array([I, gl, Vl, gca, Vca, gk, Vk, lambdaM, lambdaN, V1, V2, V3, V4, C])
        sol, tList = solveOde(reducedSystem, u0, args, deltaTMax, tList)
        plt.subplot(int('22'+str(index + 1)))
        plt.title('Current: {}'.format(round(I, 3)))
        plotSolutions(tList, sol, index)
    plt.show()


def plotSolveODEParameterChange(parametersData: ParametersData, parameter: str, startP: float, endP: float, function, tRange: list, title: str, units: str):
    paraIndex = parametersData.paraList.index(parameter)

    tInit = tRange[0]; tEnd = tRange[1]			        # The time start and end.
    numberOfPoints = 200		                        # The number of time values values between this range.
    tList = np.linspace(tInit, tEnd, numberOfPoints) 

    difference = (endP - startP)/3
    for index in range(0, 4):
        paraValue = startP + difference * index
        parametersData.args[paraIndex] = paraValue
        sol, tList = solveOde(function, parametersData.u0, parametersData.args, deltaTMax, tList)
        #plt.subplot(int('22'+str(index + 1)))

        fig = plt.figure(index)
        plt.grid()
        #plt.title(title + ': {} {}'.format(round(paraValue, 3), units))
        plotSolutions(tList, sol, index)
    plt.show()


def changeInSign(a, b):
    if (a < 0 and b > 0) or (a > 0 and b < 0):
        return True
    return False


def findMinMaxPoints(sol, tList, dim):
    def findOtherMaxMinPoints(sol, dim, minT, maxT):
        def findOtherIndex(sol, dim):
            size = len(sol.T)
            for i in range(size):
                if i != dim:
                    return i
            return None
        
        otherIndex = findOtherIndex(sol, dim)

        otherMin = sol.T[otherIndex][maxT]
        otherMax = sol.T[otherIndex][maxT]

        return otherMin, otherMax

    xMin = None; xMinT = None
    xMax = None; xMaxT = None
        
    prevX = sol.T[dim][0]
    prevX2 = sol.T[dim][1]
    for (index, x) in enumerate(sol.T[dim][2:]):
        prevDx = prevX2 - prevX
        Dx = prevX - x

        if changeInSign(prevDx, Dx):
            if prevDx < Dx:
                if xMin is not None and xMax is not None and abs(xMin - x) < 0.03:
                    if abs(xMin - xMax) < 0.75:
                        return None, None, None, None, True
                    otherMin, otherMax = findOtherMaxMinPoints(sol, dim, xMinT, xMaxT)
                    return xMin, xMax, otherMin, otherMax, True
                else:
                    xMinT = index
                    xMin = x
            if prevDx > Dx:
                if xMax is not None and xMax is not None and abs(xMax - x) < 0.03:
                    if abs(xMin - xMax) < 0.85:
                        return None, None, None, None, True
                    
                    otherMin, otherMax = findOtherMaxMinPoints(sol, dim, xMinT, xMaxT)
                    return xMin, xMax, otherMin, otherMax, True
                else:
                    xMaxT = index
                    xMax = x

        prevX2 = prevX
        prevX = x

    otherMin, otherMax = findOtherMaxMinPoints(sol, dim, xMinT, xMaxT)
    return xMin, xMax, otherMin, otherMax, False
        

def getPhasePortraitData(function, parametersData, deltaMax, tRange, dim):
    tInit = tRange[0]; tEnd = tRange[1]			        # The time start and end.
    numberOfPoints = 1000		                        # The number of time values values between this range.
    tList = np.linspace(tInit, tEnd, numberOfPoints)

    initialSol, _ = solveOde(function, parametersData.u0, parametersData.args, deltaTMax, tList)
    vMin, vMax, nMin, nMax, final = findMinMaxPoints(initialSol, tList, dim)

    if vMin is not None and final:
        print('Limit Cycle found')
        newInitialConditions = np.array([vMax, nMax])
        sol, _ = solveOde(function, newInitialConditions, parametersData.args, deltaTMax, tList)
        vMin, vMax, nMin, nMax, _ = findMinMaxPoints(sol, tList, dim)
    
    counter = 0
    while final == False:
        if counter == 6:
            final = True
        else:
            print('Trying to find equilibrium using: {}'.format([vMax, nMax]))
            newInitialConditions = np.array([vMax, nMax])
            sol, _ = solveOde(function, newInitialConditions, parametersData.args, deltaTMax, tList)
            vMin, vMax, nMin, nMax, final = findMinMaxPoints(sol, tList, dim)
        counter += 1    

    if vMin is None:
        sol = initialSol
       
    Vmax = 1.1 * sol.T[0].max()
    Vmin = sol.T[0].min() - ((Vmax - sol.T[0].min())/10)

    Nmax = 1.1 * sol.T[1].max()
    Nmin = sol.T[1].min() - ((Nmax - sol.T[1].min())/10)

    return sol, vMin, vMax, nMin, nMax, Vmin, Vmax, Nmin, Nmax
    

def plotPhasePortraitData(function, parametersData, deltaMax, tRange, dim):
    sol, vMin, vMax, nMin, nMax, Vmin, Vmax, Nmin, Nmax = getPhasePortraitData(function, parametersData, deltaMax, tRange, dim)
    V, N = np.meshgrid(np.linspace(Vmin , Vmax, 25), np.linspace(Nmin, Nmax, 25))
    plt.plot(sol.T[0], sol.T[1])
    plt.grid()
    dx = reducedSystem(0, [V, N], *parametersData.args)
    dV = dx[0]
    dN = dx[1]
    py.quiver(V, N, dV, dN * 50)
    
    py.contour(V, N, dV, levels=[0], linewidths=2, colors='green')
    py.contour(V, N, dN, levels=[0], linewidths=2, colors='green')


def plotODEPhasePortrait(function, parametersData, deltaMax, tRange, dim):
    plotPhasePortraitData(function, parametersData, deltaMax, tRange, dim)
    plt.show()


def plotQuadPhasePotrait(parametersData: ParametersData, parameter: str, startP: float, endP: float, function, tRange: list, title: str, units: str, dim: int):
    paraIndex = parametersData.paraList.index(parameter)
    
    difference = (endP - startP)/3
    deltaTMax = 0.001

    for index in range(4):
        paraValue = startP + difference * index
        parametersData.args[paraIndex] = paraValue
        
        fig = plt.figure(index)

        #plt.title('{}: {}{}'.format(title, round(paraValue, 3), units))

        plt.xlabel('Membrane Potential (mV)')
        plt.ylabel('Fraction of Calcium Channels Open ($N$)')
        plotPhasePortraitData(function, parametersData, deltaTMax, tRange, dim)

        print('Graph {} complete!'.format(index + 1))
        
    plt.show()
        

def plotBifurcation(parametersData: ParametersData, parameter: str, startP: float, endP: float, stepP: float, function, tRange):
    paraIndex = parametersData.paraList.index(parameter)

    tInit = tRange[0]; tEnd = tRange[1]			        # The time start and end.
    numberOfPoints = 2000		                        # The number of time values values between this range.
    deltaTMax = 0.001
    tList = np.linspace(tInit, tEnd, numberOfPoints)

    N = int((endP - startP) / stepP)

    singleParamList = [[], []]
    singleList = [[], []]

    doubleParamList = []
    doubleTopList = []
    doubleBottomList = []

    singleIndex = 0
    
    for i in range(N + 1):
        paramValue = startP + i * stepP
        print('Trying to find equilbirium points for a parameter value of : {}'.format(round(paramValue,3)))
        parametersData.args[paraIndex] = paramValue

        sol, vMin, vMax, nMin, nMax, Vmin, Vmax, Nmin, Nmax = getPhasePortraitData(function, parametersData, deltaTMax, tRange, dim)

        if vMin is not None:
            if singleIndex == 0:
                singleIndex += 1
            doubleParamList.append(paramValue)
            doubleTopList.append(vMax)
            doubleBottomList.append(vMin)
            print('Dual Equilibrium Point Found\n')
        else:
            
            V = sol.T[0][len(sol.T[0]) - 1]
            singleParamList[singleIndex].append(paramValue)
            singleList[singleIndex].append(V)
            print('Single Equilibrium Point Found\n')

    plt.grid()
    plt.xlabel(r'$g_{Ca}$ mmho/cm$^2$')
    plt.ylabel('Membrane Potential (mV)')
    
    plt.plot(singleParamList[1], singleList[1], 'r')
    plt.plot(singleParamList[0], singleList[0], 'r')
    
    plt.plot([singleParamList[0][len(singleParamList[0]) - 1], doubleParamList[0]], [singleList[0][len(singleList[0]) - 1], doubleTopList[0]], 'g')
    plt.plot([singleParamList[0][len(singleParamList[0]) - 1], doubleParamList[0]], [singleList[0][len(singleList[0]) - 1], doubleBottomList[0]], 'g')
    
    plt.plot([singleParamList[1][0], doubleParamList[len(doubleParamList) - 1]], [singleList[1][0], doubleTopList[len(doubleBottomList) - 1]], 'g')
    plt.plot([singleParamList[1][0], doubleParamList[len(doubleParamList) - 1]], [singleList[1][0], doubleBottomList[len(doubleBottomList) - 1]], 'g')
    
    plt.plot(doubleParamList, doubleTopList, 'g')
    plt.plot(doubleParamList, doubleBottomList, 'g')
    
    plt.show()


if __name__ == '__main__':
    paraList = ['I', 'gl', 'Vl', 'gca', 'Vca', 'gk', 'Vk', 'lambdaM', 'lambdaN', 'V1', 'V2', 'V3', 'V4', 'C']

    I = 100
    gl = 1; gca = 3; gk = 5 # 3.5
    Vl = -50; Vca = 100; Vk = -70
    lambdaM = 1.0; lambdaN = 0.1
    V1 = 0; V2 = 10; V3 = 5; V4 = 5
    C = 20
    
    V0 = -30
    N0 = N_inf(V0, V3, V4)
    u0 = np.array([V0, N0])					# Removing the period initial condition
    deltaTMax = 0.01 				                # The step size

    args = np.array([I, gl, Vl, gca, Vca, gk, Vk, lambdaM, lambdaN, V1, V2, V3, V4, C])
    parametersData = ParametersData(paraList, args, u0)

    param = 'gca'
    title = 'Current'
    units = r'mmho/cm$^2$'

    # For bifurcation paramRange = [0.05, 4.5]
    
    paramRange = [0.05, 4.5]
    tRange = [0, 500]
    dim = 0
    #plotODEPhasePortrait(reducedSystem, parametersData, deltaTMax, tRange, dim)
    #plotSolveODEParameterChange(parametersData = parametersData, parameter = param, startP = paramRange[0], endP = paramRange[1], function = reducedSystem, tRange = [0, 250], title = title, units = units)
    #plotQuadPhasePotrait(parametersData = parametersData, parameter = param, startP = paramRange[0], endP = paramRange[1], function = reducedSystem, tRange = [0, 250], title = title, units = units, dim=dim)
    plotBifurcation(parametersData = parametersData, parameter = param, startP = paramRange[0], endP = paramRange[1], stepP = 0.05, function=reducedSystem, tRange=tRange)
