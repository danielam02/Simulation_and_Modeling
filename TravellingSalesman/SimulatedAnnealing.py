# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:21:02 2021

"""

import random
import math
import time

# reads a distance matrix (composed of a list of cities and the matrix itself)
# given file name fName
def readDistanceMatrix(fName):
    cities = []
    distances = []
    with open(fName, 'r') as f:
        i=0
        for line in f:
            if i == 0:
                l = [line.rstrip().split()]
                city = l[0][1]
                cities.append(city)
            else:
                row = []
                l = [line.rstrip().split()]
                city = l[0][0]
                cities.append(city)
                j = 1
                while j<= i:
                    row.append(l[0][j])
                    j += 1
                distances.append(row)
            i += 1
    f.close()   
    dm = []
    dm.append(cities)
    dm.append(distances)
    return dm

# creates a distance matrix given another, m, and a list containing a subset
# of the cities occurring in m
def createSmallMatrix(m,clist):
    cities = clist
    distances = []
    for c in range(1,len(cities)):
        row = []
        for v in range(0,c):
            row.append(distance(m,cities[c],cities[v]))      
        distances.append(row)  
    dm = []
    dm.append(cities)
    dm.append(distances)
    return dm

# creates a distance matrix given another, m, and a String, filter, containing 
# the initals of a subset of the cities occurring in m
def createFilterMatrix(m,filter):
    return createSmallMatrix(m,getCities(m,filter))
    
# returns the distance between two cities c1 and c2, given distance matrix m
def distance(m,c1,c2):
    index1 = m[0].index(c1)
    index2 = m[0].index(c2)
    if index1<index2:
        return int(m[1][index2-1][index1])
    else:
        return int(m[1][index1-1][index2])
        
# shows the distance matrix m
def showDistances(m):
    cities = '         '
    for i in range(0,len(m[0])-1):
        cities = cities + ' ' + "{:>9}".format(m[0][i])
    print(cities)
    for i in range(0,len(m[1])):
        row = "{:>9}".format(m[0][i+1])
        for j in range(0,len(m[1][i])):
            row = row + ' ' + "{:>9}".format(m[1][i][j])
        print(row)

# from a distance matrix m returns a list of all the cites of m
def getAllCities(m):
    return m[0]

# from a distance matrix m and a String filter returns a subset of cites of m
# with initials in filter 
def getCities(m,filter):	
    cityList = []
    for initial in filter:
        cityList.append(getCity(m[0],initial))
    return cityList
    
# from a list of cities cityList return the one with the first letter initial
def getCity(cityList,initial):	
    for city in cityList:
        if city[0] == initial:
            return city
    return None

# from a list of cities cityList return a String with their initials
def getInitials(cityList):
    initials = ""
    for city in cityList:
        initials += city[0]
    return initials

#returns number of cities
def numberOfCities(m):
    return len(m[0])


def vizinho(m, solucao):
    List = solucao
    n = len(List)
    r1 = random.randrange(0,n)
    r2 = random.randrange(0,n)
    while (r1 == r2 or abs(r1 - r2) == 1) or (r1 == 0 and r2 == len(List)-1) or (r2 == 0 and r1 == len(List)-1):
        r1 = random.randrange(0,n)
        r2 = random.randrange(0,n)
    if r1 > r2:
        r1, r2 = r2, r1
    
    if r2 == n-1:
        newD = distance(m, List[r2], List[r1-1]) + distance(m, List[r1], List[0]) 
        oldD = distance(m, List[r1], List[r1-1]) + distance(m, List[r2], List[0])
    else: 
        newD = distance(m, List[r2], List[r1-1]) + distance(m, List[r1], List[r2+1]) 
        oldD = distance(m, List[r1], List[r1-1]) + distance(m, List[r2], List[r2+1])
    diferenca = newD - oldD
    
    change = []
    n = 0 
    while n < r1:
        change.append(List[n])
        n = n + 1
    change.append(List[r2])
    n = r2 - 1
    while n >= r1:
        change.append(List[n])
        n = n - 1
    n = r2 + 1
    while n < len(List):
        change.append(List[n])
        n = n + 1
        
    vz = [change, diferenca] 
    
    return vz


def distancia_total(m, solucao):
    d = 0
    for i in range(0, len(solucao)-1):
        d = d + distance(m, solucao[i], solucao[i+1])
    d = d + distance(m, solucao[0], solucao[len(solucao)-1])
    return d


def funcao_avaliacao(matrix, s1, s2):
    return distancia_total(matrix, s2) - distancia_total(matrix, s1)


def maxdistance(m):
    Lista = []
    for i in m[1]:
        for d in i:
            d = int(d)
            Lista.append(d)
    M1 = max(Lista)
    Lista.remove(M1)
    M2 = max(Lista)
    m1 = min(Lista)
    Lista.remove(m1)
    m2 = min(Lista)
    return M1 + M2 - m1 - m2


def cria_solução_inicial(m):
    n = numberOfCities(m)    
    return random.sample(m[0], n)


def temperatura_inicial(m):
    dmax = maxdistance(m)
    return -dmax / math.log(0.9, math.e)


def decaimento(m, T, metodo):
    if metodo == "P":
        n = numberOfCities(m)
        if n < 5:
            return 0.7*T
        elif n >=5 and n < 20:
            return 0.8*T
        else:
            return 0.9*T
    
    elif metodo == "GR":
        return (T/(1 + 0.01*T))
    
    elif metodo == "GE":
        return 0.8*T


def n_iter_inicial(m):
    n = numberOfCities(m)
    return n*(n-1)/2


def var_n_iter(m, n_iter, metodo):
    if metodo == "P":
        return int(n_iter*1.01)
    
    elif metodo == "GE":
        return int(n_iter*1.01)
    
    elif metodo == "CT":
        return n_iter
    
    elif metodo == "AR":
        if numberOfCities(m) <= 10:
            return n_iter + 2
        else:
            return n_iter + 4
    
    
def criterio_de_paragem(m, criterio, numero, temperatura, n_iter, percentagem_aceites):
    if criterio == "P":
            n = numberOfCities(m) 
            if n < 5:
                return n_iter > 50
            elif n >= 5 and n < 13:
                return n_iter > 10000
            elif n >= 13 and n < 20:
                return n_iter > 20000
            elif n >= 20:
                return percentagem_aceites < 0.05
            
    elif criterio == "TM":
        return temperatura <= numero
    
    elif criterio == "TI":
        return n_iter >= numero
        
    elif criterio == "PA":
        return percentagem_aceites <= numero


def decisao(d, T):
    return random.random() < math.exp(-d/T)


def simulatedAnnealing(matrix, t_inicial, metodo_temperatura, metodo_iteracoes, criterio_paragem, numero):
    solucao_inicial = cria_solução_inicial(matrix)
    corrente = solucao_inicial
    melhor = corrente
    pior = corrente
    
    if t_inicial != 0:
        T = t_inicial
    else:
        T = temperatura_inicial(matrix)
    n_iter = n_iter_inicial(matrix)
    total_n_iter = 0
    aceites = 0
    percentagem_aceites = 1
    
    s_inicial = [solucao_inicial, total_n_iter, T, distancia_total(matrix, solucao_inicial)]
    s_corrente = [corrente, total_n_iter, T, distancia_total(matrix, corrente)]
    s_melhor = [melhor, total_n_iter, T, distancia_total(matrix, melhor)]
    s_pior = [pior, total_n_iter, T, distancia_total(matrix, pior)]
    
    while criterio_de_paragem(matrix, criterio_paragem, numero, T, total_n_iter, percentagem_aceites) is False:
        n = 1
        while n <= n_iter and (criterio_de_paragem(matrix, criterio_paragem, numero, T, total_n_iter, percentagem_aceites) is False):
            v = vizinho(matrix, corrente)
            proximo = v[0]
            d = v[1]
            
            if d < 0:
                aceites = aceites + 1
                corrente = proximo
                s_corrente = [corrente, total_n_iter, T, distancia_total(matrix, corrente)]
                
                if distancia_total(matrix, corrente) < distancia_total(matrix, melhor): 
                    melhor = corrente
                    s_melhor = [melhor, total_n_iter, T, distancia_total(matrix, melhor)]
                elif distancia_total(matrix, corrente) > distancia_total(matrix, pior):
                    pior = corrente
                    s_pior = [pior, total_n_iter, T, distancia_total(matrix, pior)]
            
            elif decisao(d, T) is True:
                aceites = aceites + 1
                corrente = proximo
                s_corrente = [corrente, total_n_iter, T, distancia_total(matrix, corrente)]
                
                if distancia_total(matrix, corrente) < distancia_total(matrix, melhor): 
                    melhor = corrente
                    s_melhor = [melhor, total_n_iter, T, distancia_total(matrix, melhor)]
                elif distancia_total(matrix, corrente) > distancia_total(matrix, pior):
                    pior = corrente
                    s_pior = [pior, total_n_iter, T, distancia_total(matrix, pior)]
                
            n = n + 1
            total_n_iter = total_n_iter + 1
          
        percentagem_aceites = aceites/total_n_iter
        T = decaimento(matrix, T, metodo_temperatura)    
        n_iter = var_n_iter(matrix, n_iter, metodo_iteracoes)
    
    retornar = [s_inicial, s_corrente, s_melhor, s_pior] 
    
    return retornar


def main():
    ficheiro = "matrix2.txt"

    matrix = readDistanceMatrix(ficheiro)
    
    string_iniciais = "JITGBOKVQZ"
        
    small_matrix = createFilterMatrix(matrix, string_iniciais)  
    
    # Critérios a selecionar
    
    t_inicial = 0
    temperatura = "P" # pode ser P - Predefinido ou GR - Gradual ou GE - Geometrico
    iteracoes = "P"   # pode ser P - Predefinido ou GE - Geometrico ou CT - Constante ou AR - Aritmetico
    paragem = "P"     # pode ser P - Predefinido ou TM - Temperatura Minima ou TI - Total de Iteracoes ou PA - Percentagem de Aceites
    numero = 0        # só é utilizado para o critério de paragem em TM ou TI ou PA
    
    start_time = time.time()    
    solucao = simulatedAnnealing(small_matrix, t_inicial, temperatura, iteracoes, paragem, numero)
    
    i = 0
    while i < 4:
        if i == 0:
            print("\nSolucao Inicial:")
        elif i == 1:
            print("\nSolucao Final:")
        elif i == 2:
            print("\nSolucao Melhor:")
        elif i == 3:
            print("\nSolucao Pior:")
        
        j = 0
        while j < 4:
            if j == 0:
                print("Caminho:", solucao[i][j])
            elif j == 1:
                print("Iteracao nº:", solucao[i][j])
            elif j == 2:
                print("Temperatura:", solucao[i][j])
            elif j == 3:
                print("Distancia do percurso:", solucao[i][j])
            j += 1
            
        i += 1
    
    end_time = time.time()
    print("\nRuntime:", end_time - start_time, "segundos")


# ALTERNATIVA: main de interacao com o utilizador

# def main():
    
#     ficheiro = input("Insira o nome do ficheiro: ")
    
#     matrix = readDistanceMatrix(ficheiro)
    
#     n = int(input("Insira o numero de cidades do problema: "))
#     print("\nInsira as cidades a considerar: ")
#     i = 0
#     string_iniciais = ''
#     while i < n:
#         c = input()
#         inicial = c[0].upper()
#         string_iniciais = string_iniciais + inicial
#         i = i + 1
#     small_matrix = createFilterMatrix(matrix, string_iniciais)

#     temperatura_inicial = input("\nDeseja definir a temperatura inicial (S - Sim, N - Nao)? ")
#     if temperatura_inicial == "N":
#           t_inicial = 0
#     elif temperatura_inicial == "S":
#           t_inicial = int(input("Qual a temperatura inicial? "))    
#     temperatura = input("Qual o método de decaimento da temperatura (P = Predefinido, GE = Geometrica, GR = Gradual)? ")
#     iteracoes = input("Qual o método de variacao de iteracoes por temperatura (P = Predefinido, CT = Constante, GE = Geometrica, AR = Aritmetica)? ")
#     paragem = input("Qual o critério de paragem (P = Predefinido, TM = Temperatura Minima, TI = Total Iteracoes, PA = Percentagem Solucoes Aceites)? ")    
#     if paragem == "TM":
#         numero = int(input("Qual a temperatura mínima a atingir? "))
#     elif paragem == "TI":
#         numero = int(input("Qual o número total de iteracoes a atingir? "))
#     elif paragem == "PA":
#         numero = int(input("Qual a percentagem mínima de solucoes aceites a atingir (de 0 a 1)? "))
#     elif paragem == "P":
#         numero = 0
     
#     start_time = time.time()
    
#     solucao = simulatedAnnealing(small_matrix, t_inicial, temperatura, iteracoes, paragem, numero)
    
#     i = 0
#     while i < 4:
#         if i == 0:
#             print("\nSolucao Inicial:")
#         elif i == 1:
#             print("\nSolucao Final:")
#         elif i == 2:
#             print("\nSolucao Melhor:")
#         elif i == 3:
#             print("\nSolucao Pior:")
        
#         j = 0
#         while j < 4:
#             if j == 0:
#                 print("Caminho:", solucao[i][j])
#             elif j == 1:
#                 print("Iteracao nº:", solucao[i][j])
#             elif j == 2:
#                 print("Temperatura:", solucao[i][j])
#             elif j == 3:
#                 print("Distancia do percurso:", solucao[i][j])
#             j += 1
            
#         i += 1
    
#     end_time = time.time()
#     print("\nRuntime:", end_time - start_time, "s")


main()    
    
    
    
    
    
    
