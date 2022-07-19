import pandas as pd

import numpy as np

import numpy.matlib 

from math import pow

import nengo

from sklearn.metrics import mean_squared_error

#from scipy.spatial import distance





from py4j.java_gateway import JavaGateway, GatewayParameters

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333))



calcProfile_app = gateway.entry_point

def calcProfile(inAe,inAp,inAa,inBe,inBp,inBa,inOe,inOp,inOa,outAe, outAp, outAa, outBe, outBp, outBa, outOe, outOp, outOa):

    value = calcProfile_app.calcProfile(inAe,inAp,inAa,inBe,inBp,inBa,inOe,inOp,inOa,outAe, outAp, outAa, outBe, outBp, outBa, outOe, outOp, outOa)

    #print(value)

    return value



# This python version is recreated in from Interact Java Program by @author Aarti Malhotra (aarti.malhotra@uwaterloo.ca)



# Function to calculate transients of actor, behavior and object once event has occurred 

# It solves Equation 11.15 for tau and Equation 11.3 for deflection in D. R. Heise, 2007. "Expressive Order: Confirming Sentiments in Social Actions". New York: Springer



def calcTau(modifierIdentifierActorEPA, behaviorEPA, modifierIdentifierObjectEPA):   

    

    inAe = modifierIdentifierActorEPA[0]

    inAp = modifierIdentifierActorEPA[1]

    inAa = modifierIdentifierActorEPA[2]

    inBe = behaviorEPA[0]

    inBp = behaviorEPA[1]

    inBa = behaviorEPA[2]

    inOe = modifierIdentifierObjectEPA[0]

    inOp = modifierIdentifierObjectEPA[1]

    inOa = modifierIdentifierObjectEPA[2]

    

    # Transients preceding the event

    # t = (1 Ae	Ap	Aa	Be	Bp	Ba	Oe	Op	Oa	AeBe	AeOp	ApBp	AaBa	BeOe	BeOp	BpOe	BpOp	AeBeOe	AeBeOp)

    # Note, the initial 1 in column0 is for constant, and starting from column10 is due to M coefficient matrix for ABOmale due to the true positions of index

    # e.g. last column is AeBeOp due to Z100100010 and considering inputs at index which have 1 in (Ae	Ap	Aa	Be	Bp	Ba	Oe	Op	Oa)

    t = np.array([1,inAe,inAp,inAa,inBe,inBp,inBa,inOe,inOp,inOa,inAe*inBe,inAe*inOp,inAp*inBp,inAa*inBa,inBe*inOe,inBe*inOp,inBp*inOe,inBp*inOp,inAe*inBe*inOe,inAe*inBe*inOp]) 



    # Coefficient matrix for ABOmale used from Interact

    M = np.array([[-0.26,-0.1,0.14,-0.19,0.06,0.11,-0.11,-0.37,0.02],

    [0.41,0,0.05,0.11,0,0.02,0,0,0],

    [0,0.56,0,0,0.16,-0.06,0,0,0],

    [0,0.06,0.64,0,0,0.27,0,0,0],

    [0.42,-0.07,-0.06,0.53,-0.13,0.04,0.11,0.18,0.02],

    [-0.02,0.44,0,0,0.7,0,0,-0.11,0],

    [-0.1,0,0.29,-0.12,0,0.64,0,0,0],

    [0.03,0.04,0,0,0.03,0,0.61,-0.08,0.03],

    [0.06,0,0,0.05,0.01,0,0,0.66,-0.05],

    [0,0,0,0,0,0,0.03,0.07,0.66],

    [0.05,0,0,0,0.01,0,0.03,0,0],

    [0.03,0,0,0,0,0,0,0,0],

    [0,-0.05,0,0,0,0,0,0,0],

    [0,0,-0.06,0,0,0,0,0,0],

    [0.12,0.01,0,0.11,0.03,0,0.04,0.03,0],

    [-0.05,0,0,-0.05,0,0,0,0.03,0],

    [-0.05,0,0,-0.02,0,0,-0.03,0,0],

    [0,0,0,0,0,0,0,-0.05,0],

    [0.03,0,0,0.02,0,0,0,0,0],

    [-0.02,0,0,0,0,0,0,0,0]])



    # Transients after the event

    tau = np.dot(t,M)



    outAe = tau[0]

    outAp = tau[1]

    outAa = tau[2]

    outBe = tau[3]

    outBp = tau[4]

    outBa = tau[5]

    outOe = tau[6]

    outOp = tau[7]

    outOa = tau[8]



    # Deflection for actor  

    deflectionA = pow((outAe-inAe), 2)+pow((outAp-inAp), 2)+pow((outAa-inAa), 2)



    # Deflection for object

    deflectionO = pow((outOe-inOe), 2)+pow((outOp-inOp), 2)+pow((outOa-inOa), 2)



    # Deflection overall

    deflectionTotal = deflectionA + pow((outBe-inBe), 2)+pow((outBp-inBp), 2)+pow((outBa-inBa), 2) + deflectionO



    #print("tau=",tau)

    print("deflectionTotal=",deflectionTotal)

    print("deflectionA=",deflectionA)

    print("deflectionO=",deflectionO)



    return tau, deflectionA, deflectionO, deflectionTotal, outAe, outAp, outAa, outBe, outBp, outBa, outOe, outOp, outOa;  





# Function to calculate emotion of actor or object once event has occurred 

# It solves Equation 14.3 in D. R. Heise, 2007. "Expressive Order: Confirming Sentiments in Social Actions". New York: Springer



def calcEmotion(inAe, inAp, inAa, inOe, inOp, inOa, outAe, outAp, outAa, outOe, outOp, outOa, AorO):   

    

    # Coefficient matrix for ABOmale from Interact

    M = np.array([[-0.36,-0.17,-0.22],

    [0.5,0,0],

    [0,0.32,0],

    [-0.23,0,0.44],

    [0.46,0,-0.05],

    [0,0.62,0],

    [0,0,0.66],

    [0.12,0,0.02],

    [0,0.05,0.03]])



    if AorO=='A':

        IReQe = np.array([[M[7][0]*inAe,0,M[8][0]*inAe],[M[7][1]*inAe,0,M[8][1]*inAe],[M[7][2]*inAe,0,M[8][2]*inAe]])

    else:

        IReQe = np.array([[M[7][0]*inOe,0,M[8][0]*inOe],[M[7][1]*inOe,0,M[8][1]*inOe],[M[7][2]*inOe,0,M[8][2]*inOe]])



    IRpQp = np.array([[0,0,0],[0,0,0],[0,0,0]])

    IRaQa = np.array([[0,0,0],[0,0,0],[0,0,0]])



    P = np.array([[0.5,0,-0.23],[0,0.32,0],[0,0,0.44]])

    R = np.array([[0.46,0,0],[0,0.62,0],[-0.05,0,0.66]])



    # Fundamental for Actor or Object

    if AorO=='A':

        r = np.array([inAe, inAp, inAa])

    else:

        r = np.array([inOe, inOp, inOa])



    # P + IRe*Qe + IRp*Qp + IRa*Qa

    P = P + IReQe + IRpQp + IRaQa

    #print("P=",P)



    # inv(P + IRe*Qe + IRp*Qp + IRa*Qa)

    Pinv = np.linalg.inv(P)

    #print("Pinv=",Pinv)



    # tau-d

    if AorO=='A':

        tau_d = np.array([outAe-M[0][0],outAp-M[0][1],outAa-M[0][2]]) 

    else:

        tau_d = np.array([outOe-M[0][0],outOp-M[0][1],outOa-M[0][2]]) 



    #print("tau_d=",tau_d)



    # Rr

    Rr = np.matmul(R,r)

    #print("Rr=",Rr)

    #print("r=",r)



    # tau-d-Rr or tau-Rr-d

    tau_d_Rr = tau_d - Rr

    #print("tau_d_Rr=",tau_d_Rr)  



    # inv(P + IRe*Qe + IRp*Qp + IRa*Qa)*(tau-Rr-d)

    emotionEPA = np.matmul(Pinv,tau_d_Rr)

    print("emotion EPA=",emotionEPA)

    

    distVector = calcDistVector(emotionEPA, 'Modifiers')



    #print(distVector)

    print("Min dist index: ", end ="")

    print(np.argmin(distVector))

    minDist = get_nth_key(np.argmin(distVector),'Modifiers')

    print(" Key: ", end ="")

    print(minDist)



    return minDist;  





# In[24]:





def calcCombinedIdentityModifierEPA(modifierEPA,identityEPA):

    # Coefficient matrix for ABOmale from Interact

    M_0 = np.array([-0.36,-0.17,-0.22])



    # Coefficient matrix for ABOmale from Interact

    M6_1 = np.array([[0.5,0,0],

    [0,0.32,0],

    [-0.23,0,0.44]])



    M6_2 = np.array([[0.46,0,-0.05],

    [0,0.62,0],

    [0,0,0.66]])



    M6_3 = np.array([[0.12,0,0.02],

    [0,0.05,0.03]])

    

    combinedEPA = np.array([0.0, 0.0, 0.0])

    combinedEPA[0] = M_0[0] + M6_1[0][0]*modifierEPA[0] + M6_1[1][0]*modifierEPA[1] + M6_1[2][0]*modifierEPA[2]

    combinedEPA[0] = combinedEPA[0] + M6_2[0][0]*identityEPA[0] + M6_2[1][0]*identityEPA[1] + M6_2[2][0]*identityEPA[2]

    combinedEPA[0] = combinedEPA[0] + M6_3[0][0]*identityEPA[0]*modifierEPA[0] + M6_3[1][0]*identityEPA[0]*modifierEPA[2]



    combinedEPA[1] = M_0[1] + M6_1[0][1]*modifierEPA[0] + M6_1[1][1]*modifierEPA[1] + M6_1[2][1]*modifierEPA[2]

    combinedEPA[1] = combinedEPA[1] + M6_2[0][1]*identityEPA[0] + M6_2[1][1]*identityEPA[1] + M6_2[2][1]*identityEPA[2]

    combinedEPA[1] = combinedEPA[1] + M6_3[0][1]*identityEPA[0]*modifierEPA[0] + M6_3[1][1]*identityEPA[0]*modifierEPA[2]



    combinedEPA[2] = M_0[2] + M6_1[0][2]*modifierEPA[0] + M6_1[1][2]*modifierEPA[1] + M6_1[2][2]*modifierEPA[2]

    combinedEPA[2] = combinedEPA[2] + M6_2[0][2]*identityEPA[0] + M6_2[1][2]*identityEPA[1] + M6_2[2][2]*identityEPA[2]

    combinedEPA[2] = combinedEPA[2] + M6_3[0][2]*identityEPA[0]*modifierEPA[0] + M6_3[1][2]*identityEPA[0]*modifierEPA[2]



    #print("combinedEPA: ",combinedEPA)



    return combinedEPA

  #[0.665408, 0.55376, 0.49194399999999994]





def calcOptNextBehavior(modifierActorEPA, identityActorEPA, behaviorEPA, modifierObjectEPA, identityObjectEPA):



    # combine modifierIdentifierActorEPA

    #print("Calculate modifierIdentifierActorEPA")

    modifierIdentifierActorEPA = calcCombinedIdentityModifierEPA(modifierActorEPA,identityActorEPA)



    # combine modifierIdentifierObjectEPA

    #print("Calculate modifierIdentifierObjectEPA")

    modifierIdentifierObjectEPA = calcCombinedIdentityModifierEPA(modifierObjectEPA,identityObjectEPA)



    # Testing calcTau for scenario from Interact: Mario[_,friend],give a raise to,Player[_,enemy]

    #print("Calling calcTau")

    tau, deflectionA, deflectionO, deflectionTotal, outAe, outAp, outAa, outBe, outBp, outBa, outOe, outOp, outOa = calcTau(modifierIdentifierActorEPA, behaviorEPA, modifierIdentifierObjectEPA)



    #profileValue = calcProfile(0.67,0.55,0.49, -1.40,1.62,1.50, 0.67,0.55,0.49,-0.88,1.02,0.96, -1.11,1.45,1.12, 0.06,-0.55,0.31)

    profileValue = calcProfile(modifierIdentifierActorEPA[0],modifierIdentifierActorEPA[1],modifierIdentifierActorEPA[2], behaviorEPA[0],behaviorEPA[1],behaviorEPA[2],modifierIdentifierObjectEPA[0],modifierIdentifierObjectEPA[1],modifierIdentifierObjectEPA[2],tau[0],tau[1],tau[2],tau[3],tau[4],tau[5],tau[6],tau[7],tau[8])



    #print('calcProfile output Actor Behavior| Object Behavior | Reidentified Actor | Reidentified Object:')

    #print('---------------------------------------------------------------------------------------------')

    #print(profileValue)

    parts = profileValue.split('|')

    objBehaviorEPA = np.array([float(parts[3]),float(parts[4]),float(parts[5])])

        

    print('Object Emotion:')

    print('--------------')

    calcEmotion(modifierIdentifierActorEPA[0],modifierIdentifierActorEPA[1],modifierIdentifierActorEPA[2], modifierIdentifierObjectEPA[0],modifierIdentifierObjectEPA[1],modifierIdentifierObjectEPA[2],outAe, outAp, outAa, outOe, outOp, outOa, 'O')

   

    print('Actor Emotion:')

    print('--------------')

    calcEmotion(modifierIdentifierActorEPA[0],modifierIdentifierActorEPA[1],modifierIdentifierActorEPA[2], modifierIdentifierObjectEPA[0],modifierIdentifierObjectEPA[1],modifierIdentifierObjectEPA[2],outAe, outAp, outAa, outOe, outOp, outOa, 'A')

 

    print('Object next behavior:')

    print('--------------')

    print(objBehaviorEPA)

    

    return objBehaviorEPA



def get_nth_key(n=0, criteria ='Behaviors'):

    

    if criteria == 'Behaviors':

        dictionary = np.load('C:\\Users\\malhoa\\BehaviorsDict.npy',allow_pickle='TRUE').item()

    elif criteria == 'Identities':

        dictionary = np.load('C:\\Users\\malhoa\\IdentitiesDict.npy',allow_pickle='TRUE').item()

    elif criteria == 'Modifiers':

        dictionary = np.load('C:\\Users\\malhoa\\ModifiersDict.npy',allow_pickle='TRUE').item()

    

    if n < 0:

        n += len(dictionary)

    for i, key in enumerate(dictionary.keys()):

        #print(key)

        if i == n:

            return key

    raise IndexError("dictionary index out of range")    





def get_epa_value_for_key(key, criteria ='Behaviors'):

    

    if criteria == 'Behaviors':

        dictionary = np.load('BehaviorsDict.npy',allow_pickle='TRUE').item()

    elif criteria == 'Identities':

        dictionary = np.load('IdentitiesDict.npy',allow_pickle='TRUE').item()

    elif criteria == 'Modifiers':

        dictionary = np.load('ModifiersDict.npy',allow_pickle='TRUE').item()

        

    for k,v in dictionary.items():

        if k == key:

            #print("Found value for: ", end ="")

            #print(key, end ="")

            #print(": ", end ="")

            #print(v)

            return v

    raise IndexError("dictionary does not have this key entry")    



def calcDistVector(epaValue, criteria ='Behaviors'):

  

    distVector = []

    epaDict = {}

        

    # create a map of behaviors epa 

    if criteria == 'Behaviors':

        epaDict = np.load('C:\\Users\\malhoa\\BehaviorsDict.npy',allow_pickle='TRUE').item()

    elif criteria == 'Identities':

        epaDict = np.load('C:\\Users\\malhoa\\IdentitiesDict.npy',allow_pickle='TRUE').item()

    elif criteria == 'Modifiers':

        epaDict = np.load('C:\\Users\\malhoa\\ModifiersDict.npy',allow_pickle='TRUE').item()



    # iterate and calculate array of 500 values which are euclidean distances of objBehaviorEPA from behaviors epa

    for k,v in epaDict.items():

        #print(k)

        distVector.append(np.linalg.norm(epaValue - v))

        

    #print('distVector:')

    #print('----------')

    #print(distVector)

    return distVector





# In[29]:





def calc15InputToBehaviorDistVector(modifierActorEPA, identityActorEPA, behaviorEPA, modifierObjectEPA, identityObjectEPA):

       

    # Test calcOptNextBehavior given 15 inputs

    objBehaviorEPA = calcOptNextBehavior(modifierActorEPA, identityActorEPA, behaviorEPA, modifierObjectEPA, identityObjectEPA)



    # Test calcDistVector

    distVector = calcDistVector(objBehaviorEPA, 'Behaviors')

    return distVector



def calc15PointInputToBehaviorDistVector(maE, maP, maA, iaE, iaP, iaA, bE, bP, bA, moE, moP, moA, ioE, ioP, ioA):

       

    modifierActorEPA = np.array([maE, maP, maA])

    identityActorEPA = np.array([iaE, iaP, iaA])

    behaviorEPA = np.array([bE, bP, bA])

    modifierObjectEPA = np.array([moE, moP, moA])

    identityObjectEPA = np.array([ioE, ioP, ioA])

    # Test calcOptNextBehavior given 15 inputs

    objBehaviorEPA = calcOptNextBehavior(modifierActorEPA, identityActorEPA, behaviorEPA, modifierObjectEPA, identityObjectEPA)



    # Test calcDistVector

    distVector = calcDistVector(objBehaviorEPA, 'Behaviors')

    return distVector

    

def testcalc15PointInputToBehaviorDistVector():

       

    modifierActorEPA = np.array([2.92,2.43,1.96])

    identityActorEPA = np.array([0.02,-0.09,-0.23])

    behaviorEPA = np.array([-1.40, 1.62, 1.50])

    modifierObjectEPA = np.array([2.92,2.43,1.96])

    identityObjectEPA = np.array([0.02,-0.09,-0.23])

    # Test calcOptNextBehavior given 15 inputs

    objBehaviorEPA = calcOptNextBehavior(modifierActorEPA, identityActorEPA, behaviorEPA, modifierObjectEPA, identityObjectEPA)



    # Test calcDistVector

    distVector = calcDistVector(objBehaviorEPA, 'Behaviors')

    return distVector

# In[30]:





#Try loading epa for a behavior

# epaBehaviorDict = createDictionary('Behaviors')

# epaModifiersDict = createDictionary('Modifiers')

# epaIdentitiesDict = createDictionary('Identities')



def calcNengoEvalPointsFromInputFile(filename = 'eval_scenarios.csv'):

    

    df = pd.read_csv(filename)

    print(df)

    

    nengoEvalPointsArr = np.array([])

    

    for index, row in df.iterrows():

     # access data using column names

      #  print(index, row['modA'], row['idA'], row['behavior'], row['modO'], row['idO'])

    

        modifierActorEPA = get_epa_value_for_key(row['modA'], 'Modifiers')

        identityActorEPA = get_epa_value_for_key(row['idA'], 'Identities')

        behaviorEPA = get_epa_value_for_key(row['behavior'], 'Behaviors')

        modifierObjectEPA = get_epa_value_for_key(row['modO'], 'Modifiers')

        identityObjectEPA = get_epa_value_for_key(row['idO'], 'Identities')



        evalPoint = np.array([modifierActorEPA,identityActorEPA,behaviorEPA,modifierObjectEPA,identityObjectEPA])



        nengoEvalPointsArr.append(identityObjectEPA)



    return nengoEvalPointsArr



def calcOneInteractionRound(modifierActor, identityActor, behavior, modifierObject, identityObject):

    modifierActorEPA = get_epa_value_for_key(modifierActor, 'Modifiers')

    identityActorEPA = get_epa_value_for_key(identityActor, 'Identities')

    behaviorEPA = get_epa_value_for_key(behavior, 'Behaviors')

    modifierObjectEPA = get_epa_value_for_key(modifierObject, 'Modifiers')

    identityObjectEPA = get_epa_value_for_key(identityObject, 'Identities')

    

    print(modifierActor, end ="") 

    print(" ", end ="") 

    print(identityActor, end ="") 

    print(" ", end ="") 

    print(behavior, end ="") 

    print(" ", end ="") 

    print(modifierObject, end ="") 

    print(" ", end ="") 

    print(identityObject)

    print(modifierActorEPA, end ="") 

    print(" ", end ="") 

    print(identityActorEPA, end ="") 

    print(" ", end ="") 

    print(behaviorEPA, end ="") 

    print(" ", end ="") 

    print(modifierObjectEPA, end ="") 

    print(" ", end ="") 

    print(identityObjectEPA)

    

    distVector = calc15InputToBehaviorDistVector(modifierActorEPA, identityActorEPA, behaviorEPA, modifierObjectEPA, identityObjectEPA)



    print("Min dist index: ", end ="")

    print(np.argmin(distVector))

    minDist = get_nth_key(np.argmin(distVector),'Behaviors')

    print(" Key: ", end ="")

    print(minDist)

    

model = nengo.Network(label='Basal Ganglia', seed = 3)

with model:

    print('Hello Nengo!')





def calcEPAError(expectedEPA,actualEPA):

    mse = mean_squared_error(expectedEPA, actualEPA)

    print(mse)

    

# def calcEPADist(expectedEPA,actualEPA):

#     mse = distance.euclidean(expectedEPA, actualEPA)

#     print(mse)



def calcEPADistNorm(expectedEPA,actualEPA):

    mse = np.linalg.norm(expectedEPA - actualEPA)

    print(mse)    

    

def calc15EPAError(expected15EPA,actual15EPA):

    mse = mean_squared_error(expected15EPA, actual15EPA)

    print(mse)



def calcNengoEvalPointsFromInputFile(filename = 'eval_scenarios.csv'):

    

    df = pd.read_csv(filename)

    print(df)

    

    nengoEvalPointsArr = []

    

    for index, row in df.iterrows():

     # access data using column names

      #  print(index, row['modA'], row['idA'], row['behavior'], row['modO'], row['idO'])

    

        modifierActorEPA = get_epa_value_for_key(row['modA'], 'Modifiers')

        identityActorEPA = get_epa_value_for_key(row['idA'], 'Identities')

        behaviorEPA = get_epa_value_for_key(row['behavior'], 'Behaviors')

        modifierObjectEPA = get_epa_value_for_key(row['modO'], 'Modifiers')

        identityObjectEPA = get_epa_value_for_key(row['idO'], 'Identities')



        evalPoint = [modifierActorEPA[0],modifierActorEPA[1],modifierActorEPA[2],

                     identityActorEPA[0],identityActorEPA[1],identityActorEPA[2],

                     behaviorEPA[0],behaviorEPA[1],behaviorEPA[2],

                     modifierObjectEPA[0],modifierObjectEPA[1],modifierObjectEPA[2],

                     identityObjectEPA[0],identityObjectEPA[1],identityObjectEPA[2]]

        print(evalPoint)

        

        nengoEvalPointsArr.append(evalPoint)



    return nengoEvalPointsArr

import nengo
import nengo.spa as spa
import numpy as np

D = 3  # the dimensionality of the vectors

behaviorsDict = np.load('C:\\Users\\malhoa\\BehaviorsDict.npy',allow_pickle='TRUE').item()
vocab_behaviors = spa.Vocabulary(dimensions=3)
for key, value in behaviorsDict.items():
    vocab_behaviors.add(key.upper().replace(' ', '_'), value)
    
modifiersDict = np.load('C:\\Users\\malhoa\\ModifiersDict.npy',allow_pickle='TRUE').item()
vocab_modifiers = spa.Vocabulary(dimensions=3)
for key, value in modifiersDict.items():
    vocab_modifiers.add(key.upper(), value)
    
identitiesDict = np.load('C:\\Users\\malhoa\\IdentitiesDict.npy',allow_pickle='TRUE').item()
vocab_identities = spa.Vocabulary(dimensions=3)
for key, value in identitiesDict.items():
    vocab_identities.add(key.upper(), value)    

model = spa.SPA()
with model:
     
    model.vision1 = spa.State(D, vocab=vocab_modifiers, neurons_per_dimension=200)
    model.vision2 = spa.State(D, vocab=vocab_identities, neurons_per_dimension=200)
    #for ens in model.vision2.all_ensembles:
        #ens.neuron_type = nengo.Direct()
        #print(ens, ens.n_neurons)
    model.vision3 = spa.State(D, vocab=vocab_behaviors, neurons_per_dimension=200)
    model.vision4 = spa.State(D, vocab=vocab_modifiers, neurons_per_dimension=200)
    model.vision5 = spa.State(D, vocab=vocab_identities, neurons_per_dimension=200)
    
    for ens in model.vision1.all_ensembles:
        ens.radius=4.3 #2*4.3*np.sqrt(3)
        
    for ens in model.vision2.all_ensembles:
        ens.radius=4.3 #2*4.3*np.sqrt(3)
        
    for ens in model.vision3.all_ensembles:
        ens.radius=4.3 #2*4.3*np.sqrt(3)
    
    for ens in model.vision4.all_ensembles:
        ens.radius=4.3 #2*4.3*np.sqrt(3)
        
    for ens in model.vision5.all_ensembles:
        ens.radius=4.3 #2*4.3*np.sqrt(3)

    #model.memory = spa.State(D, feedback=1)

    #nengo.Connection(model.vision1.output, model.memory.input,
    #                 transform=0.1)
                     
        
    def vision1_input(t):
        if t < 0.5:
            return 'HAPPY'
        else:
            return 'ANGRY'
        
    def vision2_input(t):
        if t < 0.5:
            return 'STRANGER'
        else:
            return 'STRANGER'
        
    # 'SELL_SOMETHING_TO'
    # 'CAPTURE'
    def vision3_input(t):
        if t < 0.5:
            return 'CAPTURE'
        else:
            return 'CAPTURE'       

    def vision4_input(t):
        if t < 0.5:
            return 'HAPPY'
        else:
            return 'HAPPY'

    def vision5_input(t):
        if t < 0.5:
            return 'STRANGER'
        else:
            return 'STRANGER'  
        
#    model.stim = spa.Input(vision1='HAPPY', vision2='STRANGER', vision3='CAPTURE', vision4='HAPPY', vision5='STRANGER')
    model.stim = spa.Input(vision1=vision1_input, vision2=vision2_input, vision3=vision3_input, 
                           vision4=vision4_input, vision5=vision5_input)
#  0  happy  stranger  sell something to  happy  stranger
# 1  angry  stranger  sell something to  happy  stranger
# 2  happy  stranger            capture  happy  stranger
# 3  angry  stranger            capture  happy  stranger
# [2.92, 2.43, 1.96, 0.02, -0.09, -0.23, 1.6, 1.47, 1.55, 2.92, 2.43, 1.96, 0.02, -0.09, -0.23]
# [-1.45, -0.3, 1.13, 0.02, -0.09, -0.23, 1.6, 1.47, 1.55, 2.92, 2.43, 1.96, 0.02, -0.09, -0.23]
# [2.92, 2.43, 1.96, 0.02, -0.09, -0.23, -1.4, 1.62, 1.5, 2.92, 2.43, 1.96, 0.02, -0.09, -0.23]
# [-1.45, -0.3, 1.13, 0.02, -0.09, -0.23, -1.4, 1.62, 1.5, 2.92, 2.43, 1.96, 0.02, -0.09, -0.23]

    
    p1 = nengo.Probe(model.vision1.output)
    p2 = nengo.Probe(model.vision2.output)
    p3 = nengo.Probe(model.vision3.output)
    p4 = nengo.Probe(model.vision4.output)
    p5 = nengo.Probe(model.vision5.output)
    
    #combined = nengo.Ensemble(n_neurons=500, dimensions=15)
    combined = nengo.Ensemble(n_neurons=1500, dimensions=15,radius=4.3*np.sqrt(15))#, neuron_type=nengo.Direct(),radius=4.3)
    nengo.Connection(model.vision1.output, combined[[0,1,2]])
    nengo.Connection(model.vision2.output, combined[[3,4,5]])
    nengo.Connection(model.vision3.output, combined[[6,7,8]])
    nengo.Connection(model.vision4.output, combined[[9,10,11]])
    nengo.Connection(model.vision5.output, combined[[12,13,14]])

    pC = nengo.Probe(combined)
    
    bg = nengo.networks.BasalGanglia(dimensions=500)

    maximum_distance = 5.0
    nengo.Connection(combined,bg.input, 
                     eval_points=np.array([[2.92, 2.43, 1.96, 0.02, -0.09, -0.23, -1.4, 1.62, 1.5, 2.92, 2.43, 1.96, 0.02, -0.09, -0.23]
                                          ,[-1.45, -0.3, 1.13, 0.02, -0.09, -0.23, -1.4, 1.62, 1.5, 2.92, 2.43, 1.96, 0.02, -0.09, -0.23]
                                          ]),
                      function=lambda x: np.interp(calc15PointInputToBehaviorDistVector(*x), [1.5, maximum_distance], [1, 0]),
                      scale_eval_points=False)
    
    def inhibit(t):
#         if t<0.5:
#             return [0]*500
#         else:
            value = [-10]*500 # default all inhibit
            value[68] = 0     # allow 'capture'
            value[408] = 0    # allow 'sell something to'
            return value
    
#     p_bginAll = nengo.Probe(bg.input)
#     p_bgoutAll = nengo.Probe(bg.output)
        
    inhibitor = nengo.Node(inhibit)
    nengo.Connection(inhibitor, bg.input)
            
    
    p_bgin = nengo.Probe(bg.input)    
    p_bgout = nengo.Probe(bg.output)
    
    thal = nengo.networks.Thalamus(dimensions=500)
    nengo.Connection(bg.output, thal.input)
    
    p_thal = nengo.Probe(thal.output)