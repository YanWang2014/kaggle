#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.


# PassengerId  Survived  Pclass  Sex  Age SibSp Parch Fare Cabin Embarked  

def mydeap(mungedtrain):
    
    import operator
    import math
    import random
    
    import numpy
    
    from deap import algorithms
    from deap import base
    from deap import creator
    from deap import tools
    from deap import gp
    
    inputs = mungedtrain.iloc[:,2:10].values.tolist()
    outputs = mungedtrain['Survived'].values.tolist()
    
    # Define new functions
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1
    
    pset = gp.PrimitiveSet("MAIN", 8) # eight input
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2) # add more?
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-10,10)) # adjust?
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
    pset.renameArguments(ARG2='x3')
    pset.renameArguments(ARG3='x4')
    pset.renameArguments(ARG4='x5')
    pset.renameArguments(ARG5='x6')
    pset.renameArguments(ARG6='x7')
    pset.renameArguments(ARG7='x8')

    
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    def evalSymbReg(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the accuracy
        return sum(round(1.-(1./(1.+numpy.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs))/len(mungedtrain),
    
    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=21))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=21))
    
    
    
    
    random.seed(318)
    
    pop = toolbox.population(n=300) #
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 300, stats=mstats,
                                   halloffame=hof, verbose=True) #
    
    print(hof[0])
    func2 =toolbox.compile(expr=hof[0])
    return func2
